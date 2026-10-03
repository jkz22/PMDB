"""Phase 2 representation families.

Every model exposes ``loss(batch) -> (loss, logs)`` for training and ``embed(x) -> (B, D)``
for evaluation. Inputs are (B, 3, 256, 256) float in [0, 1].
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.v2.augment import patch_mask

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def imnet(x):
    return (x - IMNET_MEAN.to(x)) / IMNET_STD.to(x)


# ----------------------------------------------------------------------------- VAEs
def _down(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.GroupNorm(8, o), nn.SiLU(), nn.Conv2d(o, o, 3, 1, 1),
                         nn.GroupNorm(8, o), nn.SiLU())


def _up(i, o):
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.GroupNorm(8, o), nn.SiLU(), nn.Conv2d(o, o, 3, 1, 1),
                         nn.GroupNorm(8, o), nn.SiLU())


class VAE(nn.Module):
    """variant: 'A' plain, 'B' KPI-conditioned (CVAE: encoder + decoder see KPIs),
    'C' auxiliary KPI regression head on mu."""

    CH = (32, 64, 128, 256, 256)

    def __init__(self, variant="A", zdim=128, n_kpi=0, mask_ratio=0.0, beta=1.0, aux_weight=10.0):
        super().__init__()
        self.variant, self.zdim, self.mask_ratio, self.beta, self.aux_w = variant, zdim, mask_ratio, beta, aux_weight
        self.n_cond = 2 * n_kpi if variant == "B" else 0  # value + missing indicator
        ch = (3,) + self.CH
        self.enc = nn.Sequential(*[_down(ch[i], ch[i + 1]) for i in range(len(self.CH))])
        feat = self.CH[-1] * 8 * 8
        self.to_stats = nn.Linear(feat + self.n_cond, 2 * zdim)
        self.from_z = nn.Linear(zdim + self.n_cond, feat)
        rc = self.CH[::-1] + (32,)
        self.dec = nn.Sequential(*[_up(rc[i], rc[i + 1]) for i in range(len(self.CH))])
        self.out = nn.Conv2d(32, 3, 3, 1, 1)
        self.aux = nn.Sequential(nn.Linear(zdim, 128), nn.SiLU(), nn.Linear(128, n_kpi)) if variant == "C" else None

    @staticmethod
    def _cond(kpi):
        miss = torch.isnan(kpi)
        return torch.cat([torch.nan_to_num(kpi, 0.0), miss.float()], 1)

    def encode(self, x, kpi=None):
        h = self.enc(x).flatten(1)
        if self.n_cond:
            h = torch.cat([h, self._cond(kpi)], 1)
        mu, logvar = self.to_stats(h).chunk(2, 1)
        return mu, logvar.clamp(-10, 10)

    def decode(self, z, kpi=None):
        if self.n_cond:
            z = torch.cat([z, self._cond(kpi)], 1)
        h = self.from_z(z).view(-1, self.CH[-1], 8, 8)
        return torch.sigmoid(self.out(self.dec(h)))

    def forward(self, x, kpi=None, sample=True):
        mu, logvar = self.encode(x, kpi)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
        return self.decode(z, kpi), mu, logvar

    def loss(self, batch):
        x, kpi = batch["x"], batch.get("kpi")
        xin = patch_mask(x, self.mask_ratio)[0] if self.mask_ratio > 0 else x
        rec, mu, logvar = self(xin, kpi)
        mse = F.mse_loss(rec.float(), x.float(), reduction="sum") / x.shape[0]
        kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)).mean()
        loss = mse + self.beta * kl
        logs = {"mse_px": mse.item() / x[0].numel(), "kl": kl.item()}
        if self.aux is not None:
            pred, ok = self.aux(mu), ~torch.isnan(kpi)
            a = F.mse_loss(pred[ok].float(), kpi[ok].float()) if ok.any() else pred.sum() * 0
            loss = loss + self.aux_w * a * x[0].numel() / 1000.0
            logs["aux"] = a.item()
        return loss, logs

    @torch.no_grad()
    def embed(self, x, kpi=None):
        return self.encode(x, kpi)[0]

    @torch.no_grad()
    def reconstruct(self, x, kpi=None):
        return self(x, kpi, sample=False)[0]


# ----------------------------------------------------------------------------- MAEs
class MAE(nn.Module):
    """'adapted': continue facebook/vit-mae-base; 'scratch': ViT-S/16 MAE from random init.
    256-px crops are fed natively (position embeddings interpolated), never resized."""

    def __init__(self, init="adapted", mask_ratio=0.75):
        super().__init__()
        from transformers import ViTMAEConfig, ViTMAEForPreTraining
        if init == "adapted":
            self.m = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base", mask_ratio=mask_ratio)
            self.interp = True
        else:
            cfg = ViTMAEConfig(image_size=256, patch_size=16, hidden_size=384, num_hidden_layers=12,
                               num_attention_heads=6, intermediate_size=1536, decoder_hidden_size=256,
                               decoder_num_hidden_layers=4, decoder_num_attention_heads=8,
                               decoder_intermediate_size=1024, mask_ratio=mask_ratio, norm_pix_loss=True)
            self.m = ViTMAEForPreTraining(cfg)
            self.interp = False

    def loss(self, batch):
        out = self.m(pixel_values=imnet(batch["x"]), interpolate_pos_encoding=self.interp)
        return out.loss, {"mae": out.loss.item()}

    @torch.no_grad()
    def embed(self, x):
        vit = self.m.vit
        old = vit.config.mask_ratio
        vit.config.mask_ratio = 0.0
        try:
            h = vit(pixel_values=imnet(x), interpolate_pos_encoding=self.interp).last_hidden_state
        finally:
            vit.config.mask_ratio = old
        return h[:, 1:].mean(1)


# ----------------------------------------------------------------------------- DINOv2
DINO_CROP = 252  # largest multiple of 14 inside 256: centre crop, no resize


def dino_crop(x):
    o = (x.shape[-1] - DINO_CROP) // 2
    return x[..., o:o + DINO_CROP, o:o + DINO_CROP]


def dino_feats(model, x):
    h = model(pixel_values=imnet(dino_crop(x))).last_hidden_state
    return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)


class DinoFT(nn.Module):
    """DINOv2 ViT-S/14, last ``n_train`` blocks fine-tuned with a two-view invariance loss
    (BYOL-style: online projector+predictor vs EMA teacher, symmetric cosine)."""

    def __init__(self, n_train=2, ema=0.996):
        super().__init__()
        from transformers import Dinov2Model
        self.student = Dinov2Model.from_pretrained("facebook/dinov2-small")
        for p in self.student.parameters():
            p.requires_grad = False
        for blk in self.student.encoder.layer[-n_train:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.student.layernorm.parameters():
            p.requires_grad = True
        d = 2 * self.student.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(d, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Linear(1024, 256))
        self.pred = nn.Sequential(nn.Linear(256, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Linear(1024, 256))
        self.teacher = copy.deepcopy(self.student).requires_grad_(False)
        self.t_proj = copy.deepcopy(self.proj).requires_grad_(False)
        self.ema = ema

    def loss(self, batch):
        v1, v2 = batch["x"], batch["x2"]
        p1, p2 = self.pred(self.proj(dino_feats(self.student, v1))), self.pred(self.proj(dino_feats(self.student, v2)))
        with torch.no_grad():
            t1, t2 = self.t_proj(dino_feats(self.teacher, v1)), self.t_proj(dino_feats(self.teacher, v2))
        loss = (2 - F.cosine_similarity(p1, t2.detach(), -1).mean() - F.cosine_similarity(p2, t1.detach(), -1).mean())
        return loss, {"byol": loss.item()}

    @torch.no_grad()
    def after_step(self):
        for s, t in ((self.student, self.teacher), (self.proj, self.t_proj)):
            for ps, pt in zip(s.parameters(), t.parameters()):
                pt.mul_(self.ema).add_(ps.detach(), alpha=1 - self.ema)

    @torch.no_grad()
    def embed(self, x):
        return dino_feats(self.student, x)


# ----------------------------------------------------------------------------- off-the-shelf
MICRONET_URL = ("https://huggingface.co/jstuckner/microscopy-efficientnet-b4-imagenet-micronet/resolve/main/"
                "efficientnet-b4_imagenet-micronet_weights.pth")


class OffTheShelf(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        if name == "dinov2":
            from transformers import Dinov2Model
            self.m = Dinov2Model.from_pretrained("facebook/dinov2-small")
        elif name == "vitmae":
            from transformers import ViTMAEModel
            self.m = ViTMAEModel.from_pretrained("facebook/vit-mae-base", mask_ratio=0.0)
        elif name == "micronet":
            import segmentation_models_pytorch as smp
            self.m = smp.encoders.get_encoder("efficientnet-b4", weights=None)
            sd = torch.hub.load_state_dict_from_url(MICRONET_URL, map_location="cpu", progress=False)
            sd = {k: v for k, v in sd.items() if not k.startswith("_fc.")}  # ImageNet classifier head, unused
            missing, unexpected = nn.Module.load_state_dict(self.m, sd, strict=False)
            if len(missing) > 0 or len(unexpected) > 0:
                raise RuntimeError(f"MicroNet weights mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        else:
            raise ValueError(name)
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def embed(self, x):
        if self.name == "dinov2":
            return dino_feats(self.m, x)
        if self.name == "vitmae":
            return self.m(pixel_values=imnet(x), interpolate_pos_encoding=True).last_hidden_state[:, 1:].mean(1)
        return self.m(imnet(x))[-1].mean((2, 3))


TRAINABLE = ("vae_a", "vae_b", "vae_c", "mae_adapted", "mae_scratch", "dino_ft")
OFF_THE_SHELF = ("ots_dinov2", "ots_vitmae", "ots_micronet")


def build(family: str, n_kpi: int = 0, vae_mask: float = 0.0, mae_mask: float = 0.75) -> nn.Module:
    if family.startswith("vae_"):
        return VAE(family[-1].upper(), n_kpi=n_kpi, mask_ratio=vae_mask)
    if family == "mae_adapted":
        return MAE("adapted", mae_mask)
    if family == "mae_scratch":
        return MAE("scratch", mae_mask)
    if family == "dino_ft":
        return DinoFT()
    if family.startswith("ots_"):
        return OffTheShelf(family[4:])
    raise ValueError(family)


def n_params(m: nn.Module, trainable_only=True) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)


