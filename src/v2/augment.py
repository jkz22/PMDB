"""Phase 1: augmentation levels and input masking (torch, runs on GPU).

aug0: none. aug1: fixed-scale random crops + horizontal/vertical flips. No 90-degree rotations: the
images are cross-sections (rows = through-thickness z, columns = in-plane x), not top-down views.
aug2: aug1 + the SEM forward model of arXiv:2604.05960 with our own ranges:
    y = Poisson(dose * (a * (x conv h) + b)) / dose + Normal(0, sigma^2)
    h(r) = [2 J1(pi r) / (pi r)]^beta, r = sqrt((x'/Rx)^2 + (y'/Ry)^2), (x', y') rotated by theta.
Never uses random resized crops (particle size is a KPI).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F


@dataclass
class ForwardRanges:
    r_min: float = 0.5
    r_max: float = 3.0
    beta_min: float = 1.9
    beta_max: float = 2.0
    a_min: float = 0.8   # gain, overwritten from Phase 0 measurements
    a_max: float = 1.2
    b_min: float = -0.02  # offset / black level in [0,1] units, overwritten from Phase 0
    b_max: float = 0.05
    noise_min: float = 0.01  # total added noise sigma in [0,1] units, from Phase 0
    noise_max: float = 0.05
    poisson_frac_min: float = 0.3  # share of added noise variance that is shot noise
    poisson_frac_max: float = 1.0
    ksize: int = 15

    @classmethod
    def from_json(cls, path):
        return cls(**json.load(open(path)))

    def to_json(self, path):
        json.dump(asdict(self), open(path, "w"), indent=1)


def airy_psf(rx, ry, theta, beta, ksize: int) -> torch.Tensor:
    """Batch of elliptical Airy PSFs, shape (B, 1, k, k), each normalised to sum 1."""
    dev = rx.device
    c = (ksize - 1) / 2
    yy, xx = torch.meshgrid(torch.arange(ksize, device=dev) - c, torch.arange(ksize, device=dev) - c, indexing="ij")
    ct, st = torch.cos(theta)[:, None, None], torch.sin(theta)[:, None, None]
    xp = ct * xx + st * yy
    yp = -st * xx + ct * yy
    r = torch.sqrt((xp / rx[:, None, None]) ** 2 + (yp / ry[:, None, None]) ** 2)
    pr = math.pi * r
    j = torch.where(r < 1e-6, torch.ones_like(r), 2 * torch.special.bessel_j1(pr) / pr.clamp_min(1e-6))
    h = j.abs() ** beta[:, None, None]
    h = h / h.sum(dim=(1, 2), keepdim=True)
    return h[:, None]


def forward_model(x: torch.Tensor, fr: ForwardRanges, gen: torch.Generator | None = None) -> torch.Tensor:
    """x: (B, C, H, W) in [0, 1] (raw/255). Same nuisance per sample across its channels (co-registered stack)."""
    B, C, H, W = x.shape
    dev = x.device
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, device=dev, generator=gen)
    rx, ry = u(fr.r_min, fr.r_max), u(fr.r_min, fr.r_max)
    theta, beta = u(0.0, math.pi), u(fr.beta_min, fr.beta_max)
    h = airy_psf(rx, ry, theta, beta, fr.ksize)  # (B,1,k,k)
    p = fr.ksize // 2
    xr = F.pad(x.reshape(1, B * C, H, W), (p, p, p, p), mode="reflect")
    hk = h.repeat_interleave(C, 0)
    xb = F.conv2d(xr, hk, groups=B * C).reshape(B, C, H, W)
    a, b = u(fr.a_min, fr.a_max)[:, None, None, None], u(fr.b_min, fr.b_max)[:, None, None, None]
    mu = (a * xb + b).clamp_min(1e-4)
    s_tot = u(fr.noise_min, fr.noise_max)[:, None, None, None]
    f = u(fr.poisson_frac_min, fr.poisson_frac_max)[:, None, None, None]
    dose = mu.mean(dim=(1, 2, 3), keepdim=True) / (f * s_tot ** 2)
    y = torch.poisson(dose * mu, generator=gen) / dose
    y = y + torch.randn(y.shape, device=dev, generator=gen) * torch.sqrt((1 - f).clamp_min(0)) * s_tot
    return y.clamp(0, 1)


def flips(x: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    B = x.shape[0]
    fh = torch.rand(B, device=x.device, generator=gen) < 0.5
    fv = torch.rand(B, device=x.device, generator=gen) < 0.5
    x = torch.where(fh[:, None, None, None], x.flip(-1), x)
    return torch.where(fv[:, None, None, None], x.flip(-2), x)


def augment(x: torch.Tensor, level: str, fr: ForwardRanges | None = None, gen=None) -> torch.Tensor:
    if level == "aug0":
        return x
    x = flips(x, gen)
    if level == "aug2":
        x = forward_model(x, fr, gen)
    return x


def patch_mask(x: torch.Tensor, ratio: float, patch: int = 16, gen=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero a random `ratio` of non-overlapping patch x patch blocks. Returns (masked x, keep mask (B,1,H,W))."""
    if ratio <= 0:
        return x, torch.ones_like(x[:, :1])
    B, C, H, W = x.shape
    gh, gw = H // patch, W // patch
    n = gh * gw
    k = int(round(n * ratio))
    noise = torch.rand(B, n, device=x.device, generator=gen)
    idx = noise.argsort(dim=1)[:, :k]
    m = torch.ones(B, n, device=x.device)
    m.scatter_(1, idx, 0.0)
    m = m.reshape(B, 1, gh, gw).repeat_interleave(patch, 2).repeat_interleave(patch, 3)
    return x * m, m


def black_level_lift(x: torch.Tensor, delta: float) -> torch.Tensor:
    """Synthetic black-level lift used by selection metric 4: raise the floor without changing the top."""
    return (delta + (1 - delta) * x).clamp(0, 1)
