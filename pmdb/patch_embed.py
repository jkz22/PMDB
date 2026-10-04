"""Frozen NASA MicroNet ResNet50 patch feature extractor (torch imported lazily)."""

from __future__ import annotations

import numpy as np

WEIGHTS_URL = ("https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/"
               "resnet50_pretrained_microscopynet_v1.1.pth.tar")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FEAT_DIM = 1536


def remap_micronet_state_dict(obj: dict) -> dict:
    """Unwrap 'state_dict', strip leading 'module.'/'encoder.', drop 'fc.*'."""
    sd = obj["state_dict"] if "state_dict" in obj else obj
    out = {}
    for k, v in sd.items():
        changed = True
        while changed:
            changed = False
            for pre in ("module.", "encoder."):
                if k.startswith(pre):
                    k = k[len(pre):]
                    changed = True
        if k.startswith("fc."):
            continue
        out[k] = v
    return out


def build_backbone(weights_path: str | None, device: str):
    import torch
    import torchvision

    model = torchvision.models.resnet50(weights=None)
    if weights_path:
        obj = torch.load(weights_path, map_location="cpu", weights_only=False)
        res = model.load_state_dict(remap_micronet_state_dict(obj), strict=False)
        missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
        if any(not k.startswith("fc.") for k in missing) or unexpected:
            raise RuntimeError(f"MicroNet weights mismatch: missing={missing}, unexpected={unexpected}")
    return model.eval().to(device)


def patch_features(model, gray: np.ndarray, coords: np.ndarray, device: str, batch_size: int = 64) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    outs = []
    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            crops = np.stack([gray[y0:y0 + 224, x0:x0 + 224] for _, _, y0, x0 in coords[i:i + batch_size]])
            x = torch.from_numpy(np.ascontiguousarray(crops, dtype=np.float32)).to(device).unsqueeze(1)
            x = (x.repeat(1, 3, 1, 1) - mean) / std
            x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
            x = model.layer1(x)
            l2 = model.layer2(x)
            l3 = model.layer3(l2)
            l3u = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
            f = F.avg_pool2d(torch.cat([l2, l3u], 1), 3, stride=1, padding=1).mean(dim=(2, 3))
            outs.append(f.float().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)
