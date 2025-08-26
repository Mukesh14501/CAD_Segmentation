# cadseg/models/builder.py
from __future__ import annotations
from typing import List, Tuple, Optional
import warnings

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except Exception:
    HAS_SMP = False

from cadseg.config import ModelConfig


# ----------------------------
# Public API
# ----------------------------
def build_model(cfg: ModelConfig) -> nn.Module:
    """
    Build a multi-label segmentation model with ImageNet-pretrained encoder.
    - arch: "unet++" | "unet" | "deeplabv3plus"
    - encoder: e.g., "resnet34"
    - num_classes: C (one channel per rule)
    - in_channels: typically 3
    - No activation in forward. Returns logits: (B, C, H, W).
    """
    if not HAS_SMP:
        raise ImportError(
            "segmentation_models_pytorch is not installed. "
            "Install it with: pip install segmentation-models-pytorch timm"
        )

    arch = cfg.arch.lower().strip()
    if arch in {"unet++", "unetpp", "nestedunet"}:
        net = smp.UnetPlusPlus(
            encoder_name=cfg.encoder,
            encoder_weights=cfg.encoder_weights,
            in_channels=cfg.in_channels,
            classes=cfg.num_classes,
        )
    elif arch == "unet":
        net = smp.Unet(
            encoder_name=cfg.encoder,
            encoder_weights=cfg.encoder_weights,
            in_channels=cfg.in_channels,
            classes=cfg.num_classes,
        )
    elif arch == "deeplabv3plus":
        net = smp.DeepLabV3Plus(
            encoder_name=cfg.encoder,
            encoder_weights=cfg.encoder_weights,
            in_channels=cfg.in_channels,
            classes=cfg.num_classes,
        )
    else:
        raise ValueError(f"Unknown arch '{cfg.arch}'. Use 'unet++', 'unet', or 'deeplabv3plus'.")

    # Optional, coarse decoder dropout (best-effort; decoder might not have dropout modules by default)
    if getattr(cfg, "dropout", 0) and cfg.dropout > 0:
        _inject_decoder_dropout(net, p=float(cfg.dropout))

    return SegModel(net=net, num_classes=cfg.num_classes)


class SegModel(nn.Module):
    """
    Thin wrapper around the SMP network that:
      - Returns raw logits (no activation) for training
      - Keeps interface stable if we later add aux heads
    """
    def __init__(self, net: nn.Module, num_classes: int):
        super().__init__()
        self.net = net
        self.num_classes = int(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            logits: (B, C, H, W)
        """
        return self.net(x)


# ----------------------------
# Utilities
# ----------------------------
def count_params(model: nn.Module) -> Tuple[int, int]:
    """Return (#total, #trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = trainable


def freeze_encoder_stages(model: nn.Module, stages: List[int]) -> None:
    """
    Freeze early encoder stages (good for tiny datasets).
    Stages for ResNet-like encoders in SMP:
        0: stem (conv1/bn1/relu/maxpool)
        1: layer1
        2: layer2
        3: layer3
        4: layer4
    For other encoders, best-effort mapping is applied if available.
    """
    enc = _get_smp_encoder(model)
    if enc is None:
        warnings.warn("Could not locate encoder; skip freezing.")
        return

    stage_modules = _resnet_like_stages(enc)
    if not stage_modules:
        warnings.warn("Unknown encoder structure; freezing entire encoder instead.")
        set_trainable(enc, False)
        return

    for s in stages:
        if 0 <= s < len(stage_modules):
            set_trainable(stage_modules[s], False)
        else:
            warnings.warn(f"Stage index {s} out of range [0..{len(stage_modules)-1}] for encoder.")


def unfreeze_all(model: nn.Module) -> None:
    set_trainable(model, True)


def split_encoder_decoder_params(model: nn.Module):
    """
    Split params into (encoder_params, decoder_params+head) for discriminative LR scheduling.
    """
    enc = _get_smp_encoder(model)
    if enc is None:
        # fallback: everything as decoder group
        return [], list(model.parameters())

    enc_ids = {id(p) for p in enc.parameters()}
    enc_params = []
    dec_params = []
    for p in model.parameters():
        if id(p) in enc_ids:
            enc_params.append(p)
        else:
            dec_params.append(p)
    return enc_params, dec_params


# ----------------------------
# Internals
# ----------------------------
def _get_smp_encoder(model: nn.Module) -> Optional[nn.Module]:
    # model may be SegModel wrapper
    net = model.net if hasattr(model, "net") else model
    # SMP models expose .encoder
    return getattr(net, "encoder", None)


def _resnet_like_stages(encoder: nn.Module) -> List[nn.Module]:
    """
    Try to decompose a ResNet-like encoder into ordered stages.
    Works for SMP ResNet encoders; best-effort for others.
    """
    parts = []
    # stem
    stem = []
    for name in ["conv1", "bn1", "relu", "maxpool"]:
        if hasattr(encoder, name):
            stem.append(getattr(encoder, name))
    if stem:
        parts.append(nn.Sequential(*stem))
    # residual layers
    for name in ["layer1", "layer2", "layer3", "layer4"]:
        if hasattr(encoder, name):
            parts.append(getattr(encoder, name))
    return parts


def _inject_decoder_dropout(net: nn.Module, p: float = 0.1) -> None:
    """
    Best-effort: insert dropout modules in decoder blocks if present.
    SMP decoders differ by arch; we scan for Conv2d sequences and add Dropout2d after activations.
    If nothing matches, we silently skip.
    """
    inserted = 0

    class _Wrap(nn.Sequential):
        def __init__(self, *mods):
            super().__init__(*mods)

    def _maybe_wrap(module: nn.Module) -> nn.Module:
        nonlocal inserted
        # Heuristic: if a module has Conv2d → ReLU/SiLU-like → Conv2d pattern,
        # append a Dropout2d after the activation.
        if isinstance(module, nn.Sequential) and len(module) >= 2:
            new_children = []
            for m in module:
                new_children.append(m)
                if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.SiLU, nn.GELU)):
                    new_children.append(nn.Dropout2d(p))
                    inserted += 1
            return _Wrap(*new_children)
        return module

    # Walk common attributes
    for attr in ["decoder", "segmentation_head"]:
        if hasattr(net, attr):
            mod = getattr(net, attr)
            for name, child in list(mod.named_children()):
                wrapped = _maybe_wrap(child)
                if wrapped is not child:
                    setattr(mod, name, wrapped)

    if inserted == 0:
        warnings.warn("Decoder dropout injection skipped (no suitable locations found).")
