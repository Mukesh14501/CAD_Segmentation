"""
Config loaders and typed dataclasses for dataset/model/train/aug/infer.
Fill with pydantic or dataclasses + YAML parsing.
"""

# cadseg/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml

# ---------------------------
# Dataclasses
# ---------------------------
@dataclass
class DatasetConfig:
    root: str
    images_dir: str
    masks_dir: str
    classes: List[str]
    C: Optional[int] = None
    tile_size: int = 1024
    tile_overlap: int = 256
    num_workers: int = 4
    seed: int = 42
    folds: int = 1
    min_positive_area: int = 64

    # resolved at runtime (not in YAML)
    root_path: Path = field(init=False)
    images_path: Path = field(init=False)
    masks_path: Path = field(init=False)

    def resolve_paths(self, base: Optional[Path] = None) -> None:
        """Resolve root/images/masks to absolute Paths."""
        if base is None:
            base = Path(".").resolve()
        self.root_path = (base / self.root).resolve()
        self.images_path = (self.root_path / self.images_dir).resolve()
        self.masks_path = (self.root_path / self.masks_dir).resolve()

@dataclass
class ModelConfig:
    arch: str = "unet++"
    encoder: str = "resnet34"
    encoder_weights: Optional[str] = "imagenet"
    in_channels: int = 3
    num_classes: int = 1
    dropout: float = 0.1
    use_edge_aux: bool = False

@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 4
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    sched: str = "onecycle"  # onecycle|cosine|reduce_on_plateau
    amp: bool = True
    grad_clip: float = 1.0
    freeze_encoder_stages: List[int] = field(default_factory=lambda: [0, 1])
    unfreeze_at_epoch: int = 2
    loss: str = "dice_bce"   # dice_bce|dice_focal|tversky|dice_ce
    focal_gamma: float = 2.0
    tversky_alpha: float = 0.7
    tversky_beta: float = 0.3
    class_weights: Optional[List[float]] = None
    save_metric: str = "macro_mIoU"
    early_stop_patience: int = 8

@dataclass
class AugConfig:
    train: Dict[str, Any] = field(default_factory=dict)
    valid: Dict[str, Any] = field(default_factory=dict)
    infer: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InferConfig:
    tta: List[str] = field(default_factory=lambda: ["hflip", "vflip"])
    merge: str = "mean"  # mean|gmean|max
    tile_size: int = 1024
    tile_overlap: int = 256
    min_component_area: Optional[List[int]] = None
    thresholds_file: str = "configs/thresholds.json"

@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    train: TrainConfig
    aug: AugConfig
    infer: InferConfig
    cfg_dir: Path

# ---------------------------
# YAML helpers
# ---------------------------
def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _coerce_dataset(d: Dict[str, Any]) -> DatasetConfig:
    return DatasetConfig(**d)

def _coerce_model(d: Dict[str, Any]) -> ModelConfig:
    return ModelConfig(**d)

def _coerce_train(d: Dict[str, Any]) -> TrainConfig:
    return TrainConfig(**d)

def _coerce_aug(d: Dict[str, Any]) -> AugConfig:
    return AugConfig(**d)

def _coerce_infer(d: Dict[str, Any]) -> InferConfig:
    return InferConfig(**d)

# ---------------------------
# Public API
# ---------------------------
def load_configs(cfg_dir: str | Path = "configs") -> Config:
    """Load and validate all config files from a directory."""
    cfg_dir = Path(cfg_dir).resolve()

    ds = _coerce_dataset(_load_yaml(cfg_dir / "dataset.yaml"))
    md = _coerce_model(_load_yaml(cfg_dir / "model.yaml"))
    tr = _coerce_train(_load_yaml(cfg_dir / "train.yaml"))
    ag = _coerce_aug(_load_yaml(cfg_dir / "aug.yaml"))
    inf = _coerce_infer(_load_yaml(cfg_dir / "infer.yaml"))

    # Resolve paths and basic validations
    ds.resolve_paths(base=cfg_dir.parent)

    # Consistency: classes and channel counts
    if not ds.classes or not isinstance(ds.classes, list):
        raise ValueError("dataset.yaml must include a non-empty 'classes' list.")
    if ds.C is None or ds.C == 0:
        ds.C = len(ds.classes)
    if ds.C != len(ds.classes):
        raise ValueError(f"dataset.C ({ds.C}) does not match len(classes) ({len(ds.classes)}).")
    if md.num_classes != ds.C:
        # keep dataset as source of truth; adjust model
        md.num_classes = ds.C
        print(f"[config] Adjusted model.num_classes to dataset.C = {ds.C}")

    # Existence checks (directories)
    for pth, name in [(ds.root_path, "dataset.root"),
                      (ds.images_path, "dataset.images_dir"),
                      (ds.masks_path, "dataset.masks_dir")]:
        if not pth.exists():
            print(f"[warn] {name} does not exist yet: {pth}")

    return Config(dataset=ds, model=md, train=tr, aug=ag, infer=inf, cfg_dir=cfg_dir)
