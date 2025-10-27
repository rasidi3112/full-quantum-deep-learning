from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml # type: ignore
@dataclass
class DataConfig:
    n_qubits: int = 6
    n_train: int = 4096
    n_val: int = 1024
    n_test: int = 1024
    batch_size: int = 32
    num_workers: int = 0
    shuffle: bool = True
    synthetic: bool = True
    dataset_path: Optional[str] = None
    cache_dir: str = "artifacts/cache"
@dataclass
class ModelConfig:
    n_qubits: int = 6
    circuit_layers: int = 4
    measurement_wires: List[int] = field(default_factory=lambda: [0, 1, 2])
    use_trainable_post_rotations: bool = True
    classical_head_width: int = 64
    output_dim: int = 2
    feature_map: str = "hybrid"
    ansatz: str = "strongly_entangling"
    shots: Optional[int] = None
    device: str = "lightning.qubit"

    def __post_init__(self) -> None:
        if self.n_qubits <= 1:
            raise ValueError("n_qubits must be greater than 1 for entanglement.")
        if any(w >= self.n_qubits or w < 0 for w in self.measurement_wires):
            raise ValueError("measurement_wires must be in range [0, n_qubits).")
        if len(set(self.measurement_wires)) != len(self.measurement_wires):
            raise ValueError("measurement_wires must be unique.")
@dataclass
class SchedulerConfig:
    type: Optional[str] = "cosine"
    min_lr: float = 1e-4
@dataclass
class EarlyStoppingConfig:
    patience: int = 10
    metric: str = "val_loss"
    mode: str = "min"
@dataclass
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    optimizer: str = "adamw"
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    mixed_precision: bool = False


@dataclass
class LoggingConfig:
    log_interval: int = 10
    checkpoint_dir: str = "artifacts/checkpoints"
    best_checkpoint: str = "artifacts/checkpoints/best.pt"
    enable_tensorboard: bool = False


@dataclass
class ProjectConfig:
    name: str = "full-quantum-deep-learning"


@dataclass
class ConfigBundle:
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig
    seed: int = 42


def load_config(path: Path | str) -> ConfigBundle:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw: Dict[str, Any] = yaml.safe_load(handle)

    project_cfg = ProjectConfig(**raw.get("project", {}))
    data_cfg = DataConfig(**raw.get("data", {}))
    model_cfg = ModelConfig(**raw.get("model", {}))

    sched_cfg = SchedulerConfig(**raw.get("training", {}).get("scheduler", {}))
    early_cfg = EarlyStoppingConfig(**raw.get("training", {}).get("early_stopping", {}))
    training_dict = {
        k: v
        for k, v in raw.get("training", {}).items()
        if k not in {"scheduler", "early_stopping"}
    }
    training_cfg = TrainingConfig(**training_dict, scheduler=sched_cfg, early_stopping=early_cfg)

    logging_cfg = LoggingConfig(**raw.get("logging", {}))
    seed = raw.get("seed", 42)

    return ConfigBundle(
        project=project_cfg,
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        logging=logging_cfg,
        seed=seed,
    )