from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np # type: ignore
import torch # type: ignore
from rich.console import Console # type: ignore

console = Console()
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
def save_checkpoint(state: Dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)
def load_checkpoint(path: str | Path, map_location: Optional[str | torch.device] = None) -> Dict:
    return torch.load(Path(path), map_location=map_location)
def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def format_metrics(metrics: Dict[str, float]) -> str:
    return " | ".join(f"{key}: {value:.4f}" for key, value in metrics.items())