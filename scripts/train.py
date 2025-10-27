from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro # type: ignore

from qdl.config import load_config
from qdl.data import create_dataloaders
from qdl.model import QuantumDeepLearningModel
from qdl.trainer import QDLTrainer
from qdl.utils import console, count_trainable_parameters, set_seed


@dataclass
class CLIArgs:
    config: Path = Path("configs/default.yaml")
    resume: bool = False
    checkpoint: Optional[Path] = None


def main(args: CLIArgs) -> None:
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    model = QuantumDeepLearningModel(cfg.model)

    console.rule(f"[bold blue]{cfg.project.name}")
    console.log(f"Trainable parameters: {count_trainable_parameters(model):,}")

    trainer = QDLTrainer(model, cfg)

    if args.resume and args.checkpoint is not None:
        trainer.load_checkpoint(args.checkpoint)

    trainer.fit(train_loader, val_loader)
    trainer.load_best_checkpoint()
    test_loss, test_metrics = trainer.evaluate(test_loader, split="test", log=True)
    console.rule(f"[bold green]Test loss: {test_loss:.4f}")
    console.log(f"[bold green]{test_metrics}")


if __name__ == "__main__":
    main(tyro.cli(CLIArgs))