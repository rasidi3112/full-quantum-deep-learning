from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch  # type: ignore
from torch import nn  # type: ignore
from torch.optim import Optimizer  # type: ignore
from torch.optim import lr_scheduler  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from tqdm.auto import tqdm  # type: ignore
import pennylane as qml  # type: ignore
import wandb  # type: ignore

from .config import ConfigBundle
from .metrics import compute_classification_metrics
from .utils import console, ensure_dir, format_metrics, save_checkpoint


class QDLTrainer:
    """Training loop profesional untuk model QDL dengan W&B, QNG, dan Continual Learning."""

    def __init__(self, model: nn.Module, cfg: ConfigBundle) -> None:
        self.model = model
        self.cfg = cfg
        self.training_cfg = cfg.training
        self.logging_cfg = cfg.logging

        self.device = torch.device("cpu")
        self.model.to(self.device)

        self.criterion = nn.NLLLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        mode = getattr(getattr(self.training_cfg, "early_stopping", {}), "mode", "min").lower()
        self.best_metric_value = float("inf") if mode == "min" else float("-inf")
        self.early_stop_counter = 0
        self._validation_available = False

        self.continual_task = 0  # 🔹 Untuk continual learning

        ensure_dir(getattr(self.logging_cfg, "checkpoint_dir", "artifacts/checkpoints"))

        # ------------------- W&B Init -------------------
        wandb.init(
            project=getattr(self.logging_cfg, "project_name", "full-quantum-dl"),
            entity=getattr(self.logging_cfg, "wandb_entity", None),
            config={
                "lr": getattr(self.training_cfg, "learning_rate", 1e-3),
                "batch_size": getattr(self.training_cfg, "batch_size", 32),
                "optimizer": getattr(self.training_cfg, "optimizer", "adam"),
                "scheduler": getattr(getattr(self.training_cfg, "scheduler", {}), "type", None),
                "n_qubits": getattr(self.model, "n_qubits", None),
                "feature_strategy": getattr(self.model, "feature_strategy", None),
            },
        )

    # ---------------- Optimizer & Scheduler ----------------
    def _build_optimizer(self) -> Optimizer:
        opt_name = getattr(self.training_cfg, "optimizer", "adam").lower()
        lr = getattr(self.training_cfg, "learning_rate", 1e-3)
        wd = getattr(self.training_cfg, "weight_decay", 0.0)
        if opt_name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        if opt_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        if opt_name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        raise ValueError(f"Unsupported optimizer {opt_name}")

    def _build_scheduler(self):
        sched_cfg = getattr(self.training_cfg, "scheduler", {})
        sched_type = (getattr(sched_cfg, "type", "none") or "none").lower()
        if sched_type in {"none", "off"}:
            return None
        if sched_type == "cosine":
            return lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=getattr(self.training_cfg, "epochs", 100),
                eta_min=getattr(sched_cfg, "min_lr", 0.0),
            )
        if sched_type == "plateau":
            return lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=getattr(sched_cfg, "min_lr", 0.0),
            )
        raise ValueError(f"Unsupported scheduler type: {sched_type}")

    # ---------------- Training Loop ----------------
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        self._validation_available = val_loader is not None
        console.rule(f"[bold cyan]Training Task {self.continual_task}")

        for epoch in range(1, getattr(self.training_cfg, "epochs", 10) + 1):
            train_loss = self._train_one_epoch(train_loader, epoch)

            val_loss, val_metrics = None, None
            log_message = f"[Epoch {epoch:03d}] train_loss: {train_loss:.4f}"

            if val_loader is not None:
                val_loss, val_metrics = self.evaluate(val_loader, split="val", log=False)
                log_message += f" | val_loss: {val_loss:.4f} | {format_metrics(val_metrics)}"
                self._maybe_checkpoint(val_loss, val_metrics)
                if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
            else:
                if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(train_loss)

            console.log(log_message)
            wandb.log(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss or 0.0, **(val_metrics or {})}
            )

            if self._should_stop():
                console.log("[red]Early stopping triggered.[/red]")
                break

            if self.scheduler and not isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()

    # ---------------- Single Epoch ----------------
    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        progress = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
        for step, (inputs, targets) in enumerate(progress, start=1):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            log_probs = self.model(inputs)
            loss = self.criterion(log_probs, targets)
            loss.backward()

            clip_norm = getattr(self.training_cfg, "grad_clip_norm", 0.0)
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
            self.optimizer.step()
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            log_interval = getattr(self.logging_cfg, "log_interval", 10)
            if step % log_interval == 0:
                progress.set_postfix({"loss": loss.item()})

        progress.close()
        return total_loss / max(total_samples, 1)

    # ---------------- Evaluation ----------------
    def evaluate(self, loader: DataLoader, split: str = "val", log: bool = True) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_log_probs = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                log_probs = self.model(inputs)
                loss = self.criterion(log_probs, targets)

                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                all_log_probs.append(log_probs)
                all_targets.append(targets)

        mean_loss = total_loss / max(total_samples, 1)
        metrics = compute_classification_metrics(torch.cat(all_log_probs), torch.cat(all_targets))

        if log:
            console.log(f"[{split}] loss: {mean_loss:.4f} | {format_metrics(metrics)}")

        return mean_loss, metrics

    # ---------------- Continual Learning ----------------
    def next_task(self, new_train_loader: DataLoader, new_val_loader: Optional[DataLoader] = None):
        """Switch ke task baru, freeze quantum layer untuk transfer learning."""
        self.continual_task += 1
        console.log(f"[cyan]Switching to Task {self.continual_task}[/cyan]")

        # Freeze quantum layer untuk transfer learning
        if hasattr(self.model, "freeze_quantum_layer"):
            self.model.freeze_quantum_layer()

        # Fit classical layer untuk task baru
        self.fit(new_train_loader, new_val_loader)

        # Simpan checkpoint tiap task
        checkpoint_dir = Path(getattr(self.logging_cfg, "checkpoint_dir", "artifacts/checkpoints"))
        ensure_dir(checkpoint_dir)
        save_checkpoint(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "task": self.continual_task,
            },
            checkpoint_dir / f"task_{self.continual_task}.pt"
        )
        console.log(f"[green]Task {self.continual_task} finished and checkpoint saved[/green]")

    # ---------------- Checkpointing ----------------
    def _maybe_checkpoint(self, val_loss: float, val_metrics: Dict[str, float]) -> None:
        metric_name = getattr(getattr(self.training_cfg, "early_stopping", {}), "metric", "val_loss")
        mode = getattr(getattr(self.training_cfg, "early_stopping", {}), "mode", "min").lower()

        if metric_name == "val_loss":
            current = val_loss
        else:
            current = val_metrics.get(metric_name, None)
            if current is None:
                console.log(f"[yellow]Metric {metric_name} not found in validation metrics.[/yellow]")
                return

        improved = current < self.best_metric_value if mode == "min" else current > self.best_metric_value
        if improved:
            self.best_metric_value = current
            self.early_stop_counter = 0
            checkpoint_state = {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "metric": current,
            }
            save_checkpoint(checkpoint_state, getattr(self.logging_cfg, "best_checkpoint", "artifacts/checkpoints/best.pt"))
            console.log("[green]Saved new best checkpoint.[/green]")
        else:
            self.early_stop_counter += 1

    def _should_stop(self) -> bool:
        if not self._validation_available:
            return False
        patience = getattr(getattr(self.training_cfg, "early_stopping", {}), "patience", 0)
        if patience <= 0:
            return False
        return self.early_stop_counter >= patience

    # ---------------- Load Checkpoints ----------------
    def load_best_checkpoint(self) -> None:
        best_path = Path(getattr(self.logging_cfg, "best_checkpoint", "artifacts/checkpoints/best.pt"))
        if not best_path.exists():
            console.log(f"[yellow]Best checkpoint not found at {best_path}.")
            return
        checkpoint = torch.load(best_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        console.log(f"[green]Loaded best checkpoint from {best_path}[/green]")

    def load_checkpoint(self, path: Path | str) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        console.log(f"[green]Resumed from checkpoint: {path}[/green]")

    # ---------------- Quantum Natural Gradient ----------------
    def optimize_quantum_layer_qng(self, inputs: torch.Tensor, targets: torch.Tensor, steps: int = 1) -> None:
        if not hasattr(self.model, "quantum_layer"):
            console.log("[yellow]Model tidak memiliki quantum_layer.[/yellow]")
            return

        qlayer = self.model.quantum_layer
        if not hasattr(qlayer, "weight"):
            console.log("[yellow]Quantum layer tidak memiliki atribut weight.[/yellow]")
            return

        opt_qng = qml.QNGOptimizer(stepsize=0.1)
        weights = qlayer.weight.detach().numpy()

        for _ in range(steps):
            def loss_fn(w):
                qlayer.weight.data = torch.tensor(w, dtype=torch.float32)
                log_probs = self.model(inputs)
                return torch.nn.functional.nll_loss(log_probs, targets).item()
            weights = opt_qng.step(loss_fn, weights)

        qlayer.weight.data = torch.tensor(weights, dtype=torch.float32)
        console.log("[green]Quantum layer updated with QNG.[/green]")
