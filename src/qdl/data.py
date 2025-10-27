from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np # type: ignore
import torch # type: ignore
from torch.utils.data import DataLoader, Dataset # type: ignore

from .config import ConfigBundle, DataConfig
from .utils import ensure_dir


class QuantumSyntheticDataset(Dataset):
    """Dataset sintetis dengan struktur inspirasi interferensi kuantum."""

    def __init__(
        self,
        n_qubits: int,
        n_samples: int,
        split: str,
        seed: int,
        difficulty: float = 1.0,
    ) -> None:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        self.n_qubits = n_qubits
        self.split = split

        rng = np.random.default_rng(seed)
        base = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_qubits))
        mixing = rng.normal(loc=0.0, scale=difficulty, size=(n_qubits, n_qubits))
        transformed = np.tanh(base @ mixing)

        phase_operator = rng.normal(loc=0.0, scale=1.0, size=(n_qubits, n_qubits))
        projected = transformed @ phase_operator
        interference = np.sin(projected) + 0.5 * np.cos(2.0 * projected)
        global_phase = interference.sum(axis=1) + 0.2 * np.sin(projected).sum(axis=1)

        labels = (global_phase > 0).astype(np.int64)
        features = transformed + rng.normal(scale=0.03, size=transformed.shape)
        features = np.clip(features, -1.0, 1.0)

        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.metadata = {"split": split, "seed": seed, "difficulty": difficulty}

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def _build_synthetic_splits(cfg: ConfigBundle) -> Tuple[Dataset, Dataset, Dataset]:
    data_cfg = cfg.data
    seed = cfg.seed
    train = QuantumSyntheticDataset(
        n_qubits=data_cfg.n_qubits,
        n_samples=data_cfg.n_train,
        split="train",
        seed=seed,
    )
    val = QuantumSyntheticDataset(
        n_qubits=data_cfg.n_qubits,
        n_samples=data_cfg.n_val,
        split="val",
        seed=seed + 1,
    )
    test = QuantumSyntheticDataset(
        n_qubits=data_cfg.n_qubits,
        n_samples=data_cfg.n_test,
        split="test",
        seed=seed + 2,
    )
    return train, val, test


def create_dataloaders(
    cfg: ConfigBundle,
    dataset_override: Optional[Tuple[Dataset, Dataset, Dataset]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg: DataConfig = cfg.data
    ensure_dir(data_cfg.cache_dir)

    if dataset_override is not None:
        train_set, val_set, test_set = dataset_override
    elif data_cfg.synthetic or data_cfg.dataset_path is None:
        train_set, val_set, test_set = _build_synthetic_splits(cfg)
    else:
        raise NotImplementedError(
            "Custom dataset loading belum diimplementasikan. Silakan set synthetic=true atau "
            "tambahkan loader Anda sendiri."
        )

    def _loader(dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=data_cfg.batch_size,
            shuffle=shuffle,
            num_workers=data_cfg.num_workers,
            pin_memory=False,
        )

    train_loader = _loader(train_set, shuffle=data_cfg.shuffle)
    val_loader = _loader(val_set, shuffle=False)
    test_loader = _loader(test_set, shuffle=False)
    return train_loader, val_loader, test_loader