import torch # type: ignore
from torch.utils.data import DataLoader, TensorDataset, random_split # type: ignore
from torch_geometric.datasets import QM9 # type: ignore
from torch_geometric.data import Data # type: ignore

def load_qm9(batch_size=32, val_ratio=0.1, seed=42):
    """
    Load QM9 dataset, preprocess, dan buat DataLoader.
    """
    # Unduh dataset
    dataset = QM9(root="data/QM9")
    
    # Contoh: ambil 10 fitur pertama sebagai input, target energi (U0)
    features = []
    targets = []
    for data in dataset:
        # data.x biasanya node features, data.y target sifat
        # ambil 10 fitur pertama untuk input
        x = data.x.view(-1)[:10]  # flatten node features
        y = data.y[0]              # contoh prediksi energi U0
        features.append(x)
        targets.append(y)
    
    x_tensor = torch.stack(features).float()
    y_tensor = torch.stack(targets).float().unsqueeze(1)


    val_size = int(len(x_tensor) * val_ratio)
    train_size = len(x_tensor) - val_size
    train_dataset, val_dataset = random_split(TensorDataset(x_tensor, y_tensor),
                                              [train_size, val_size],
                                              generator=torch.Generator().manual_seed(seed))
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = load_qm9()
    print("Train batches:", len(train_loader), "Validation batches:", len(val_loader))
