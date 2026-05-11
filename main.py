"""
Train BrainConnectivityMLP on CIFAR-10.

Full pipeline:
  Flatten image (32x32x3 = 3072)
  -> Linear input projection  (3072 -> N)
  -> BrainConnectivityMLP     (N -> N, weights optionally from FC matrix)
  -> Linear classifier        (N -> 10 classes)

Set FC_MATRIX_PATH to your HCP pickle file to use a real brain FC matrix,
or leave it as None to use a random NxN matrix.
"""

import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from src.brain_to_dnn import BrainConnectivityMLP


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Path to the HCP pickle file — set to None to use a random FC matrix.
FC_MATRIX_PATH = '/content/drive/MyDrive/IMPERIAL/data/subject_data_1_cleaned_precise_age.pkl'
SAVED_DATA_PATH = 'content/drive/MyDrive/IMPERIAL/data/image_dataset'
# FC_MATRIX_PATH = "/Users/stefanovannoni/Desktop/IMPERIAL COLLEGE/Data/data/hcp_ya_dataset/subject_data_1_cleaned_precise_age.pkl"

N_NODES      = 379   # neurons per layer; ignored when FC_MATRIX_PATH is set (N comes from the file)
N_HIDDEN     = 2     # number of hidden layers inside BrainConnectivityMLP
USE_FC_INIT  = True  # True = init weights from FC matrix | False = Kaiming random init

BATCH_SIZE   = 256
EPOCHS       = 100
LR           = 1e-3

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


# ---------------------------------------------------------------------------
# FC matrix loading
# ---------------------------------------------------------------------------

def load_fc_matrix(path: str | None, n: int) -> np.ndarray:
    """Return an NxN FC matrix: loaded from an HCP pickle or randomly generated."""
    if path is None:
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((n, n))
        fc  = (raw + raw.T) / 2          # make symmetric like a real FC matrix
        np.fill_diagonal(fc, 1.0)
        print(f"Using random {n}x{n} FC matrix (no path provided).")
        return fc

    with open(path, "rb") as f:
        data = pickle.load(f)

    matrices = [v["FC"] for v in data.values() if "FC" in v]

    random_idx = np.random.randint(len(matrices))
    fc = matrices[random_idx]

    print(f"Loaded mean FC from {len(matrices)} subjects — shape {fc.shape}.")
    print(f"Sex: {data[list(data.keys())[random_idx]]['gender']}, Age: {data[list(data.keys())[random_idx]]['age']}")
    return fc


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CIFARClassifier(nn.Module):
    """
    Adapts BrainConnectivityMLP to CIFAR-10:
      - input_proj:  maps flattened image (3072) to brain-network dimension (N)
      - brain_mlp:   the brain-connectivity MLP (N -> N)
      - classifier:  linear head (N -> 10 classes)
    """

    IMAGE_DIM = 32 * 32 * 3   # 3072
    N_CLASSES  = 10

    def __init__(
        self,
        fc_matrix: np.ndarray,
        n_hidden_layers: int,
        use_fc_init: bool,
    ) -> None:
        super().__init__()
        n = fc_matrix.shape[0]

        self.input_proj = nn.Linear(self.IMAGE_DIM, n)
        self.brain_mlp  = BrainConnectivityMLP(
            fc_matrix,
            n_hidden_layers=n_hidden_layers,
            use_fc_init=use_fc_init,
        )
        self.classifier = nn.Linear(n, self.N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)           # (B, 3072)
        x = torch.relu(self.input_proj(x))   # (B, N)
        x = self.brain_mlp(x)                # (B, N)
        return self.classifier(x)            # (B, 10)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    # CIFAR-10 channel mean / std (pre-computed on the training set)
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=SAVED_DATA_PATH, train=True,  download=True, transform=transform)
    val_set   = torchvision.datasets.CIFAR10(
        root=SAVED_DATA_PATH, train=False, download=True, transform=transform)

    # num_workers > 0 and pin_memory require CUDA; MPS/CPU work fine with 0 / False
    pin  = torch.cuda.is_available()
    nw   = 2 if torch.cuda.is_available() else 0

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=pin)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Device : {DEVICE}")
    print(f"FC init: {USE_FC_INIT}  |  hidden layers: {N_HIDDEN}  |  epochs: {EPOCHS}\n")

    fc    = load_fc_matrix(FC_MATRIX_PATH, N_NODES)
    model = CIFARClassifier(fc, N_HIDDEN, USE_FC_INIT).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model  : {model.brain_mlp}")
    print(f"Params : {n_params:,}\n")

    train_loader, val_loader = get_dataloaders(BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    header = f"{'Epoch':>5}  {'Train loss':>10}  {'Train acc':>9}  {'Val loss':>9}  {'Val acc':>8}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, DEVICE)

        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>9.3%}  {val_loss:>9.4f}  {val_acc:>8.3%}")

    torch.save(model.state_dict(), "checkpoint.pt")
    print("\nCheckpoint saved to checkpoint.pt")


if __name__ == "__main__":
    main()
