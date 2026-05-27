"""
Train BrainConnectivityMLP on CIFAR-10 or MNIST.

Full pipeline:
  Flatten image (D pixels)
  -> Linear input projection  (D -> N)
  -> BrainConnectivityMLP     (N -> N, weights optionally from FC matrix)
  -> Linear classifier        (N -> 10 classes)

Usage examples:
  python main.py
  python main.py --dataset mnist
  python main.py --fc_path /path/to/data.pkl --use_fc_init
  python main.py --n_nodes 128 --n_hidden 3 --epochs 20 --lr 5e-4
"""

import argparse
import os
import pickle
import sys
import random

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts and Colab
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from src.brain_to_dnn import BrainConnectivityMLP

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

# python main.py --dataset mnist --fc_path '/Users/stefanovannoni/Desktop/IMPERIAL COLLEGE/Data/data/hcp_ya_dataset/subject_data_1_cleaned_precise_age.pkl' --batch_size 64 --epochs 30 --n_hidden 2 --layer_config brain_random --keep_ratio 0.2

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BrainConnectivityMLP on CIFAR-10 or MNIST.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"],
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--fc_path", type=str,
        default="/content/drive/MyDrive/IMPERIAL/data/subject_data_1_cleaned_precise_age.pkl",
        help="Path to HCP pickle file. Omit or set to 'none' to use a random FC matrix.",
    )
    parser.add_argument(
        "--data_path", type=str,
        default="content/drive/MyDrive/IMPERIAL/data/image_dataset",
        help="Directory where dataset is stored (or will be downloaded).",
    )

    # Model
    parser.add_argument(
        "--n_nodes", type=int, default=379,
        help="Neurons per layer. Ignored when --fc_path is set (N comes from the file).",
    )
    parser.add_argument(
        "--n_hidden", type=int, default=2,
        help="Number of hidden layers inside BrainConnectivityMLP.",
    )
    parser.add_argument(
        "--use_fc_init", action='store_true', default=False,
        help="Initialise MLP weights from the FC matrix (default: Kaiming random).",
    )
    parser.add_argument(
        "--keep_ratio", type=float, default=None,
        help="If set, only keep the top X% strongest connections from the FC matrix; "
             "the rest are zeroed out. Only applies if --use_fc_init is set.",
    )
    parser.add_argument(
        "--sample", type=str, default="single",
        help="If --fc_path is set, whether to sample a single subject's FC ('single') or use the average FC across all subjects ('average')."
    )
    parser.add_argument(
        "--n_frozen_layers", type=int, default=0,
        help="Number of the first N brain-MLP layers to freeze (no gradient updates). "
             "Must be <= n_hidden + 1.",
    )
    parser.add_argument(
        "--frozen_fc_init", action=argparse.BooleanOptionalAction, default=True,
        help="If set (default), frozen layers are initialised from the FC matrix. "
             "Use --no_frozen_fc_init for Kaiming random initialisation instead. "
             "Has no effect when --n_frozen_layers=0.",
    )
    parser.add_argument(
        "--layer_config", type=str, default=None,
        choices=["brain_brain", "brain_random", "random_brain"],
        help="Per-layer init/freeze config (only valid with --n_hidden 2). "
             "brain_brain: both layers FC-init & frozen. "
             "brain_random: layer 0 FC-init & frozen, layer 1 random & trainable. "
             "random_brain: layer 0 random & trainable, layer 1 FC-init & frozen. "
             "Overrides --use_fc_init, --n_frozen_layers, --frozen_fc_init.",
    )

    # Training
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--subject_id", type=str, default=None,
        help="If --sample=single, specify a subject ID from the FC pickle file to use. Omit to select a random subject."
    )

    # Misc
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (e.g. 'cpu', 'cuda', 'mps'). Auto-detected if not set.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="/content/drive/MyDrive/IMPERIAL/checkpoint/checkpoint.pt",
        help="Path to save the trained model weights.",
    )
    parser.add_argument(
        "--plots_dir", type=str, default="/content/drive/MyDrive/IMPERIAL/plots",
        help="Directory where per-epoch plots are saved.",
    )
    parser.add_argument(
        "--metrics_dir", type=str, default="/content/drive/MyDrive/IMPERIAL/metrics",
        help="Directory where per-subject .npy metric files are saved.",
    )

    return parser.parse_args()


def resolve_device(override: str | None) -> str:
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# FC matrix loading
# ---------------------------------------------------------------------------

def load_fc_matrix(path: str | None, n: int, sample: str, subject_id: str | None = None) -> tuple[np.ndarray, str]:
    """Return (fc_matrix, subject_tag) where subject_tag identifies the run."""
    if path is None or path.lower() == "none":
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((n, n))
        fc  = (raw + raw.T) / 2          # make symmetric like a real FC matrix
        np.fill_diagonal(fc, 1.0)
        print(f"Using random {n}x{n} FC matrix (no path provided).")
        return fc, "random"

    with open(path, "rb") as f:
        data = pickle.load(f)

    if sample == 'single':
        if subject_id is None:
            subject_id = random.choice(list(data.keys()))
        fc = data[subject_id]['FC']
        sex = data[subject_id]['gender']
        age = data[subject_id]['age']
        print(f"Loaded FC from {subject_id} — shape {fc.shape}.")
        print(f"Selected subject: sex={sex}, age={age}")
        return fc, str(subject_id)
    elif sample == 'average':
        fc_matrices = [data[subject_id]['FC'] for subject_id in data]
        fc = np.mean(fc_matrices, axis=0)
        print(f"Loaded average FC across {len(fc_matrices)} subjects — shape {fc.shape}")
        return fc, "average"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ImageClassifier(nn.Module):
    """
    Adapts BrainConnectivityMLP to an image classification task:
      - input_proj:  maps flattened image (image_dim) to brain-network dimension (N)
      - brain_mlp:   the brain-connectivity MLP (N -> N)
      - classifier:  linear head (N -> n_classes)
    """

    N_CLASSES = 10

    def __init__(
        self,
        fc_matrix: np.ndarray,
        image_dim: int,
        n_hidden_layers: int,
        use_fc_init: bool,
        keep_ratio: float,
        n_frozen_layers: int = 0,
        frozen_fc_init: bool = True,
        layer_config: str | None = None,
    ) -> None:
        super().__init__()
        n = fc_matrix.shape[0]

        self.input_proj = nn.Linear(image_dim, n)
        self.brain_mlp  = BrainConnectivityMLP(
            fc_matrix,
            n_hidden_layers=n_hidden_layers,
            use_fc_init=use_fc_init,
            keep_ratio=keep_ratio,
            n_frozen_layers=n_frozen_layers,
            frozen_fc_init=frozen_fc_init,
            layer_config=layer_config,
        )
        print(self.brain_mlp)
        self.classifier = nn.Linear(n, self.N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        x = torch.relu(self.input_proj(x))
        x = self.brain_mlp(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_DATASET_CFG = {
    "cifar10": {
        "cls":       torchvision.datasets.CIFAR10,
        "image_dim": 32 * 32 * 3,
        "mean":      (0.4914, 0.4822, 0.4465),
        "std":       (0.2470, 0.2435, 0.2616),
    },
    "mnist": {
        "cls":       torchvision.datasets.MNIST,
        "image_dim": 28 * 28,
        "mean":      (0.1307,),
        "std":       (0.3081,),
    },
}


def get_dataloaders(
    batch_size: int, data_path: str, dataset: str
) -> tuple[DataLoader, DataLoader, int]:
    cfg = _DATASET_CFG[dataset]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])

    train_set = cfg["cls"](root=data_path, train=True,  download=True, transform=transform)
    val_set   = cfg["cls"](root=data_path, train=False, download=True, transform=transform)

    # num_workers > 0 and pin_memory require CUDA; MPS/CPU work fine with 0 / False
    pin = torch.cuda.is_available()
    nw  = 2 if torch.cuda.is_available() else 0

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=pin)
    return train_loader, val_loader, cfg["image_dim"]


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
# Plotting
# ---------------------------------------------------------------------------

def save_plots(
    model: nn.Module,
    epoch: int,
    train_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    plots_dir: str,
    use_fc_init: bool,
) -> None:
    """Save three plots for the current epoch to plots_dir."""
    os.makedirs(plots_dir, exist_ok=True)
    epochs = list(range(1, epoch + 1))
    init_tag = "fc_init" if use_fc_init else "random_init"

    # 1. Weight distribution — snapshot of brain_mlp weights at this epoch
    all_weights = torch.cat([
        layer.weight.data.cpu().flatten()
        for layer in model.brain_mlp.linear_layers()
    ]).numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(all_weights, bins=100, color="steelblue", edgecolor="none")
    ax.set_title(f"Weight distribution ({init_tag}) — epoch {epoch}")
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"weights_{init_tag}_epoch_{epoch:03d}.png"), dpi=100)
    plt.close(fig)

    # 2. Training loss curve (full history up to this epoch)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_losses, color="tomato", marker="o", markersize=3)
    ax.set_title(f"Training loss ({init_tag})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_xlim(1, max(epochs) if len(epochs) > 1 else 2)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"loss_{init_tag}.png"), dpi=100)
    plt.close(fig)

    # 3. Accuracy curve — train and val (full history up to this epoch)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [a * 100 for a in train_accs], color="steelblue",
            marker="o", markersize=3, label="Train")
    ax.plot(epochs, [a * 100 for a in val_accs],   color="darkorange",
            marker="o", markersize=3, label="Val")
    ax.set_title(f"Accuracy ({init_tag})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(1, max(epochs) if len(epochs) > 1 else 2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"accuracy_{init_tag}.png"), dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)

    print(f"Device  : {device}")
    print(f"Dataset : {args.dataset}")
    if args.layer_config:
        print(f"FC init : layer_config={args.layer_config}  |  hidden layers: {args.n_hidden}  |  epochs: {args.epochs}\n")
    else:
        print(f"FC init : {args.use_fc_init}  |  hidden layers: {args.n_hidden}  |  frozen layers: {args.n_frozen_layers} (fc_init={args.frozen_fc_init})  |  epochs: {args.epochs}\n")

    fc, subject_tag = load_fc_matrix(args.fc_path, args.n_nodes, args.sample, args.subject_id)
    train_loader, val_loader, image_dim = get_dataloaders(
        args.batch_size, args.data_path, args.dataset
    )
    model = ImageClassifier(
        fc, image_dim, args.n_hidden, args.use_fc_init, args.keep_ratio,
        args.n_frozen_layers, args.frozen_fc_init, args.layer_config,
    ).to(device)
    print(f"Model architecture:\n{model}\n")

    n_params    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model  : {model.brain_mlp}")
    print(f"Params : {n_params:,} total  |  {n_trainable:,} trainable\n")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.plots_dir, exist_ok=True)
    print(f"Plots  : {args.plots_dir}/\n")

    header = f"{'Epoch':>5}  {'Train loss':>10}  {'Train acc':>9}  {'Val loss':>9}  {'Val acc':>8}"
    print(header)
    print("-" * len(header))

    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    import time
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>9.3%}  {val_loss:>9.4f}  {val_acc:>8.3%}")

        save_plots(model, epoch, train_losses, train_accs, val_accs, args.plots_dir, args.use_fc_init)

    torch.save(model.state_dict(), args.checkpoint)
    print(f"\nCheckpoint saved to {args.checkpoint}")

    os.makedirs(args.metrics_dir, exist_ok=True)
    np.save(os.path.join(args.metrics_dir, f"val_loss_{subject_tag}.npy"), np.array(val_losses))
    np.save(os.path.join(args.metrics_dir, f"val_acc_{subject_tag}.npy"),  np.array(val_accs))
    print(f"Metrics saved to {args.metrics_dir}/ (subject: {subject_tag})")

    elapsed = time.time() - t_start
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)
    print(f"\nTraining time : {h:02d}:{m:02d}:{s:02d}  ({elapsed:.1f}s total)")


if __name__ == "__main__":
    main()


# python main.py --dataset mnist --batch_size 64 --epochs 30 --n_hidden 1 --n_frozen_layer 1 --subject_id 185139
