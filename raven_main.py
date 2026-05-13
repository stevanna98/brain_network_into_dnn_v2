"""
Train BrainConnectivityMLP on RAVEN to predict the missing shape.

Full pipeline:
  8 context panels (resize_dim × resize_dim, grayscale)
  -> flatten (8 * resize_dim^2)
  -> Linear input projection  (8 * resize_dim^2 -> N)
  -> BrainConnectivityMLP     (N -> N, weights optionally from FC matrix)
  -> Linear classifier        (N -> 8 answer choices)

The correct answer index (0-7) is predicted as a classification task.
Chance level is 12.5% (1 in 8).

Dataset is loaded from HuggingFace: HuggingFaceM4/RAVEN
Available configs: center, distribute_four, distribute_nine, in_center_single_out,
                   in_distribute_four_out, left_right, up_down

Usage examples:
  python raven_main.py
  python raven_main.py --hf_config center
  python raven_main.py --fc_path /path/to/data.pkl --use_fc_init
  python raven_main.py --n_nodes 379 --n_hidden 3 --epochs 50 --resize_dim 40
"""

import argparse
import os
import pickle
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from src.brain_to_dnn import BrainConnectivityMLP

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BrainConnectivityMLP on RAVEN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--fc_path", type=str,
        default="/content/drive/MyDrive/IMPERIAL/data/subject_data_1_cleaned_precise_age.pkl",
        help="Path to HCP pickle file. Set to 'none' to use a random FC matrix.",
    )
    parser.add_argument(
        "--hf_dataset", type=str, default="HuggingFaceM4/RAVEN",
        help="HuggingFace dataset repository.",
    )
    parser.add_argument(
        "--hf_config", type=str, default="distribute_nine",
        help="RAVEN configuration name (e.g. center, distribute_nine, left_right).",
    )
    parser.add_argument(
        "--resize_dim", type=int, default=40,
        help="Resize each panel to resize_dim x resize_dim before flattening.",
    )

    # Model
    parser.add_argument("--n_nodes",  type=int,   default=379,
                        help="Neurons per BrainMLP layer. Ignored when --fc_path is set.")
    parser.add_argument("--n_hidden", type=int,   default=2,
                        help="Number of hidden layers inside BrainConnectivityMLP.")
    parser.add_argument("--use_fc_init", action="store_true", default=False,
                        help="Initialise MLP weights from FC matrix (default: Kaiming random).")
    parser.add_argument("--keep_ratio", type=float, default=None,
                        help="If set, only keep the top X% strongest connections from the FC matrix; "
                             "the rest are zeroed out. Only applies if --use_fc_init is set.")
    parser.add_argument("--sample", type=str, default="single",
                        help="If --fc_path is set, whether to sample a single subject's FC ('single') or use the average FC across all subjects ('average').")  

    # Training
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-3)

    # Misc
    parser.add_argument("--device", type=str, default=None,
                        help="Device override ('cpu', 'cuda', 'mps'). Auto-detected if unset.")
    parser.add_argument(
        "--checkpoint", type=str,
        default="/content/drive/MyDrive/IMPERIAL/checkpoint/raven_checkpoint.pt",
    )
    parser.add_argument(
        "--plots_dir", type=str,
        default="/content/drive/MyDrive/IMPERIAL/plots/raven",
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

def load_fc_matrix(path: str | None, n: int, sample: str) -> np.ndarray:
    """Return an NxN FC matrix: loaded from an HCP pickle or randomly generated."""
    if path is None or path.lower() == "none":
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((n, n))
        fc  = (raw + raw.T) / 2
        np.fill_diagonal(fc, 1.0)
        print(f"Using random {n}x{n} FC matrix (no path provided).")
        return fc

    with open(path, "rb") as f:
        data = pickle.load(f)
    
    if sample == 'single':
        subject_id = random.choice(list(data.keys()))
        fc  = data[subject_id]["FC"]
        sex = data[subject_id]["gender"]
        age = data[subject_id]["age"]
        print(f"Loaded FC from {subject_id} — shape {fc.shape}, sex={sex}, age={age}")
    elif sample == 'average':
        fc_matrices = [data[subject_id]['FC'] for subject_id in data]
        fc = np.mean(fc_matrices, axis=0)
        print(f"Loaded average FC across {len(fc_matrices)} subjects — shape {fc.shape}")
    return fc


# ---------------------------------------------------------------------------
# RAVEN Dataset (HuggingFace)
# ---------------------------------------------------------------------------

class RAVENHFDataset(Dataset):
    """
    Wraps a HuggingFace RAVEN split.

    Expected HF schema per example:
      images:  list of 16 PIL images — first 8 are context panels, last 8 are answer candidates
      answer:  int in [0, 7] — index of the correct answer

    Returns:
      context: (8, resize_dim, resize_dim) float32 in [0, 1]
      target:  long scalar
    """

    N_CONTEXT = 8
    _IMAGE_FIELDS  = ["images", "image", "context", "panels"]
    _ANSWER_FIELDS = ["answer", "target", "label"]

    def __init__(self, hf_split, resize_dim: int = 40) -> None:
        self.data      = hf_split
        self.transform = T.Compose([
            T.Grayscale(),
            T.Resize((resize_dim, resize_dim), antialias=True),
            T.ToTensor(),   # -> (1, H, W) in [0, 1]
        ])
        features = set(hf_split.features.keys())
        self.image_field  = self._detect(features, self._IMAGE_FIELDS,  "image")
        self.answer_field = self._detect(features, self._ANSWER_FIELDS, "answer")
        print(f"  image field : '{self.image_field}'")
        print(f"  answer field: '{self.answer_field}'")

    @staticmethod
    def _detect(features: set, candidates: list[str], role: str) -> str:
        for name in candidates:
            if name in features:
                return name
        raise KeyError(
            f"Cannot find {role} field in dataset. "
            f"Available fields: {sorted(features)}. "
            f"Tried: {candidates}"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        example = self.data[idx]
        panels  = example[self.image_field][:self.N_CONTEXT]
        target  = int(example[self.answer_field])

        context = torch.stack([self.transform(p) for p in panels])  # (8, 1, H, W)
        context = context.squeeze(1)                                  # (8, H, W)
        return context, torch.tensor(target, dtype=torch.long)


def get_dataloaders(
    hf_dataset: str,
    hf_config: str,
    resize_dim: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader | None]:
    print(f"Loading '{hf_dataset}' config='{hf_config}' from HuggingFace...")
    ds = load_dataset(hf_dataset, hf_config)

    # Print schema once so the user can verify field names
    print(f"Dataset features: {ds['train'].features}\n")

    train_split = ds["train"]
    val_split   = ds.get("validation") or ds.get("val")

    print(f"Train problems : {len(train_split)}")
    if val_split is not None:
        print(f"Val   problems : {len(val_split)}")
    else:
        print("Val   problems : none found (skipping val metrics)")

    pin = torch.cuda.is_available()
    nw  = 2 if torch.cuda.is_available() else 0

    train_loader = DataLoader(
        RAVENHFDataset(train_split, resize_dim),
        batch_size=batch_size, shuffle=True,  num_workers=nw, pin_memory=pin,
    )
    val_loader = DataLoader(
        RAVENHFDataset(val_split, resize_dim),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin,
    ) if val_split is not None else None

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RAVENClassifier(nn.Module):
    """
    BrainConnectivityMLP adapted for RAVEN 8-way answer selection.

      input_proj:  flattened 8 context panels (8 * resize_dim^2) -> N
      brain_mlp:   N -> N
      classifier:  N -> 8
    """

    N_ANSWERS = 8

    def __init__(
        self,
        fc_matrix: np.ndarray,
        n_hidden_layers: int,
        use_fc_init: bool,
        resize_dim: int,
        keep_ratio: float
    ) -> None:
        super().__init__()
        n         = fc_matrix.shape[0]
        input_dim = 8 * resize_dim * resize_dim

        self.input_proj = nn.Linear(input_dim, n)
        self.brain_mlp  = BrainConnectivityMLP(
            fc_matrix,
            n_hidden_layers=n_hidden_layers,
            use_fc_init=use_fc_init,
            keep_ratio=keep_ratio
        )
        self.classifier = nn.Linear(n, self.N_ANSWERS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 8, resize_dim, resize_dim)
        x = x.flatten(start_dim=1)          # (B, 8 * resize_dim^2)
        x = torch.relu(self.input_proj(x))  # (B, N)
        x = self.brain_mlp(x)              # (B, N)
        return self.classifier(x)           # (B, 8)


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

    for context, labels in loader:
        context, labels = context.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(context)
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

    for context, labels in loader:
        context, labels = context.to(device), labels.to(device)
        logits = model(context)
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
    os.makedirs(plots_dir, exist_ok=True)
    epochs   = list(range(1, epoch + 1))
    init_tag = "fc_init" if use_fc_init else "random_init"

    # 1. Weight distribution
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

    # 2. Training loss curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_losses, color="tomato", marker="o", markersize=3)
    ax.set_title(f"Training loss ({init_tag})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_xlim(1, max(epochs) if len(epochs) > 1 else 2)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"loss_{init_tag}.png"), dpi=100)
    plt.close(fig)

    # 3. Accuracy curve (train + val), with chance baseline
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [a * 100 for a in train_accs], color="steelblue",
            marker="o", markersize=3, label="Train")
    if val_accs:
        ax.plot(epochs, [a * 100 for a in val_accs], color="darkorange",
                marker="o", markersize=3, label="Val")
    ax.axhline(12.5, color="gray", linestyle="--", linewidth=1, label="Chance (12.5%)")
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

    input_dim = 8 * args.resize_dim ** 2
    print(f"Device     : {device}")
    print(f"FC init    : {args.use_fc_init}  |  hidden layers: {args.n_hidden}  |  epochs: {args.epochs}")
    print(f"Resize dim : {args.resize_dim}  |  MLP input dim: {input_dim}\n")

    fc    = load_fc_matrix(args.fc_path, args.n_nodes, args.sample)
    model = RAVENClassifier(fc, args.n_hidden, args.use_fc_init, args.resize_dim, args.keep_ratio).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model  : {model.brain_mlp}")
    print(f"Params : {n_params:,}\n")

    train_loader, val_loader = get_dataloaders(
        args.hf_dataset, args.hf_config, args.resize_dim, args.batch_size,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.plots_dir, exist_ok=True)
    print(f"Plots  : {args.plots_dir}/\n")

    header = f"{'Epoch':>5}  {'Train loss':>10}  {'Train acc':>9}  {'Val loss':>9}  {'Val acc':>8}"
    print(header)
    print("-" * len(header))

    train_losses, train_accs, val_accs = [], [], []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss_str = val_acc_str = "       N/A"
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            val_accs.append(val_acc)
            val_loss_str = f"{val_loss:>9.4f}"
            val_acc_str  = f"{val_acc:>8.3%}"

        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>9.3%}  {val_loss_str}  {val_acc_str}")

        save_plots(model, epoch, train_losses, train_accs, val_accs,
                   args.plots_dir, args.use_fc_init)

    torch.save(model.state_dict(), args.checkpoint)
    print(f"\nCheckpoint saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
