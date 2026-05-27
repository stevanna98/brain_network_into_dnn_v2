"""
Train BrainConnectivityMLP on CIFAR-10 or MNIST for multiple subjects sequentially.

For each subject:
  - loads their FC matrix from the pickle file
  - trains ImageClassifier with early stopping
  - evaluates accuracy, macro F1, precision, and recall on the validation set
  - saves per-epoch val_acc / val_loss curves as .npy files
  - appends a row to a summary CSV

Usage:
  # train all subjects in the pickle file
  python main_v2.py --fc_path /path/to/data.pkl

  # train a specific subset
  python main_v2.py --fc_path /path/to/data.pkl --subject_ids 194443 185139 177645

  # with layer_config
  python main_v2.py --fc_path /path/to/data.pkl --n_hidden 2 --layer_config brain_random
"""

import argparse
import csv
import os
import pickle
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BrainConnectivityMLP for multiple subjects sequentially.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"])
    parser.add_argument(
        "--fc_path", type=str,
        default="/content/drive/MyDrive/IMPERIAL/data/subject_data_1_cleaned_precise_age.pkl",
        help="Path to HCP pickle file.",
    )
    parser.add_argument(
        "--data_path", type=str,
        default="content/drive/MyDrive/IMPERIAL/data/image_dataset",
        help="Directory where the image dataset is stored (or will be downloaded).",
    )
    parser.add_argument(
        "--subject_ids", type=str, nargs="+", default=None,
        help="Subject IDs to train. Defaults to all subjects in the pickle file.",
    )

    # Model
    parser.add_argument("--n_hidden", type=int, default=2)
    parser.add_argument("--use_fc_init", action="store_true", default=False)
    parser.add_argument("--keep_ratio", type=float, default=None)
    parser.add_argument("--n_frozen_layers", type=int, default=0)
    parser.add_argument(
        "--frozen_fc_init", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--layer_config", type=str, default=None,
        choices=["brain_brain", "brain_random", "random_brain"],
        help="Per-layer init/freeze config (only valid with --n_hidden 2). "
             "Overrides --use_fc_init, --n_frozen_layers, --frozen_fc_init.",
    )

    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Early stopping
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early stopping: number of epochs with no improvement before stopping.",
    )
    parser.add_argument(
        "--min_delta", type=float, default=1e-4,
        help="Early stopping: minimum improvement in val loss to reset the patience counter.",
    )

    # Output
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--metrics_dir", type=str, default="/content/drive/MyDrive/IMPERIAL/metrics",
        help="Directory for per-epoch .npy files and the summary CSV.",
    )
    parser.add_argument(
        "--plots_dir", type=str, default="/content/drive/MyDrive/IMPERIAL/plots",
        help="Directory for per-subject training curve plots.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# Helpers (device, data, model)
# ---------------------------------------------------------------------------

def resolve_device(override: str | None) -> str:
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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


def get_dataloaders(batch_size: int, data_path: str, dataset: str):
    cfg = _DATASET_CFG[dataset]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    pin = torch.cuda.is_available()
    nw  = 2 if torch.cuda.is_available() else 0
    train_set = cfg["cls"](root=data_path, train=True,  download=True, transform=transform)
    val_set   = cfg["cls"](root=data_path, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=pin)
    return train_loader, val_loader, cfg["image_dim"]


class ImageClassifier(nn.Module):
    N_CLASSES = 10

    def __init__(
        self,
        fc_matrix: np.ndarray,
        image_dim: int,
        n_hidden_layers: int,
        use_fc_init: bool,
        keep_ratio: float | None,
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
        self.classifier = nn.Linear(n, self.N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        x = torch.relu(self.input_proj(x))
        x = self.brain_mlp(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Train / eval / metrics
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
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
def evaluate(model, loader, criterion, device):
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


@torch.no_grad()
def compute_metrics(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        preds  = model(images).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return {
        "accuracy":  accuracy_score(all_labels, all_preds),
        "f1":        f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall":    recall_score(all_labels, all_preds, average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Per-subject plot
# ---------------------------------------------------------------------------

def save_subject_plot(subject_id, train_losses, train_accs, val_accs, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    epochs = list(range(1, len(train_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Subject {subject_id}", fontsize=13)

    ax1.plot(epochs, train_losses, color="tomato", marker="o", markersize=3)
    ax1.set_title("Training loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in train_accs], color="steelblue",
             marker="o", markersize=3, label="Train")
    ax2.plot(epochs, [a * 100 for a in val_accs], color="darkorange",
             marker="o", markersize=3, label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"curves_{subject_id}.png"), dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)

    print(f"Device  : {device}  |  Dataset: {args.dataset}")
    if args.layer_config:
        print(f"Config  : layer_config={args.layer_config}  |  n_hidden={args.n_hidden}")
    else:
        print(f"Config  : use_fc_init={args.use_fc_init}  |  n_hidden={args.n_hidden}  |  frozen={args.n_frozen_layers}")
    print(f"Epochs  : {args.epochs}  |  patience={args.patience}  |  min_delta={args.min_delta}\n")

    # Load FC data
    with open(args.fc_path, "rb") as f:
        fc_data = pickle.load(f)

    subject_ids = args.subject_ids if args.subject_ids else list(fc_data.keys())
    print(f"Subjects to train: {len(subject_ids)}\n")

    # Dataloaders (shared across subjects)
    train_loader, val_loader, image_dim = get_dataloaders(
        args.batch_size, args.data_path, args.dataset
    )

    os.makedirs(args.metrics_dir, exist_ok=True)
    os.makedirs(args.plots_dir,   exist_ok=True)

    summary_path = os.path.join(args.metrics_dir, "summary.csv")
    csv_fields   = ["subject_id", "epochs_trained", "accuracy", "f1",
                    "precision", "recall", "training_time_s"]

    # Write CSV header (overwrite if exists)
    with open(summary_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # -----------------------------------------------------------------------
    # Subject loop
    # -----------------------------------------------------------------------
    for idx, subject_id in enumerate(subject_ids, 1):
        sid = str(subject_id)
        if sid not in fc_data and subject_id not in fc_data:
            print(f"[{idx}/{len(subject_ids)}] Subject {subject_id} — not found in pickle, skipping.")
            continue

        entry = fc_data[sid] if sid in fc_data else fc_data[subject_id]
        fc    = entry["FC"]
        print(f"[{idx}/{len(subject_ids)}] Subject {subject_id}  "
              f"(sex={entry.get('gender','?')}, age={entry.get('age','?')})")

        model = ImageClassifier(
            fc, image_dim, args.n_hidden, args.use_fc_init, args.keep_ratio,
            args.n_frozen_layers, args.frozen_fc_init, args.layer_config,
        ).to(device)

        optimizer    = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )
        criterion    = nn.CrossEntropyLoss()
        early_stop   = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

        train_losses, train_accs, val_losses, val_accs = [], [], [], []
        header = f"  {'Ep':>4}  {'Tr loss':>8}  {'Tr acc':>7}  {'Val loss':>8}  {'Val acc':>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        t_start = time.time()
        stopped_epoch = args.epochs

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            train_losses.append(tr_loss)
            train_accs.append(tr_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(f"  {epoch:>4}  {tr_loss:>8.4f}  {tr_acc:>7.3%}  {val_loss:>8.4f}  {val_acc:>7.3%}")

            if early_stop.step(val_loss, model):
                stopped_epoch = epoch
                print(f"  Early stopping at epoch {epoch} (best val loss: {early_stop.best_loss:.4f})")
                break

        elapsed = time.time() - t_start
        h, rem  = divmod(int(elapsed), 3600)
        m, s    = divmod(rem, 60)
        print(f"  Training time: {h:02d}:{m:02d}:{s:02d}")

        # Restore best weights and compute final metrics
        early_stop.restore_best(model)
        metrics = compute_metrics(model, val_loader, device)
        print(f"  Accuracy={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
              f"Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}\n")

        # Save per-epoch curves
        np.save(os.path.join(args.metrics_dir, f"val_acc_{subject_id}.npy"),  np.array(val_accs))
        np.save(os.path.join(args.metrics_dir, f"val_loss_{subject_id}.npy"), np.array(val_losses))

        # Save plot
        save_subject_plot(subject_id, train_losses, train_accs, val_accs, args.plots_dir)

        # Append to summary CSV
        with open(summary_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow({
                "subject_id":      subject_id,
                "epochs_trained":  stopped_epoch,
                "accuracy":        round(metrics["accuracy"],  4),
                "f1":              round(metrics["f1"],         4),
                "precision":       round(metrics["precision"],  4),
                "recall":          round(metrics["recall"],     4),
                "training_time_s": round(elapsed, 1),
            })

    print(f"Done. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
