"""
Train BrainConnectivityMLP on CIFAR-10 or MNIST for multiple subjects,
either sequentially or in parallel (one process per subject).

GPU note: MPS/CUDA memory is shared across processes, so parallel workers
automatically fall back to CPU. Use --n_workers 1 to keep the GPU.

Usage:
  # sequential, GPU
  python main_v2.py --fc_path /path/to/data.pkl --n_workers 1

  # 4 subjects in parallel (CPU)
  python main_v2.py --fc_path /path/to/data.pkl --n_workers 4

  # specific subjects + layer config
  python main_v2.py --fc_path /path/to/data.pkl --subject_ids 194443 185139 \
      --n_hidden 2 --layer_config brain_random --n_workers 2
"""

import argparse
import csv
import os
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

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
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BrainConnectivityMLP for multiple subjects (sequential or parallel).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"])
    parser.add_argument(
        "--fc_path", type=str,
        default="/content/drive/MyDrive/IMPERIAL/data/subject_data_1_cleaned_precise_age.pkl",
    )
    parser.add_argument(
        "--data_path", type=str,
        default="content/drive/MyDrive/IMPERIAL/data/image_dataset",
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
    parser.add_argument("--frozen_fc_init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--layer_config", type=str, default=None,
        choices=["brain_brain", "brain_random", "random_brain"],
        help="Per-layer init/freeze config (only valid with --n_hidden 2).",
    )

    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--lr",         type=float, default=1e-3)

    # Early stopping
    parser.add_argument("--patience",  type=int,   default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)

    # Parallelism
    parser.add_argument(
        "--n_workers", type=int, default=1,
        help="Number of subjects to train in parallel. >1 forces CPU per worker.",
    )

    # Output
    parser.add_argument("--device", type=str, default=None,
                        help="Device for sequential mode (n_workers=1). Auto-detected if not set.")
    parser.add_argument("--metrics_dir", type=str, default="/content/drive/MyDrive/IMPERIAL/metrics")
    parser.add_argument("--plots_dir",   type=str, default="/content/drive/MyDrive/IMPERIAL/plots")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter   = 0
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
# Model & data helpers
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


def get_dataloaders(batch_size: int, data_path: str, dataset: str, download: bool = False):
    cfg = _DATASET_CFG[dataset]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    pin = torch.cuda.is_available()
    nw  = 2 if torch.cuda.is_available() else 0
    train_set = cfg["cls"](root=data_path, train=True,  download=download, transform=transform)
    val_set   = cfg["cls"](root=data_path, train=False, download=download, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=pin)
    return train_loader, val_loader, cfg["image_dim"]


class ImageClassifier(nn.Module):
    N_CLASSES = 10

    def __init__(self, fc_matrix, image_dim, n_hidden_layers, use_fc_init,
                 keep_ratio, n_frozen_layers=0, frozen_fc_init=True, layer_config=None):
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

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = torch.relu(self.input_proj(x))
        x = self.brain_mlp(x)
        return self.classifier(x)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (model(images).detach().argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def compute_metrics(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        all_preds.extend(model(images.to(device)).argmax(1).cpu().numpy())
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
    ax1.set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in train_accs], color="steelblue",
             marker="o", markersize=3, label="Train")
    ax2.plot(epochs, [a * 100 for a in val_accs], color="darkorange",
             marker="o", markersize=3, label="Val")
    ax2.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"curves_{subject_id}.png"), dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Single-subject training job (top-level so it is picklable for subprocesses)
# ---------------------------------------------------------------------------

def train_subject(job: dict) -> dict:
    """Train one subject. Returns a results dict. Safe to call in a subprocess."""
    subject_id  = job["subject_id"]
    fc          = job["fc"]
    image_dim   = job["image_dim"]
    device      = job["device"]
    metrics_dir = job["metrics_dir"]
    plots_dir   = job["plots_dir"]

    torch.manual_seed(seed_value)

    train_loader, val_loader, _ = get_dataloaders(
        job["batch_size"], job["data_path"], job["dataset"]
    )

    model = ImageClassifier(
        fc, image_dim,
        job["n_hidden"], job["use_fc_init"], job["keep_ratio"],
        job["n_frozen_layers"], job["frozen_fc_init"], job["layer_config"],
    ).to(device)

    optimizer  = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=job["lr"]
    )
    criterion  = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=job["patience"], min_delta=job["min_delta"])

    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    stopped_epoch = job["epochs"]

    t_start = time.time()
    for epoch in range(1, job["epochs"] + 1):
        tr_loss, tr_acc   = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if early_stop.step(val_loss, model):
            stopped_epoch = epoch
            break

    elapsed = time.time() - t_start
    early_stop.restore_best(model)
    metrics = compute_metrics(model, val_loader, device)

    # Save per-epoch curves and plot
    np.save(os.path.join(metrics_dir, f"val_acc_{subject_id}.npy"),  np.array(val_accs))
    np.save(os.path.join(metrics_dir, f"val_loss_{subject_id}.npy"), np.array(val_losses))
    save_subject_plot(subject_id, train_losses, train_accs, val_accs, plots_dir)

    return {
        "subject_id":     subject_id,
        "epochs_trained": stopped_epoch,
        "accuracy":       round(metrics["accuracy"],  4),
        "f1":             round(metrics["f1"],         4),
        "precision":      round(metrics["precision"],  4),
        "recall":         round(metrics["recall"],     4),
        "training_time_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def assign_devices(base_device: str, n_jobs: int) -> list[str]:
    """
    Return a device string for each job.
    - Multiple CUDA GPUs: round-robin across all available GPUs.
    - Single CUDA GPU / MPS / CPU: same device for every job.
    """
    if base_device == "cuda" and torch.cuda.device_count() > 1:
        n_gpus = torch.cuda.device_count()
        print(f"[INFO] {n_gpus} CUDA GPUs found — assigning workers round-robin.")
        return [f"cuda:{i % n_gpus}" for i in range(n_jobs)]
    return [base_device] * n_jobs


def _use_threads(device: str) -> bool:
    """
    True  → ThreadPoolExecutor  (single GPU/MPS: workers share one device context)
    False → ProcessPoolExecutor (multiple CUDA GPUs: each process owns a device)
    """
    if device == "cuda" and torch.cuda.device_count() > 1:
        return False   # multiple GPUs → separate processes, round-robin assignment
    return True        # single GPU, MPS, or CPU → threads share the device safely


def main() -> None:
    import multiprocessing as _mp
    args   = parse_args()
    device = resolve_device(args.device)

    if args.n_workers > 1 and not _use_threads(device):
        # Multi-GPU ProcessPoolExecutor requires spawn to avoid inheriting
        # a forked GPU context that can deadlock or corrupt state.
        _mp.set_start_method("spawn", force=True)

    print(f"Device  : {device}  |  Dataset: {args.dataset}  |  Workers: {args.n_workers}")
    if args.layer_config:
        print(f"Config  : layer_config={args.layer_config}  |  n_hidden={args.n_hidden}")
    else:
        print(f"Config  : use_fc_init={args.use_fc_init}  |  n_hidden={args.n_hidden}  "
              f"|  frozen={args.n_frozen_layers}")
    print(f"Epochs  : {args.epochs}  |  patience={args.patience}  |  min_delta={args.min_delta}\n")

    # Pre-download dataset in the main process to avoid race conditions in workers
    get_dataloaders(args.batch_size, args.data_path, args.dataset, download=True)
    _, _, image_dim = get_dataloaders(args.batch_size, args.data_path, args.dataset)

    # Load FC data
    with open(args.fc_path, "rb") as f:
        fc_data = pickle.load(f)

    subject_ids = args.subject_ids if args.subject_ids else list(fc_data.keys())
    print(f"Subjects to train: {len(subject_ids)}\n")

    os.makedirs(args.metrics_dir, exist_ok=True)
    os.makedirs(args.plots_dir,   exist_ok=True)

    # Build one job dict per subject (plain dicts are picklable)
    valid_subjects = []
    for sid in subject_ids:
        key   = sid if sid in fc_data else str(sid)
        entry = fc_data.get(key)
        if entry is None:
            print(f"Subject {sid} not found in pickle — skipping.")
        else:
            valid_subjects.append((sid, entry))

    devices = assign_devices(device, len(valid_subjects))

    jobs = []
    for (sid, entry), dev in zip(valid_subjects, devices):
        jobs.append({
            "subject_id":     sid,
            "fc":             entry["FC"],
            "image_dim":      image_dim,
            "device":         dev,
            "data_path":      args.data_path,
            "dataset":        args.dataset,
            "batch_size":     args.batch_size,
            "epochs":         args.epochs,
            "lr":             args.lr,
            "patience":       args.patience,
            "min_delta":      args.min_delta,
            "n_hidden":       args.n_hidden,
            "use_fc_init":    args.use_fc_init,
            "keep_ratio":     args.keep_ratio,
            "n_frozen_layers":args.n_frozen_layers,
            "frozen_fc_init": args.frozen_fc_init,
            "layer_config":   args.layer_config,
            "metrics_dir":    args.metrics_dir,
            "plots_dir":      args.plots_dir,
        })

    summary_path = os.path.join(args.metrics_dir, "summary.csv")
    csv_fields   = ["subject_id", "epochs_trained", "accuracy", "f1",
                    "precision", "recall", "training_time_s"]

    with open(summary_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    t_total = time.time()

    # -----------------------------------------------------------------------
    # Sequential (n_workers=1) or parallel (n_workers>1)
    # -----------------------------------------------------------------------
    if args.n_workers == 1:
        for i, job in enumerate(jobs, 1):
            print(f"[{i}/{len(jobs)}] Subject {job['subject_id']}")
            result = train_subject(job)
            h, rem = divmod(int(result["training_time_s"]), 3600)
            m, s   = divmod(rem, 60)
            print(f"  Done — acc={result['accuracy']:.4f}  f1={result['f1']:.4f}  "
                  f"prec={result['precision']:.4f}  rec={result['recall']:.4f}  "
                  f"time={h:02d}:{m:02d}:{s:02d}  "
                  f"epochs={result['epochs_trained']}\n")
            with open(summary_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=csv_fields).writerow(result)

    else:
        use_threads = _use_threads(device)
        Executor    = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        mode        = "threads (shared GPU context)" if use_threads else "processes (multi-GPU)"
        print(f"[INFO] Parallel mode: {mode}\n")

        completed = 0
        with Executor(max_workers=args.n_workers) as pool:
            future_to_sid = {pool.submit(train_subject, job): job["subject_id"]
                             for job in jobs}
            for future in as_completed(future_to_sid):
                completed += 1
                sid = future_to_sid[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"[{completed}/{len(jobs)}] Subject {sid} FAILED: {exc}")
                    continue
                h, rem = divmod(int(result["training_time_s"]), 3600)
                m, s   = divmod(rem, 60)
                print(f"[{completed}/{len(jobs)}] Subject {sid} done — "
                      f"acc={result['accuracy']:.4f}  f1={result['f1']:.4f}  "
                      f"prec={result['precision']:.4f}  rec={result['recall']:.4f}  "
                      f"time={h:02d}:{m:02d}:{s:02d}  "
                      f"epochs={result['epochs_trained']}")
                with open(summary_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=csv_fields).writerow(result)

    wall = time.time() - t_total
    h, rem = divmod(int(wall), 3600)
    m, s   = divmod(rem, 60)
    print(f"\nAll subjects done in {h:02d}:{m:02d}:{s:02d}.")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
