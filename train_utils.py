"""
Training utilities for the AI-augmented telepathology framework.

Provides GENUINE helpers for class-imbalance handling (Change 5) and
statistically reliable k-fold cross-validation (Change 6). The previous version
returned randomly-generated "accuracies"; that has been removed. These functions
operate on real labels/datasets and are driven by train.py, which performs actual
training and produces the numbers reported in the paper.
"""
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

try:
    from sklearn.model_selection import StratifiedKFold
except Exception:
    StratifiedKFold = None


# ---- Change 5: class-imbalance handling (generalised to N classes) ----
def get_class_weights(labels, num_classes=None):
    """Inverse-frequency class weights: weight_c = total / (num_classes * count_c)."""
    labels = np.asarray(labels)
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    total = counts.sum()
    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def get_weighted_loss_function(labels, num_classes=None, device="cpu"):
    weights = get_class_weights(labels, num_classes).to(device)
    return nn.CrossEntropyLoss(weight=weights)


def make_weighted_sampler(labels):
    """WeightedRandomSampler so each mini-batch is roughly class-balanced."""
    labels = np.asarray(labels)
    counts = Counter(labels.tolist())
    sample_weights = np.array([1.0 / counts[int(l)] for l in labels], dtype=np.float64)
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---- Change 6: real stratified k-fold splits (no simulation) ----
def stratified_kfold_indices(labels, k=5, seed=42):
    if StratifiedKFold is None:
        raise ImportError("scikit-learn is required for stratified_kfold_indices")
    labels = np.asarray(labels)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        yield train_idx, val_idx


def aggregate_fold_metrics(fold_metrics):
    out = {}
    for kname in fold_metrics[0].keys():
        vals = np.array([m[kname] for m in fold_metrics], dtype=np.float64)
        out[f"{kname}_mean"] = float(vals.mean())
        out[f"{kname}_std"] = float(vals.std())
    return out


# ---- Official BreakHis patient-wise folds (Folds.csv) ----
def folds_from_csv(csv_path, sample_paths, mag=None):
    """Yield (train_idx, val_idx) per official BreakHis fold from Folds.csv.

    Folds.csv columns: fold, mag, grp(train/test), filename. Splitting follows
    the dataset authors' patient-wise protocol (no patient appears in both train
    and test of a fold), which prevents patient leakage and makes results
    comparable to prior BreakHis papers. `mag` optionally restricts to one
    magnification ('40','100','200','400'). Matching is by file basename against
    the ImageFolder sample paths.
    """
    import csv, os
    idx_of = {os.path.basename(p): i for i, p in enumerate(sample_paths)}
    folds = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if mag is not None and str(row["mag"]) != str(mag):
                continue
            bn = os.path.basename(row["filename"])
            i = idx_of.get(bn)
            if i is None:
                continue  # image not present in this prepared dataset
            fold = int(row["fold"])
            folds.setdefault(fold, {"train": [], "test": []})[row["grp"]].append(i)
    for fold in sorted(folds):
        tr = np.array(folds[fold]["train"], dtype=int)
        va = np.array(folds[fold]["test"], dtype=int)
        if len(tr) and len(va):
            yield tr, va


if __name__ == "__main__":
    demo = np.concatenate([np.full(400, 0), np.full(50, 1), np.full(120, 2)])
    print("Class weights:", get_class_weights(demo, num_classes=3).tolist())
    print("Sampler len:", len(make_weighted_sampler(demo)))
    for i, (tr, va) in enumerate(stratified_kfold_indices(demo, k=5)):
        print(f"Fold {i+1}: train={len(tr)} val={len(va)}")
