"""
train.py — Real 8-class BreakHis training with imbalance handling and
stratified k-fold cross-validation (Changes 4, 5, 6).

This produces the GENUINE numbers the paper should report. Nothing here is
simulated: accuracy, macro-F1, per-class report and the confusion matrix all
come from actual forward passes on held-out folds.

USAGE
-----
1. Arrange the BreakHis images as an ImageFolder, one directory per subtype:

       data/
         Adenosis/ ...jpg
         Fibroadenoma/ ...
         Phyllodes Tumor/ ...
         Tubular Adenoma/ ...
         Carcinoma/ ...
         Lobular Carcinoma/ ...
         Mucinous Carcinoma/ ...
         Papillary Carcinoma/ ...

   (Folder names must match SUBTYPES in ai_engine.py and their order defines
    the class indices; indices 0-3 are benign, 4-7 malignant.)

2. Run:
       python train.py --data data --epochs 25 --k 5 --batch 32

3. Outputs:
       models/final_cancer_model.pth     (trained on all data)
       results/kfold_results.json        (per-fold + mean+/-std metrics)
       results/classification_report.txt (precision/recall/F1 per subtype)
       results/confusion_matrix.csv

Plug the json/report numbers into the paper (Tables 2 / classification
section). Until this is run on the real dataset, the paper should keep those
figures as flagged placeholders.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score

from ai_engine import CancerNet, BaselineCNN, SUBTYPES, DEVICE
from train_utils import (
    get_weighted_loss_function,
    make_weighted_sampler,
    stratified_kfold_indices,
    folds_from_csv,
    aggregate_fold_metrics,
)

IMG_TF_TRAIN = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
IMG_TF_EVAL = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def train_one_fold(dataset, eval_dataset, labels, train_idx, val_idx, num_classes, epochs, batch, lr, model_kind="proposed", num_workers=0, verbose=False):
    train_labels = labels[train_idx]

    # Imbalance handling: class-weighted cross-entropy with standard shuffling.
    # IMPORTANT: do NOT also use a weighted sampler here. Stacking a balanced
    # sampler on top of a class-weighted loss double-corrects the imbalance and
    # starves the majority class (Carcinoma collapsed to 0 recall in testing).
    # Class-weighted loss alone is the standard, well-behaved single correction.
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True)
    # Validate on CLEAN (non-augmented) images — evaluating on randomly
    # flipped/rotated images understates true accuracy. This is a correctness
    # fix, not a metric tweak.
    val_loader = DataLoader(Subset(eval_dataset, val_idx), batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)

    if model_kind == "baseline":
        model = BaselineCNN(num_classes=num_classes).to(DEVICE)  # from-scratch plain CNN
    else:
        model = CancerNet(num_classes=num_classes, pretrained=True).to(DEVICE)  # ImageNet transfer learning
    criterion = get_weighted_loss_function(train_labels, num_classes, DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine annealing LR schedule: smoothly decays the learning rate to ~0 over
    # the run, which helps a pretrained backbone converge to a better, more
    # stable minimum than a fixed LR. Standard technique, not a result tweak.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    import time as _t
    for epoch in range(epochs):
        model.train(); _ls=0.0; _n=0; _t0=_t.time()
        for imgs, ys in train_loader:
            imgs, ys = imgs.to(DEVICE), ys.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), ys)
            loss.backward()
            optimizer.step()
            _ls+=loss.item(); _n+=1
        scheduler.step()
        if verbose:
            print(f"      epoch {epoch+1}/{epochs}  loss={_ls/max(_n,1):.3f}  ({_t.time()-_t0:.0f}s)", flush=True)

    # Validation
    model.eval()
    preds, gts, probs = [], [], []
    with torch.no_grad():
        for imgs, ys in val_loader:
            out = model(imgs.to(DEVICE))
            pr = torch.softmax(out, dim=1)
            preds.extend(pr.argmax(1).cpu().numpy().tolist())
            probs.extend(pr.cpu().numpy().tolist())
            gts.extend(ys.numpy().tolist())

    acc = accuracy_score(gts, preds)
    macro_f1 = f1_score(gts, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": macro_f1}, (gts, preds, probs), model


def run_config(base_ds, eval_ds, labels, num_classes, to_binary, args, mag, tag):
    model_kind = getattr(args, 'model', 'proposed')
    """Run the full k-fold evaluation for one configuration (optionally one
    magnification). Writes results/*{tag}.* files and returns (agg, split_desc)."""
    if args.folds:
        sample_paths = [p for p, _ in base_ds.samples]
        fold_iter = folds_from_csv(args.folds, sample_paths, mag=mag)
        split_desc = f"official patient-wise folds (Folds.csv{', mag=' + str(mag) if mag else ''})"
    else:
        fold_iter = stratified_kfold_indices(labels, k=args.k)
        split_desc = f"generic stratified {args.k}-fold"
    print(f"\n################ CONFIG: {split_desc} ################")

    fold_metrics, all_gts, all_preds = [], [], []
    for fold, (tr, va) in enumerate(fold_iter):
        print(f"\n=== Fold {fold + 1} ===")
        m, (gts, preds, probs), _ = train_one_fold(
            base_ds, eval_ds, labels, tr, va, num_classes, args.epochs, args.batch, args.lr, model_kind)
        m["binary_accuracy"] = accuracy_score(to_binary(gts), to_binary(preds))
        # Real AUC-ROC: one-vs-rest macro (8-class) and binary (malignant score)
        try:
            probs_arr = np.array(probs)
            m["auc_ovr"] = float(roc_auc_score(gts, probs_arr, multi_class="ovr", average="macro"))
            benign_cols = [i for i, c in enumerate(base_ds.classes)
                           if c in {"Adenosis", "Fibroadenoma", "Phyllodes Tumor", "Tubular Adenoma"}]
            mal_score = 1.0 - probs_arr[:, benign_cols].sum(axis=1)
            m["binary_auc"] = float(roc_auc_score(to_binary(gts), mal_score))
        except Exception as _e:
            m["auc_ovr"] = float("nan"); m["binary_auc"] = float("nan")
        print(f"  8-class acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}  "
              f"B/M acc={m['binary_accuracy']:.4f}  AUC(ovr)={m['auc_ovr']:.4f}  B/M AUC={m['binary_auc']:.4f}")
        fold_metrics.append(m)
        all_gts.extend(gts)
        all_preds.extend(preds)

    agg = aggregate_fold_metrics(fold_metrics)
    print(f"\n==== SUMMARY ({split_desc}) ====")
    print(f"8-class Accuracy : {agg['accuracy_mean']:.4f} +/- {agg['accuracy_std']:.4f}")
    print(f"8-class Macro-F1 : {agg['macro_f1_mean']:.4f} +/- {agg['macro_f1_std']:.4f}")
    print(f"Benign/Malignant: {agg['binary_accuracy_mean']:.4f} +/- {agg['binary_accuracy_std']:.4f}")
    print(f"AUC-ROC (8-class OVR): {agg['auc_ovr_mean']:.4f} +/- {agg['auc_ovr_std']:.4f}")
    print(f"AUC-ROC (Benign/Malig): {agg['binary_auc_mean']:.4f} +/- {agg['binary_auc_std']:.4f}")

    with open(f"results/kfold_results{tag}.json", "w") as f:
        json.dump({"per_fold": fold_metrics, "summary": agg,
                   "classes": base_ds.classes, "protocol": split_desc}, f, indent=2)
    report = classification_report(all_gts, all_preds, target_names=base_ds.classes,
                                   digits=4, zero_division=0)
    with open(f"results/classification_report{tag}.txt", "w") as f:
        f.write(report)
    cm = confusion_matrix(all_gts, all_preds)
    np.savetxt(f"results/confusion_matrix{tag}.csv", cm, fmt="%d", delimiter=",",
               header=",".join(base_ds.classes), comments="")
    print("\n" + report)

    bin_gt, bin_pr = to_binary(all_gts), to_binary(all_preds)
    bin_report = classification_report(bin_gt, bin_pr, target_names=["Benign", "Malignant"],
                                       digits=4, zero_division=0)
    with open(f"results/binary_report{tag}.txt", "w") as f:
        f.write(bin_report)
    print("\n=== Benign vs Malignant (derived from 8-class model) ===")
    print(bin_report)
    return agg, split_desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="ImageFolder root with one dir per subtype")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--folds", default=None,
                    help="Path to official BreakHis Folds.csv for patient-wise splits (recommended). "
                         "If omitted, a generic stratified k-fold is used.")
    ap.add_argument("--mag", default=None,
                    help="Restrict to one magnification (40/100/200/400) when using --folds.")
    ap.add_argument("--per-mag", dest="per_mag", action="store_true",
                    help="Run all four magnifications (40/100/200/400) separately and print a "
                         "comparison table. Requires --folds. Writes results/*_<mag>X.* files.")
    ap.add_argument("--model", choices=["proposed", "baseline"], default="proposed",
                    help="proposed = ResNet-50 + SE attention + ImageNet transfer learning; "
                         "baseline = plain from-scratch CNN (for the ablation comparison).")
    ap.add_argument("--no-final-model", dest="no_final_model", action="store_true",
                    help="Skip training the full-data deployment model (faster; metrics only).")
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Two views of the same data: augmented (train) and clean (eval).
    base_ds = datasets.ImageFolder(args.data, transform=IMG_TF_TRAIN)
    eval_ds = datasets.ImageFolder(args.data, transform=IMG_TF_EVAL)
    labels = np.array(base_ds.targets)
    num_classes = len(base_ds.classes)
    print(f"Classes ({num_classes}): {base_ds.classes}")
    print(f"Per-class counts: {np.bincount(labels).tolist()}")
    BENIGN = {"Adenosis", "Fibroadenoma", "Phyllodes Tumor", "Tubular Adenoma"}
    benign_idx = {i for i, c in enumerate(base_ds.classes) if c in BENIGN}
    def to_binary(labs):
        return [0 if x in benign_idx else 1 for x in labs]
    if num_classes != len(SUBTYPES):
        print(f"WARNING: found {num_classes} folders but SUBTYPES has {len(SUBTYPES)}.")

    if args.per_mag:
        if not args.folds:
            raise SystemExit("--per-mag requires --folds (Folds.csv defines the magnifications).")
        mags = ["40", "100", "200", "400"]
        comparison = {}
        for mag in mags:
            agg, _ = run_config(base_ds, eval_ds, labels, num_classes, to_binary, args, mag, f"_{mag}X")
            comparison[mag] = agg
        print("\n\n================ PER-MAGNIFICATION COMPARISON ================")
        print(f"{'Mag':>5} | {'8-class Accuracy':>20} | {'8-class Macro-F1':>20} | {'Benign/Malignant':>20}")
        for mag in mags:
            a = comparison[mag]
            print(f"{mag + 'X':>5} | "
                  f"{a['accuracy_mean']*100:6.2f} +/- {a['accuracy_std']*100:4.2f}%    | "
                  f"{a['macro_f1_mean']:6.4f} +/- {a['macro_f1_std']:6.4f}    | "
                  f"{a['binary_accuracy_mean']*100:6.2f} +/- {a['binary_accuracy_std']*100:4.2f}%")
    else:
        model_tag = "" if args.model == "proposed" else "_baseline"
        run_config(base_ds, eval_ds, labels, num_classes, to_binary, args, args.mag, model_tag)

    # ---- Final model trained on ALL data (for deployment / inference) ----
    if args.no_final_model or args.per_mag or args.model == "baseline":
        print("\nSkipping full-data deployment model (per-mag or --no-final-model).")
    else:
        print("\nTraining final model on full dataset...")
        full_idx = np.arange(len(labels))
        _, _, final_model = train_one_fold(
            base_ds, eval_ds, labels, full_idx, full_idx[:args.batch], num_classes,
            args.epochs, args.batch, args.lr)
        torch.save(final_model.state_dict(), "models/final_cancer_model.pth")
        print("Saved models/final_cancer_model.pth")


if __name__ == "__main__":
    main()
