"""
r10_baselines.py - Reviewer-4 comment #10 ablation baselines.

Adds the two baselines the reviewer asked for, to isolate each component,
using the SAME patient-wise 5-fold protocol, transforms, optimiser, cosine
schedule and inverse-frequency class-weighted loss as train.py (so they are
directly comparable to the proposed ResNet-50 + SE model):

  resnet50_nose : pretrained ResNet-50, NO squeeze-excitation, 8-class head
                  (identical to CancerNet except SE block -> Identity).
                  Isolates the contribution of the SE attention.
  direct_binary : pretrained ResNet-50 + SE, trained DIRECTLY on the two
                  benign/malignant labels (2-class head), instead of the
                  8-class-then-derive route the proposed model uses.
                  Isolates the direct-binary vs 8-class-derive choice.

Outputs: results/kfold_results_resnet50_nose.json
         results/kfold_results_direct_binary.json
(per-fold + mean/std: binary acc, binary AUC, sensitivity, specificity; the
nose model also reports 8-class accuracy / macro-F1 / OVR-AUC.)

REPRODUCIBILITY (added, methodology unchanged): --seed (default 42) seeds
Python/NumPy/PyTorch/CUDA and a per-fold DataLoader generator + worker_init_fn,
with cudnn.deterministic=True, so a rerun with the same seed reproduces the
reported numbers. Each fold uses seed+fold. SPEED (added): persistent_workers
+ pin_memory on the training loader. NO change to folds, transforms, optimiser,
schedule, loss or model definitions.

FULL RUN (GPU server):
  python r10_baselines.py --data "<BreakHis_8class>" --folds "<archive\\Folds.csv>" --epochs 25 --batch 32 --seed 42
SMOKE TEST (fast, correctness only - NOT for reporting):
  python r10_baselines.py --data "<...>" --folds "<...>" --quick
"""
import argparse, json, os, time, random                                          # [seed] +random
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from ai_engine import CancerNet, BENIGN_SUBTYPES, DEVICE
from train_utils import get_weighted_loss_function, folds_from_csv, aggregate_fold_metrics


def set_seed(seed):                                                              # [seed]
    """Seed every RNG + make cuDNN deterministic so runs are reproducible."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id):                                                     # [seed]
    ws = torch.initial_seed() % (2 ** 32)
    np.random.seed(ws); random.seed(ws)


TF_TRAIN = transforms.Compose([
    transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), transforms.RandomRotation(15), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
TF_EVAL = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def build_nose(num_classes):
    m = CancerNet(num_classes=num_classes, pretrained=True)
    m.attention = nn.Identity()            # remove squeeze-excitation ONLY
    return m

def build_binary(num_classes):
    return CancerNet(num_classes=num_classes, pretrained=True)   # SE kept, 2-class head

class Binarize(torch.utils.data.Dataset):
    """Wrap an ImageFolder so its label becomes 0=benign / 1=malignant."""
    def __init__(self, ds, benign_idx):
        self.ds = ds; self.benign = set(benign_idx)
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        x, y = self.ds[i]; return x, (0 if y in self.benign else 1)

def train_eval_fold(builder, num_classes, tr, va, base_ds, eval_ds, y_train_all, epochs, batch, lr, quick, seed):
    set_seed(seed)                         # [seed] reproducible init/shuffle/augmentation for this fold
    if quick:                              # smoke test: RANDOM tiny subset, 2 epochs
        _r = np.random.default_rng(0)
        tr = _r.permutation(tr)[:min(len(tr), 384)]; va = _r.permutation(va)[:min(len(va), 256)]; epochs = 2
    yl = y_train_all[tr]
    g = torch.Generator(); g.manual_seed(seed)                                   # [seed] reproducible shuffle order
    tl = DataLoader(Subset(base_ds, tr), batch_size=batch, shuffle=True, num_workers=16,
                    pin_memory=True, persistent_workers=True,                    # [speed]
                    generator=g, worker_init_fn=_seed_worker)                    # [seed]
    vl = DataLoader(Subset(eval_ds, va), batch_size=batch, shuffle=False, num_workers=16,
                    pin_memory=True, worker_init_fn=_seed_worker)                # [speed]/[seed]
    model = builder(num_classes).to(DEVICE)                                      # init is now deterministic (set_seed above)
    crit = get_weighted_loss_function(yl, num_classes, DEVICE)      # inverse-freq weighted CE
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train(); t0 = time.time(); ls = 0.0; n = 0
        for imgs, ys in tl:
            imgs, ys = imgs.to(DEVICE), ys.to(DEVICE)
            opt.zero_grad(); loss = crit(model(imgs), ys); loss.backward(); opt.step()
            ls += loss.item(); n += 1
        sch.step()
        print(f"      epoch {ep+1}/{epochs} loss={ls/max(n,1):.3f} ({time.time()-t0:.0f}s)", flush=True)
    model.eval(); P = []; G = []; PR = []
    with torch.no_grad():
        for imgs, ys in vl:                # validate on CLEAN (non-augmented) images
            out = torch.softmax(model(imgs.to(DEVICE)), dim=1)
            P.extend(out.argmax(1).cpu().numpy().tolist())
            PR.extend(out.cpu().numpy().tolist()); G.extend(ys.numpy().tolist())
    return np.array(G), np.array(P), np.array(PR)

def binary_metrics(gt_bin, mal_score, pred_bin):
    tn, fp, fn, tp = confusion_matrix(gt_bin, pred_bin, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float('nan')
    spec = tn / (tn + fp) if (tn + fp) else float('nan')
    try: auc = float(roc_auc_score(gt_bin, mal_score))
    except Exception: auc = float('nan')
    return dict(binary_accuracy=float((tp + tn) / max(tp + tn + fp + fn, 1)), binary_auc=auc,
                sensitivity=float(sens), specificity=float(spec),
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))

def run(which, base_ds, eval_ds, y8, benign_idx, folds_csv, sample_paths, epochs, batch, lr, quick, seed):
    is_bin = (which == "direct_binary")
    num_classes = 2 if is_bin else 8
    builder = build_binary if is_bin else build_nose
    ybin = np.array([0 if x in benign_idx else 1 for x in y8])
    y_train_all = ybin if is_bin else y8
    tr_ds = Binarize(base_ds, benign_idx) if is_bin else base_ds   # feed 0/1 labels for direct-binary
    ev_ds = Binarize(eval_ds, benign_idx) if is_bin else eval_ds
    folds = []
    for fold, (tr, va) in enumerate(folds_from_csv(folds_csv, sample_paths)):
        print(f"\n=== [{which}] Fold {fold+1} ({len(tr)} train / {len(va)} val) ===", flush=True)
        G, Pp, PR = train_eval_fold(builder, num_classes, tr, va, tr_ds, ev_ds,
                                    y_train_all, epochs, batch, lr, quick, seed + fold)   # [seed] per-fold seed
        if is_bin:
            m = binary_metrics(G, PR[:, 1], Pp)
        else:
            gt_bin = np.array([0 if x in benign_idx else 1 for x in G])
            pred_bin = np.array([0 if x in benign_idx else 1 for x in Pp])
            mal_score = 1.0 - PR[:, sorted(benign_idx)].sum(axis=1)
            m = binary_metrics(gt_bin, mal_score, pred_bin)
            m["accuracy"] = float(accuracy_score(G, Pp))
            m["macro_f1"] = float(f1_score(G, Pp, average="macro", zero_division=0))
            try: m["auc_ovr"] = float(roc_auc_score(G, PR, multi_class="ovr", average="macro"))
            except Exception: m["auc_ovr"] = float('nan')
        print("    fold metrics:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, flush=True)
        folds.append(m)
        if quick: break
    agg = aggregate_fold_metrics(folds)
    out = {"baseline": which, "num_classes": num_classes, "quick": quick, "seed": seed,   # [seed] recorded
           "protocol": "official patient-wise folds (Folds.csv)", "per_fold": folds, "summary": agg}
    path = os.path.join("results", f"kfold_results_{which}.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nWROTE {path}")
    print("SUMMARY:", {k: round(v, 4) for k, v in agg.items() if isinstance(v, float)}, flush=True)
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)                             # [seed]
    ap.add_argument("--which", choices=["resnet50_nose", "direct_binary", "both"], default="both")
    ap.add_argument("--quick", action="store_true", help="fast smoke test (1 fold, 2 epochs, subset)")
    a = ap.parse_args()
    set_seed(a.seed)                                                            # [seed] global seed before dataset build
    os.makedirs("results", exist_ok=True)
    base_ds = datasets.ImageFolder(a.data, transform=TF_TRAIN)
    eval_ds = datasets.ImageFolder(a.data, transform=TF_EVAL)
    y8 = np.array(base_ds.targets)
    benign_idx = {i for i, c in enumerate(base_ds.classes) if c in BENIGN_SUBTYPES}
    sample_paths = [p for p, _ in base_ds.samples]
    print("Classes:", base_ds.classes)
    print("Counts:", np.bincount(y8).tolist(), "| benign idx:", sorted(benign_idx),
          "| DEVICE:", DEVICE, "| seed:", a.seed, "| quick:", a.quick)          # [seed] print seed
    todo = ["resnet50_nose", "direct_binary"] if a.which == "both" else [a.which]
    for w in todo:
        run(w, base_ds, eval_ds, y8, benign_idx, a.folds, sample_paths, a.epochs, a.batch, a.lr, a.quick, a.seed)  # [seed]

if __name__ == "__main__":
    main()
