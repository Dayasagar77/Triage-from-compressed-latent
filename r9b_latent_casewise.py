"""
r9b_latent_casewise.py - Reviewer-4 comments #5, #9 and #11 (latent triage, repeatable, with saved predictions).

Same model and protocol as r2_caselevel.py (codec lambda=0.01 fixed; latent head retrained on each official
fold's training images; 20 epochs, Adam 1e-3, batch 64, inverse-frequency weighted CE), but
  * seeds fixed, so the numbers are repeatable;
  * every held-out prediction is SAVED (results/latent_heldout_predictions.csv) so any aggregation can be audited;
  * binary operating characteristics at several thresholds (comment #11);
  * two case-level aggregations (comment #9):
      A. pooled  - mean malignant score over ALL held-out images of a patient, across every fold in which
                   that patient was a test patient (as r2_caselevel.py);
      B. per-fold - mean score over the patient's images within one fold (one case per fold-patient pair);
    95% CIs by bootstrap over patients (cluster bootstrap for B);
  * a sensitivity analysis for patient identity (comment #5): the official folds separate slide folders
    (the dataset's patient unit); nine numeric IDs recur with letter suffixes (e.g. 22549AB/CD/G) and one ID
    (13412) has two tumour-class folders. The analysis is repeated after dropping, from each fold's test set,
    every image whose numeric ID also occurs in that fold's training set.

    python r9b_latent_casewise.py --data "<BreakHis_8class>" --folds "<Folds.csv>"
Output: results/latent_casewise_results.json, results/latent_heldout_predictions.csv
"""
import argparse, csv, json, os, re
from collections import defaultdict
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from learned_codec import FactorizedCodec
from train_utils import folds_from_csv, get_weighted_loss_function

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
BENIGN = {"Adenosis", "Fibroadenoma", "Phyllodes Tumor", "Tubular Adenoma"}
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class LatentHead(nn.Module):
    def __init__(self, in_ch, n=8):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                 nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n))

    def forward(self, x): return self.net(x)


def ids_of(path):
    pid = os.path.splitext(os.path.basename(path))[0].split('-')[2]      # slide-folder ID, e.g. 22549AB
    return pid, re.match(r"(\d+)", pid).group(1)                          # numeric ID, e.g. 22549


def binm(y, s, thr=0.5):
    y = np.asarray(y); s = np.asarray(s); p = (s >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    return dict(n=int(len(y)), acc=(tp + tn) / len(y), sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
                auc=float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else float("nan"), tp=tp, tn=tn, fp=fp, fn=fn)


def rnd(d): return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def boot_ci(groups, ys, ss, B=2000, seed=0):
    """Cluster bootstrap over the unit in `groups` (patient); ys/ss are per-case arrays."""
    rng = np.random.default_rng(seed); uniq = np.unique(groups); where = {g: np.where(groups == g)[0] for g in uniq}
    acc = {k: [] for k in ("acc", "sens", "spec", "auc")}
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True); idx = np.concatenate([where[g] for g in pick])
        m = binm(ys[idx], ss[idx])
        for k in acc: acc[k].append(m[k])
    return {k: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)] for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--folds", required=True)
    ap.add_argument("--codec", default=os.path.join("models", "learned_codec_l0.01.pth"))
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    codec = FactorizedCodec().to(DEVICE)
    codec.load_state_dict(torch.load(a.codec, map_location=DEVICE, weights_only=True)); codec.eval()
    ds = datasets.ImageFolder(a.data, transform=TF); classes = ds.classes; labels = np.array(ds.targets)
    mal_cols = [i for i, c in enumerate(classes) if c not in BENIGN]; benign_idx = {i for i, c in enumerate(classes) if c in BENIGN}
    paths = [p for p, _ in ds.samples]
    feats = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=32, shuffle=False, num_workers=0):
            feats.append(codec.encode_latent(x.to(DEVICE)).cpu())
    feats = torch.cat(feats); in_ch = feats.shape[1]

    rows = []
    for fold, (tr, va) in enumerate(folds_from_csv(a.folds, paths), start=1):
        torch.manual_seed(fold)
        head = LatentHead(in_ch, len(classes)).to(DEVICE)
        crit = get_weighted_loss_function(labels[tr], len(classes), DEVICE)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
        g = torch.Generator().manual_seed(fold)
        tl = DataLoader(TensorDataset(feats[tr], torch.tensor(labels[tr])), batch_size=64, shuffle=True, generator=g)
        for ep in range(20):
            head.train()
            for fx, fy in tl:
                opt.zero_grad(); crit(head(fx.to(DEVICE)), fy.to(DEVICE)).backward(); opt.step()
            sch.step()
        head.eval()
        with torch.no_grad():
            pr = torch.softmax(head(feats[va].to(DEVICE)), 1).cpu().numpy()
        train_num = {ids_of(paths[i])[1] for i in tr}
        for k, i in enumerate(va):
            pid, num = ids_of(paths[i])
            rows.append(dict(fold=fold, file=os.path.basename(paths[i]), patient=pid, numeric_id=num,
                             y=0 if labels[i] in benign_idx else 1, score=float(pr[k, mal_cols].sum()),
                             numeric_id_in_train=int(num in train_num)))
        print(f"fold {fold}: {len(va)} held-out images", flush=True)

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "latent_heldout_predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    def analyse(rs):
        y = np.array([r["y"] for r in rs]); s = np.array([r["score"] for r in rs])
        out = {"patch": rnd(binm(y, s)), "patch_thresholds": {str(t): rnd(binm(y, s, t)) for t in THRESHOLDS},
               "per_fold_patch": {str(f): rnd(binm(np.array([r["y"] for r in rs if r["fold"] == f]),
                                                   np.array([r["score"] for r in rs if r["fold"] == f])))
                                  for f in sorted({r["fold"] for r in rs})}}
        # A: pooled over folds, one case per patient
        agg = defaultdict(list); lab = {}
        for r in rs: agg[r["patient"]].append(r["score"]); lab[r["patient"]] = r["y"]
        pts = sorted(agg); py = np.array([lab[p] for p in pts]); ps = np.array([np.mean(agg[p]) for p in pts])
        out["case_pooled"] = rnd(binm(py, ps)) | {"ci95": boot_ci(np.array(pts), py, ps)}
        out["case_pooled_thresholds"] = {str(t): rnd(binm(py, ps, t)) for t in THRESHOLDS}
        # B: per fold, one case per fold-patient pair; cluster bootstrap over patients
        agg2 = defaultdict(list); lab2 = {}
        for r in rs: agg2[(r["fold"], r["patient"])].append(r["score"]); lab2[(r["fold"], r["patient"])] = r["y"]
        keys = sorted(agg2); by = np.array([lab2[k] for k in keys]); bs = np.array([np.mean(agg2[k]) for k in keys])
        out["case_per_fold"] = rnd(binm(by, bs)) | {"ci95_cluster": boot_ci(np.array([k[1] for k in keys]), by, bs)}
        return out

    res = {"note": "latent triage, codec lambda=0.01 trained image-wise (see r5 for the per-fold codec); head retrained per official fold; seeds fixed",
           "all_heldout": analyse(rows),
           "excluding_numeric_id_in_train": analyse([r for r in rows if not r["numeric_id_in_train"]]),
           "n_images_excluded": int(sum(r["numeric_id_in_train"] for r in rows)), "n_heldout_images": len(rows)}
    json.dump(res, open(os.path.join("results", "latent_casewise_results.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("n_heldout_images", "n_images_excluded")}))
    for part in ("all_heldout", "excluding_numeric_id_in_train"):
        r = res[part]; print(part, "patch", r["patch"], "\n  case_pooled", r["case_pooled"], "\n  case_per_fold", r["case_per_fold"])
    print("WROTE results/latent_casewise_results.json")


if __name__ == "__main__":
    main()
