"""
r13_source_predictions.py - Reviewer-4 comments #1 (E1 operating point), #9 (patient-level uncertainty of the
source classifier) and the proposed arm for #10. GPU job (run on the server with r10_baselines.py).

Re-trains the PROPOSED source classifier (CancerNet: ImageNet ResNet-50 + squeeze-and-excitation, 8 classes) on
each official split's training images with the protocol of train.py / r10_baselines.py (same transforms, Adam
1e-4, cosine schedule, 25 epochs, batch 32, inverse-frequency weighted cross-entropy), with FIXED seeds, and SAVES
every held-out prediction so that any aggregation can be audited:

  results/source_heldout_predictions.csv   one row per (split, test image): 8 probabilities, malignant score, flags
  results/source_casewise_results.json     field-level and patient-level benign/malignant metrics, 95% CIs

Decision rules reported
  flag_argmax : the deployed rule (analyze_image): most probable subtype, grouped benign/malignant -> used for E1
  flag_score  : summed malignant probability >= 0.5 (the rule used for latent triage)
Patient level : mean malignant score over all held-out images of a patient, pooled over the splits in which the
                patient is a test patient (as for latent triage), threshold 0.5; 95% bootstrap CI over patients.

    python r13_source_predictions.py --data "<BreakHis_8class>" --folds "<archive/Folds.csv>" --seed 42
Smoke test (correctness only, NOT for reporting): add --quick --no-pretrained
"""
import argparse, csv, json, os, random, time
os.environ.setdefault("AI_ENGINE_LITE", "1")            # import the model class without loading deployed weights
import numpy as np, torch
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, f1_score
from ai_engine import CancerNet, BENIGN_SUBTYPES, DEVICE
from train_utils import get_weighted_loss_function, folds_from_csv

TF_TRAIN = transforms.Compose([
    transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), transforms.RandomRotation(15), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
TF_EVAL = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def binm(y, s, flag):
    y = np.asarray(y); flag = np.asarray(flag)
    tp = int(((flag == 1) & (y == 1)).sum()); tn = int(((flag == 0) & (y == 0)).sum())
    fp = int(((flag == 1) & (y == 0)).sum()); fn = int(((flag == 0) & (y == 1)).sum())
    return dict(n=int(len(y)), acc=(tp + tn) / len(y), sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
                auc=float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else float("nan"), tp=tp, tn=tn, fp=fp, fn=fn)


def boot(groups, y, s, B=2000, seed=0):
    rng = np.random.default_rng(seed); uniq = np.unique(groups); where = {g: np.where(groups == g)[0] for g in uniq}
    acc = {k: [] for k in ("acc", "sens", "spec", "auc")}
    for _ in range(B):
        idx = np.concatenate([where[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        m = binm(y[idx], s[idx], (s[idx] >= 0.5).astype(int))
        for k in acc: acc[k].append(m[k])
    return {k: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)] for k, v in acc.items()}


def rnd(d): return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--folds", required=True)
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 split, 2 epochs, small subset")
    ap.add_argument("--no-pretrained", action="store_true", help="smoke test only: skip the ImageNet download")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    base_ds = datasets.ImageFolder(a.data, transform=TF_TRAIN); eval_ds = datasets.ImageFolder(a.data, transform=TF_EVAL)
    classes = base_ds.classes; y8 = np.array(base_ds.targets); paths = [p for p, _ in base_ds.samples]
    benign = sorted(i for i, c in enumerate(classes) if c in BENIGN_SUBTYPES); mal = [i for i in range(len(classes)) if i not in benign]
    print("classes:", classes, "| benign idx:", benign, "| device:", DEVICE, "| seed:", a.seed, flush=True)
    rows = []
    for fold, (tr, va) in enumerate(folds_from_csv(a.folds, paths), start=1):
        seed_all(a.seed + fold)
        epochs = a.epochs
        if a.quick:
            r_ = np.random.default_rng(0); tr = r_.permutation(tr)[:min(len(tr), 96)]; va = r_.permutation(va)[:min(len(va), 64)]; epochs = 2
        g = torch.Generator().manual_seed(a.seed + fold)
        tl = DataLoader(Subset(base_ds, tr.tolist()), batch_size=a.batch, shuffle=True, num_workers=16, generator=g)
        vl = DataLoader(Subset(eval_ds, va.tolist()), batch_size=a.batch, shuffle=False, num_workers=16)
        model = CancerNet(num_classes=len(classes), pretrained=not a.no_pretrained).to(DEVICE)
        crit = get_weighted_loss_function(y8[tr], len(classes), DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=a.lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for ep in range(epochs):
            model.train(); t0 = time.time(); ls = 0.0; n = 0
            for x, y in tl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step(); ls += loss.item(); n += 1
            sch.step(); print(f"  split {fold} epoch {ep + 1}/{epochs} loss={ls / max(n, 1):.3f} ({time.time() - t0:.0f}s)", flush=True)
        model.eval(); PR = []
        with torch.no_grad():
            for x, _ in vl: PR.append(torch.softmax(model(x.to(DEVICE)), 1).cpu().numpy())
        PR = np.concatenate(PR)
        for k, i in enumerate(va):
            f = os.path.basename(paths[i]); pid = os.path.splitext(f)[0].split('-')[2]
            p = PR[k]; score = float(p[mal].sum()); arg = int(p.argmax())
            rows.append(dict(split=fold, file=f, patient=pid, y8=int(y8[i]), y=int(y8[i] in mal), score=score,
                             flag_argmax=int(arg in mal), flag_score=int(score >= 0.5), pred8=arg,
                             **{f"p_{c.replace(' ', '_')}": float(p[j]) for j, c in enumerate(classes)}))
        print(f"split {fold}: {len(va)} held-out images", flush=True)
        if a.quick: break
    with open(os.path.join("results", "source_heldout_predictions.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    y = np.array([r["y"] for r in rows]); s = np.array([r["score"] for r in rows])
    fa = np.array([r["flag_argmax"] for r in rows]); fs = np.array([r["flag_score"] for r in rows])
    agg = defaultdict(list); lab = {}
    for r in rows: agg[r["patient"]].append(r["score"]); lab[r["patient"]] = r["y"]
    pts = sorted(agg); py = np.array([lab[p] for p in pts]); ps = np.array([np.mean(agg[p]) for p in pts])
    if not a.quick:                                        # a patient's images must share one benign/malignant label
        bad = sorted({r["patient"] for r in rows if r["y"] != lab[r["patient"]]})
        assert not bad, f"patients with mixed benign/malignant labels: {bad}"
    per_split = {}
    for f in sorted({r["split"] for r in rows}):
        rs = [r for r in rows if r["split"] == f]
        m = binm([r["y"] for r in rs], [r["score"] for r in rs], [r["flag_argmax"] for r in rs])
        m["acc8"] = float(np.mean([r["pred8"] == r["y8"] for r in rs]))
        m["macro_f1"] = float(f1_score([r["y8"] for r in rs], [r["pred8"] for r in rs], average="macro", zero_division=0))
        per_split[str(f)] = rnd(m)
    out = {"note": "proposed source classifier re-trained per official split with fixed seeds; every held-out prediction saved",
           "seed": a.seed, "epochs": a.epochs, "quick": a.quick, "n_predictions": len(rows), "n_patients": len(pts),
           "field_argmax": rnd(binm(y, s, fa)), "field_score": rnd(binm(y, s, fs)),
           "field_ci95_cluster_argmax_note": "field-level CIs below resample patients (cluster bootstrap), score >= 0.5 rule",
           "field_ci95_cluster": boot(np.array([r["patient"] for r in rows]), y, s),
           "patient": rnd(binm(py, ps, (ps >= 0.5).astype(int))), "patient_ci95": boot(np.array(pts), py, ps),
           "per_split_field_argmax": per_split,
           "per_split_summary": {k: {"mean": float(np.mean([v[k] for v in per_split.values()])), "sd": float(np.std([v[k] for v in per_split.values()]))}
                                 for k in ("acc8", "macro_f1", "acc", "auc", "sens", "spec")}}
    json.dump(out, open(os.path.join("results", "source_casewise_results.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("n_predictions", "n_patients", "field_argmax", "patient", "patient_ci95")}, indent=1))
    print("WROTE results/source_casewise_results.json and results/source_heldout_predictions.csv")


if __name__ == "__main__":
    main()
