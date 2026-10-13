"""
r5_codec_perfold.py - Reviewer-4 comment #5 (leakage-free latent triage).

Fixes the feature-extractor leakage: for each patient-wise fold, the codec is
retrained on ONLY that fold's TRAINING patients, then latents for the held-out
patients are extracted with that fold's codec and classified. The codec therefore
never sees the test patients, so the latent-triage number is fully patient-wise.
Compares against the leaky (single image-wise codec) result.

REPRODUCIBILITY (added, methodology unchanged): --seed (default 42) seeds
Python/NumPy/PyTorch/CUDA and per-fold DataLoader generators + worker_init_fn,
with cudnn.deterministic=True, so a rerun with the same seed reproduces the
reported numbers. Codec stream uses seed+fold; head stream uses seed+1000+fold.
SPEED (added): persistent_workers + pin_memory on the codec loader, pin_memory
on the encode loader. NO change to folds, transforms, codec/head definitions,
optimiser, schedule or loss.

This is a GPU job (codec retrain x5 folds). Run on the server alongside r10_baselines.py:
  python r5_codec_perfold.py --data <BreakHis_8class> --folds <archive/Folds.csv> --codec-epochs 40 --lmbda 0.01 --seed 42
Smoke test (fast, correctness only): add --quick
Output: results/latent_leakagefree_results.json
"""
import argparse, json, os, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F  # [seed] +random
from collections import defaultdict
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from learned_codec import FactorizedCodec
from train_utils import folds_from_csv, get_weighted_loss_function

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
BENIGN = {"Adenosis", "Fibroadenoma", "Phyllodes Tumor", "Tubular Adenoma"}


def set_seed(seed):                                                              # [seed]
    """Seed every RNG + make cuDNN deterministic so runs are reproducible."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id):                                                     # [seed]
    ws = torch.initial_seed() % (2 ** 32)
    np.random.seed(ws); random.seed(ws)


class LatentHead(nn.Module):
    def __init__(self, in_ch, n=8):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n))
    def forward(self, x): return self.net(x)

def patient_of(path):
    parts = os.path.splitext(os.path.basename(path))[0].split('-')
    return parts[2] if len(parts) >= 3 else os.path.basename(path)

def train_codec(idx, ds, lmbda, epochs, batch, quick, seed):
    set_seed(seed)                                                               # [seed] deterministic codec init/shuffle
    if quick: idx = idx[:min(len(idx), 300)]; epochs = 2
    codec = FactorizedCodec().to(DEVICE)
    opt = torch.optim.Adam(codec.parameters(), lr=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator(); g.manual_seed(seed)                                   # [seed] reproducible shuffle order
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=batch, shuffle=True, num_workers=16,
                        pin_memory=True, persistent_workers=True,                # [speed]
                        generator=g, worker_init_fn=_seed_worker)                # [seed]
    for ep in range(epochs):
        codec.train()
        for x, _ in loader:
            x = x.to(DEVICE); npix = x.shape[0]*x.shape[2]*x.shape[3]
            opt.zero_grad(); out = codec(x)
            loss = lmbda*(255**2)*F.mse_loss(out["x_hat"], x) + out["bits"]/npix
            loss.backward(); torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0); opt.step()
        sch.step()
    codec.eval()
    for p in codec.parameters(): p.requires_grad_(False)
    return codec

def encode(codec, ds, idx, batch=32):
    feats = []
    with torch.no_grad():
        for x, _ in DataLoader(Subset(ds, idx.tolist()), batch_size=batch, shuffle=False,
                               num_workers=16, pin_memory=True):                 # [speed] (shuffle=False -> order preserved)
            feats.append(codec.encode_latent(x.to(DEVICE)).cpu())
    return torch.cat(feats)

def metrics(y, s, thr=0.5):
    yp = (s >= thr).astype(int)
    tp=int(((yp==1)&(y==1)).sum()); tn=int(((yp==0)&(y==0)).sum())
    fp=int(((yp==1)&(y==0)).sum()); fn=int(((yp==0)&(y==1)).sum())
    return dict(acc=(tp+tn)/len(y), sens=tp/max(tp+fn,1), spec=tn/max(tn+fp,1),
                auc=float(roc_auc_score(y,s)) if len(set(y))>1 else float('nan'))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--folds", required=True)
    ap.add_argument("--lmbda", type=float, default=0.01)
    ap.add_argument("--codec-epochs", type=int, default=40)
    ap.add_argument("--head-epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)                             # [seed]
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    set_seed(a.seed)                                                            # [seed] global seed before dataset build
    ds = datasets.ImageFolder(a.data, transform=TF)
    classes = ds.classes; labels = np.array(ds.targets)
    benign_idx = {i for i, c in enumerate(classes) if c in BENIGN}
    mal_cols = [i for i in range(len(classes)) if i not in benign_idx]
    to_bin = lambda arr: np.array([0 if x in benign_idx else 1 for x in arr])
    paths = [p for p, _ in ds.samples]
    print("DEVICE:", DEVICE, "| seed:", a.seed, "| quick:", a.quick)            # [seed] print seed

    img_pid, img_y, img_s = [], [], []
    for fold, (tr, va) in enumerate(folds_from_csv(a.folds, paths)):
        print(f"\n=== Fold {fold+1}: training codec on {len(tr)} TRAIN-patient images only ===", flush=True)
        codec = train_codec(tr, ds, a.lmbda, a.codec_epochs, a.batch, a.quick, a.seed + fold)   # [seed] codec stream
        ftr = encode(codec, ds, tr); fva = encode(codec, ds, va); in_ch = ftr.shape[1]
        set_seed(a.seed + 1000 + fold)                                          # [seed] separate head stream
        head = LatentHead(in_ch, len(classes)).to(DEVICE)
        crit = get_weighted_loss_function(labels[tr], len(classes), DEVICE)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.head_epochs)
        g_head = torch.Generator(); g_head.manual_seed(a.seed + 1000 + fold)    # [seed] reproducible head shuffle
        tl = DataLoader(TensorDataset(ftr, torch.tensor(labels[tr])), batch_size=64, shuffle=True, generator=g_head)  # [seed]
        ep_h = 2 if a.quick else a.head_epochs
        for ep in range(ep_h):
            head.train()
            for fx, fy in tl:
                opt.zero_grad(); crit(head(fx.to(DEVICE)), fy.to(DEVICE)).backward(); opt.step()
            sch.step()
        head.eval()
        with torch.no_grad():
            pr = torch.softmax(head(fva.to(DEVICE)), 1).cpu().numpy()
        for k, idx in enumerate(va):
            img_pid.append(patient_of(paths[idx])); img_y.append(int(to_bin([labels[idx]])[0]))
            img_s.append(float(pr[k, mal_cols].sum()))
        m = metrics(np.array([img_y[-len(va):]][0]), np.array(img_s[-len(va):]))
        print(f"  fold {fold+1} held-out (leakage-free): " + str({k: round(v,4) for k,v in m.items()}), flush=True)
        if a.quick: break

    img_y = np.array(img_y); img_s = np.array(img_s)
    agg = defaultdict(list); plab = {}
    for pid, y, s in zip(img_pid, img_y, img_s): agg[pid].append(s); plab[pid] = y
    pats = sorted(agg); pat_y = np.array([plab[p] for p in pats]); pat_s = np.array([np.mean(agg[p]) for p in pats])
    patch = metrics(img_y, img_s); patient = metrics(pat_y, pat_s)
    print("\n==== LEAKAGE-FREE latent triage (codec retrained per fold) ====")
    print("PATCH  :", {k: round(v,4) for k,v in patch.items()})
    print("PATIENT:", {k: round(v,4) for k,v in patient.items()})
    print("(compare: leaky image-wise codec gave patch AUC ~0.879, patient AUC ~0.956)")
    out = {"note": "codec retrained per fold on train patients only -> fully patient-wise latent triage",
           "seed": a.seed,                                                       # [seed] recorded
           "lmbda": a.lmbda, "codec_epochs": a.codec_epochs, "quick": a.quick,
           "n_patches": len(img_y), "n_patients": len(pats),
           "patch_level": {k: round(v,4) for k,v in patch.items()},
           "patient_level": {k: round(v,4) for k,v in patient.items()}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/latent_leakagefree_results.json", "w"), indent=2)
    print("\nWROTE results/latent_leakagefree_results.json")

if __name__ == "__main__":
    main()
