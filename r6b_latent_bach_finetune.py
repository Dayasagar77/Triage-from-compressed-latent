"""
r6b_latent_bach_finetune.py - Reviewer-4 comment #6, follow-up to r6_latent_bach.py.

r6 showed that LATENT triage does not transfer zero-shot to BACH (AUC 0.58). This script asks the
same question the paper already asks of the full-image classifier (finetune_bach.py, Table 16):
does a SMALL amount of target-site data close the gap for the latent head?

Protocol - identical to finetune_bach.py wherever it applies:
  * BACH split: the same gather() order, random.Random(0) per-class shuffle, 50 benign + 50 malignant
    fine-tune images, the remaining 300 as the held-out test set (the Table 16 split).
  * The codec (FactorizedCodec lambda=0.01) is FIXED; BACH images are only encoded, never used to train it.
  * Zero-shot: a latent head trained on all BreakHis latents exactly as in r6 (20 epochs, Adam 1e-3,
    batch 64, inverse-frequency weighted CE), 8-class, malignant probability = sum of malignant subtypes.
  * Fine-tune: copy that head, replace its final Linear(128, 8) by Linear(128, 2), fine-tune the whole head
    on the 100 BACH latents with Adam 1e-4, batch 16, 15 epochs, plain CE (the finetune_bach.py settings).
  * Both evaluated on the SAME 300-image test set, threshold 0.5.
Seeds are fixed (torch 0, random 0) so the run is repeatable.

    python r6b_latent_bach_finetune.py --breakhis "<BreakHis_8class>" --bach "<BACH Photos folder>"
Output: results/latent_bach_finetune_results.json
"""
import argparse, copy, json, os, random
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from learned_codec import FactorizedCodec
from train_utils import get_weighted_loss_function

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
BENIGN = {"Adenosis", "Fibroadenoma", "Phyllodes Tumor", "Tubular Adenoma"}


class LatentHead(nn.Module):
    def __init__(self, in_ch, n=8):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                 nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n))

    def forward(self, x): return self.net(x)


def bach_binary(name):                                  # as finetune_bach.py
    n = name.strip().lower().replace("-", " ").replace("_", " ")
    if n in ("normal", "n", "benign", "b"): return 0
    if n in ("in situ", "insitu", "is", "invasive", "iv"): return 1
    return None


def gather(bach):                                       # as finetune_bach.py
    exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"); items = []
    for root, _, files in os.walk(bach):
        lab = bach_binary(os.path.basename(root))
        if lab is None: continue
        for f in files:
            if f.lower().endswith(exts): items.append((os.path.join(root, f), lab))
    return items


def metrics(y, s, thr=0.5):
    y = np.asarray(y); s = np.asarray(s); p = (s >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    return {"accuracy": round((tp + tn) / len(y), 4), "sensitivity": round(tp / max(tp + fn, 1), 4),
            "specificity": round(tn / max(tn + fp, 1), 4), "balanced_accuracy": round((tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2, 4),
            "auc": round(float(roc_auc_score(y, s)), 4), "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}


@torch.no_grad()
def encode_paths(codec, paths, bs=16):
    out = []
    for i in range(0, len(paths), bs):
        xb = torch.stack([TF(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(DEVICE)
        out.append(codec.encode_latent(xb).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--breakhis", required=True); ap.add_argument("--bach", required=True)
    ap.add_argument("--codec", default=os.path.join("models", "learned_codec_l0.01.pth"))
    ap.add_argument("--train-per-class", type=int, default=50); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join("results", "latent_bach_finetune_results.json"))
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    codec = FactorizedCodec().to(DEVICE)
    codec.load_state_dict(torch.load(a.codec, map_location=DEVICE, weights_only=True)); codec.eval()
    for p in codec.parameters(): p.requires_grad_(False)

    # 1) latent head on ALL BreakHis latents (as r6_latent_bach.py)
    ds = datasets.ImageFolder(a.breakhis, transform=TF); classes = ds.classes; labels = np.array(ds.targets)
    mal_cols = [i for i, c in enumerate(classes) if c not in BENIGN]
    feats = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=32, shuffle=False, num_workers=0):
            feats.append(codec.encode_latent(x.to(DEVICE)).cpu())
    feats = torch.cat(feats); in_ch = feats.shape[1]
    head = LatentHead(in_ch, len(classes)).to(DEVICE)
    crit = get_weighted_loss_function(labels, len(classes), DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    g = torch.Generator().manual_seed(0)
    tl = DataLoader(TensorDataset(feats, torch.tensor(labels)), batch_size=64, shuffle=True, generator=g)
    for ep in range(20):
        head.train()
        for fx, fy in tl:
            opt.zero_grad(); crit(head(fx.to(DEVICE)), fy.to(DEVICE)).backward(); opt.step()
        sch.step()
    head.eval()

    # 2) BACH split exactly as finetune_bach.py
    items = gather(a.bach); rng = random.Random(a.seed); by = {0: [], 1: []}
    for p, l in items: by[l].append(p)
    for l in by: rng.shuffle(by[l])
    train = [(p, l) for l in (0, 1) for p in by[l][:a.train_per_class]]
    test = [(p, l) for l in (0, 1) for p in by[l][a.train_per_class:]]
    ztr = encode_paths(codec, [p for p, _ in train]); zte = encode_paths(codec, [p for p, _ in test])
    ytr = torch.tensor([l for _, l in train]); yte = [l for _, l in test]

    with torch.no_grad():
        zs = torch.softmax(head(zte.to(DEVICE)), 1)[:, mal_cols].sum(1).cpu().numpy()
    zero = metrics(yte, zs)

    # 3) fine-tune a binary copy of the head on the 100 BACH latents (finetune_bach.py settings)
    h2 = copy.deepcopy(head); h2.net[-1] = nn.Linear(128, 2).to(DEVICE)
    opt2 = torch.optim.Adam(h2.parameters(), lr=a.lr); ce = nn.CrossEntropyLoss(); order = list(range(len(train)))
    for ep in range(a.epochs):
        h2.train(); rng.shuffle(order)
        for i in range(0, len(order), 16):
            idx = order[i:i + 16]
            opt2.zero_grad(); ce(h2(ztr[idx].to(DEVICE)), ytr[idx].to(DEVICE)).backward(); opt2.step()
    h2.eval()
    with torch.no_grad():
        fs = torch.softmax(h2(zte.to(DEVICE)), 1)[:, 1].cpu().numpy()
    fine = metrics(yte, fs)
    out = {"dataset": "BACH ICIAR-2018 photos", "codec": "FactorizedCodec lambda=0.01 (fixed)",
           "n_train": len(train), "n_test": len(test), "train_per_class": a.train_per_class,
           "split": "finetune_bach.py split (random.Random(0), per-class shuffle) - the Table 16 split",
           "zero_shot_latent": zero, "fine_tuned_latent": fine,
           "finetune": {"epochs": a.epochs, "lr": a.lr, "batch": 16, "loss": "cross-entropy"}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2)); print("WROTE", a.out)


if __name__ == "__main__":
    main()
