"""
classify_from_latent.py — diagnose DIRECTLY from the learned compressed latent.

The novel angle a learned codec enables (JPEG cannot): the receiver classifies from
the transmitted code without decoding the image. We extract the codec's quantized
latent for every image, then train a small head on it under the official patient-wise
5-fold protocol.

    python classify_from_latent.py --data DATA --folds Folds.csv --codec-lambda 0.01 --epochs 20

Output: results/latent_classification_results.json
Needs models/learned_codec_l{lambda}.pth (run train_compression_v2.py first).
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from learned_codec import FactorizedCodec
from train_utils import folds_from_csv, aggregate_fold_metrics, get_weighted_loss_function

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
BENIGN = {"Adenosis", "Fibroadenoma", "Phyllodes Tumor", "Tubular Adenoma"}


class LatentHead(nn.Module):
    def __init__(self, in_ch, num_classes=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes))
    def forward(self, x): return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--codec-lambda", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--mag", default=None)
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    ds = datasets.ImageFolder(args.data, transform=TF)
    labels = np.array(ds.targets); classes = ds.classes
    benign_idx = {i for i, c in enumerate(classes) if c in BENIGN}
    def to_bin(a): return [0 if x in benign_idx else 1 for x in a]

    codec = FactorizedCodec().to(DEVICE)
    sd = os.path.join("models", f"learned_codec_l{args.codec_lambda}.pth")
    if not os.path.exists(sd):
        raise SystemExit(f"{sd} not found. Run train_compression_v2.py first (with lambda {args.codec_lambda}).")
    codec.load_state_dict(torch.load(sd, map_location=DEVICE)); codec.eval()
    for p in codec.parameters(): p.requires_grad_(False)
    print(f"Loaded {sd}")

    print("Extracting quantized latents (transmitted codes)...")
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    feats = []
    with torch.no_grad():
        for x, _ in loader:
            feats.append(codec.encode_latent(x.to(DEVICE)).cpu())
    feats = torch.cat(feats); in_ch = feats.shape[1]
    print(f"Latent per image: {tuple(feats.shape[1:])}")

    sample_paths = [p for p, _ in ds.samples]
    fold_metrics = []
    for fold, (tr, va) in enumerate(folds_from_csv(args.folds, sample_paths, mag=args.mag)):
        head = LatentHead(in_ch, len(classes)).to(DEVICE)
        crit = get_weighted_loss_function(labels[tr], len(classes), DEVICE)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        tl = DataLoader(TensorDataset(feats[tr], torch.tensor(labels[tr])), batch_size=args.batch, shuffle=True)
        for ep in range(args.epochs):
            head.train()
            for fx, fy in tl:
                opt.zero_grad(); crit(head(fx.to(DEVICE)), fy.to(DEVICE)).backward(); opt.step()
            sch.step()
        head.eval()
        with torch.no_grad():
            probs = torch.softmax(head(feats[va].to(DEVICE)), 1).cpu().numpy()
            preds = probs.argmax(1); gts = labels[va]
        m = {"accuracy": accuracy_score(gts, preds),
             "macro_f1": f1_score(gts, preds, average="macro", zero_division=0),
             "binary_accuracy": accuracy_score(to_bin(gts), to_bin(preds))}
        try: m["auc_ovr"] = float(roc_auc_score(gts, probs, multi_class="ovr", average="macro"))
        except Exception: m["auc_ovr"] = float("nan")
        fold_metrics.append(m)
        print(f"  fold {fold+1}: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} B/M={m['binary_accuracy']:.4f} AUC={m['auc_ovr']:.4f}")

    agg = aggregate_fold_metrics(fold_metrics)
    out = {"codec_lambda": args.codec_lambda, "latent_channels": int(in_ch),
           "per_fold": fold_metrics, "summary": agg, "classes": classes}
    json.dump(out, open("results/latent_classification_results.json", "w"), indent=2)
    print("\n==== CLASSIFY-FROM-LATENT (patient-wise 5-fold) ====")
    print(f"8-class acc : {agg['accuracy_mean']*100:.2f}% +/- {agg['accuracy_std']*100:.2f}")
    print(f"Macro-F1    : {agg['macro_f1_mean']:.4f} +/- {agg['macro_f1_std']:.4f}")
    print(f"Benign/Malig: {agg['binary_accuracy_mean']*100:.2f}% +/- {agg['binary_accuracy_std']*100:.2f}")
    print(f"AUC (ovr)   : {agg['auc_ovr_mean']:.4f}")


if __name__ == "__main__":
    main()
