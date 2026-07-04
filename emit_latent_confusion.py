"""
emit_latent_confusion.py — MEASURE the latent-triage binary confusion matrix, giving the
malignant sensitivity r that Table 8 currently reports only as a sweep.

Same setup as classify_from_latent.py (patient-wise folds, quantized latent -> lightweight
head), but it additionally pools the benign/malignant predictions across all validation folds
and writes sensitivity/specificity + the confusion counts, then prints the realistic
promote-only speed-up r * (oracle 31.8%) using the measured r.

    python emit_latent_confusion.py --data "path/to/BreakHis_8class" \
        --folds "path/to/archive/Folds.csv" --codec-lambda 0.01 --epochs 20

Output: results/latent_confusion.json   (send me this file)
Needs models/learned_codec_l0.01.pth (already trained).
"""
import argparse, json, os
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from learned_codec import FactorizedCodec
from train_utils import folds_from_csv, get_weighted_loss_function

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
    ap.add_argument("--data", required=True); ap.add_argument("--folds", required=True)
    ap.add_argument("--codec-lambda", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--mag", default=None)
    args = ap.parse_args(); os.makedirs("results", exist_ok=True)

    ds = datasets.ImageFolder(args.data, transform=TF)
    labels = np.array(ds.targets); classes = ds.classes
    benign_idx = {i for i, c in enumerate(classes) if c in BENIGN}
    to_bin = lambda a: np.array([0 if x in benign_idx else 1 for x in a])

    codec = FactorizedCodec().to(DEVICE)
    sd = os.path.join("models", f"learned_codec_l{args.codec_lambda}.pth")
    if not os.path.exists(sd): raise SystemExit(f"{sd} not found.")
    codec.load_state_dict(torch.load(sd, map_location=DEVICE)); codec.eval()
    for p in codec.parameters(): p.requires_grad_(False)

    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    feats = []
    with torch.no_grad():
        for x, _ in loader: feats.append(codec.encode_latent(x.to(DEVICE)).cpu())
    feats = torch.cat(feats); in_ch = feats.shape[1]

    sample_paths = [p for p, _ in ds.samples]
    all_true_bin, all_pred_bin = [], []
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
            preds = torch.softmax(head(feats[va].to(DEVICE)), 1).cpu().numpy().argmax(1)
        tb, pb = to_bin(labels[va]), to_bin(preds)
        all_true_bin.append(tb); all_pred_bin.append(pb)
        print(f"  fold {fold+1}: benign/malignant acc={(tb==pb).mean():.4f}")

    yt = np.concatenate(all_true_bin); yp = np.concatenate(all_pred_bin)
    tp=int(((yp==1)&(yt==1)).sum()); tn=int(((yp==0)&(yt==0)).sum())
    fp=int(((yp==1)&(yt==0)).sum()); fn=int(((yp==0)&(yt==1)).sum())
    sens=tp/max(tp+fn,1); spec=tn/max(tn+fp,1)
    # realistic promote-only speed-up using measured r
    try:
        tr_=json.load(open("results/triage_results.json"))
        tf_=tr_["malignant_delivery_time_fifo_s"]["mean"]; tp_=tr_["malignant_delivery_time_priority_s"]["mean"]
        oracle=(tf_-tp_)/tf_; realistic=round(100*sens*oracle,1)
    except Exception:
        realistic=None
    out={"codec_lambda":args.codec_lambda,"confusion":{"tp":tp,"tn":tn,"fp":fp,"fn":fn},
         "malignant_sensitivity_r":round(sens,4),"specificity":round(spec,4),
         "realistic_promote_only_speedup_pct":realistic,
         "note":"r = measured malignant sensitivity of latent triage; realistic speedup = r * oracle(31.8%)"}
    json.dump(out, open("results/latent_confusion.json","w"), indent=2)
    print("\n==== LATENT-TRIAGE CONFUSION (pooled over folds) ====")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"malignant sensitivity r = {sens:.4f} | specificity = {spec:.4f}")
    if realistic is not None: print(f"realistic promote-only speed-up = r*31.8% = {realistic}%")
    print("saved results/latent_confusion.json")

if __name__ == "__main__": main()
