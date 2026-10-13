"""
r6_latent_bach.py - Reviewer-4 comment #6.
External validation of the LATENT triage (not just the full-image classifier) on BACH.
Codec (FactorizedCodec l0.01) + latent head are trained on BreakHis only; BACH is unseen
by both, so this is a clean cross-dataset test of the diagnose-from-latent result.
Output: results/latent_bach_results.json
"""
import json, os, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from learned_codec import FactorizedCodec
from train_utils import get_weighted_loss_function

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
BENIGN = {"Adenosis", "Fibroadenoma", "Phyllodes Tumor", "Tubular Adenoma"}
BREAKHIS = r"D:\DAYA PHD\PHD WORK\Journal paper 3\BreakHis_8class"
BACH = r"D:\DAYA PHD\PHD WORK\Journal paper 3\BACH\ICIAR2018_BACH_Challenge\ICIAR2018_BACH_Challenge\Photos"
CODEC = r"models\learned_codec_l0.01.pth"

class LatentHead(nn.Module):
    def __init__(self, in_ch, n=8):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n))
    def forward(self, x): return self.net(x)

def bach_binary(name):
    n = name.strip().lower()
    if n in ("normal", "benign"): return 0
    if n in ("insitu", "invasive"): return 1
    return None

def main():
    codec = FactorizedCodec().to(DEVICE)
    codec.load_state_dict(torch.load(CODEC, map_location=DEVICE)); codec.eval()
    for p in codec.parameters(): p.requires_grad_(False)

    ds = datasets.ImageFolder(BREAKHIS, transform=TF)
    classes = ds.classes; labels = np.array(ds.targets)
    benign_idx = {i for i, c in enumerate(classes) if c in BENIGN}
    mal_cols = [i for i in range(len(classes)) if i not in benign_idx]
    print(f"Encoding {len(ds)} BreakHis latents...", flush=True)
    feats = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=32, shuffle=False, num_workers=0):
            feats.append(codec.encode_latent(x.to(DEVICE)).cpu())
    feats = torch.cat(feats); in_ch = feats.shape[1]

    head = LatentHead(in_ch, len(classes)).to(DEVICE)
    crit = get_weighted_loss_function(labels, len(classes), DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    tl = DataLoader(TensorDataset(feats, torch.tensor(labels)), batch_size=64, shuffle=True)
    for ep in range(20):
        head.train()
        for fx, fy in tl:
            opt.zero_grad(); crit(head(fx.to(DEVICE)), fy.to(DEVICE)).backward(); opt.step()
        sch.step()
    print("Latent head trained on BreakHis. Evaluating on BACH...", flush=True)

    exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
    bach = []
    for root, _, files in os.walk(BACH):
        lab = bach_binary(os.path.basename(root))
        if lab is None: continue
        for f in files:
            if f.lower().endswith(exts): bach.append((os.path.join(root, f), lab))
    print("BACH images:", len(bach), flush=True)
    yb, ys = [], []
    head.eval()
    with torch.no_grad():
        for i in range(0, len(bach), 32):
            b = bach[i:i+32]
            xs = torch.stack([TF(Image.open(p).convert("RGB")) for p, _ in b]).to(DEVICE)
            pr = torch.softmax(head(codec.encode_latent(xs)), 1).cpu().numpy()
            for j, (_, lab) in enumerate(b):
                yb.append(lab); ys.append(float(pr[j, mal_cols].sum()))
    yb = np.array(yb); ys = np.array(ys); yp = (ys >= 0.5).astype(int)
    tp = int(((yp==1)&(yb==1)).sum()); tn = int(((yp==0)&(yb==0)).sum())
    fp = int(((yp==1)&(yb==0)).sum()); fn = int(((yp==0)&(yb==1)).sum())
    acc = (tp+tn)/len(yb); sens = tp/max(tp+fn,1); spec = tn/max(tn+fp,1)
    auc = float(roc_auc_score(yb, ys)) if len(set(yb)) > 1 else None
    out = {"dataset": "BACH ICIAR-2018 (external, LATENT triage, inference only)", "n": len(yb),
           "codec": "FactorizedCodec l0.01", "binary_accuracy": round(acc,4),
           "sensitivity": round(sens,4), "specificity": round(spec,4),
           "auc": round(auc,4) if auc else None, "confusion": {"tp":tp,"tn":tn,"fp":fp,"fn":fn},
           "note": "latent head trained on BreakHis latents only; BACH unseen by codec and head"}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/latent_bach_results.json", "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
