"""
classify_latent_curve.py — the paper's thesis as a CURVE: benign/malignant accuracy read
DIRECTLY from the compressed latent, as a function of transmitted size. Runs classify-from-
latent at each trained codec rate (lambda in {0.002, 0.01, 0.05}) under the SAME patient-wise
folds, and records (avg_bytes, binary accuracy). No standard codec can produce this curve,
because JPEG's bitstream is not diagnosable without decoding.

    python classify_latent_curve.py --data "path/to/BreakHis_8class" \
        --folds "path/to/archive/Folds.csv" --epochs 20

Needs models/learned_codec_l{0.002,0.01,0.05}.pth (already trained).
Output: results/latent_rate_curve.json  -> send me this and I plot it as a new figure.
"""
import argparse, json, os
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, roc_auc_score
from learned_codec import FactorizedCodec
from train_utils import folds_from_csv, get_weighted_loss_function

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
TF=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
BENIGN={"Adenosis","Fibroadenoma","Phyllodes Tumor","Tubular Adenoma"}

class LatentHead(nn.Module):
    def __init__(self,in_ch,num_classes=8):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(in_ch,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),
            nn.Conv2d(256,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(0.3),nn.Linear(128,num_classes))
    def forward(self,x): return self.net(x)

def rate_bytes(codec, loader):
    tot=0.0; n=0
    with torch.no_grad():
        for x,_ in loader:
            out=codec(x.to(DEVICE)); tot+=float(out["bits"])/8.0; n+=x.shape[0]
    return tot/n

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--folds",required=True)
    ap.add_argument("--lambdas",type=float,nargs="+",default=[0.002,0.01,0.05])
    ap.add_argument("--epochs",type=int,default=20); ap.add_argument("--batch",type=int,default=64)
    a=ap.parse_args(); os.makedirs("results",exist_ok=True)
    ds=datasets.ImageFolder(a.data,transform=TF); labels=np.array(ds.targets); classes=ds.classes
    benign_idx={i for i,c in enumerate(classes) if c in BENIGN}
    to_bin=lambda arr:[0 if x in benign_idx else 1 for x in arr]
    paths=[p for p,_ in ds.samples]
    rate_loader=DataLoader(ds,batch_size=32,shuffle=False,num_workers=0)

    points=[]
    for lm in a.lambdas:
        sd=f"models/learned_codec_l{lm}.pth"
        if not os.path.exists(sd): print(f"skip {sd} (missing)"); continue
        codec=FactorizedCodec().to(DEVICE); codec.load_state_dict(torch.load(sd,map_location=DEVICE)); codec.eval()
        for p in codec.parameters(): p.requires_grad_(False)
        avg_bytes=rate_bytes(codec, rate_loader)
        feats=[]
        with torch.no_grad():
            for x,_ in rate_loader: feats.append(codec.encode_latent(x.to(DEVICE)).cpu())
        feats=torch.cat(feats); in_ch=feats.shape[1]
        accs=[]
        for tr,va in folds_from_csv(a.folds, paths, mag=None):
            head=LatentHead(in_ch,len(classes)).to(DEVICE)
            crit=get_weighted_loss_function(labels[tr],len(classes),DEVICE)
            opt=torch.optim.Adam(head.parameters(),lr=1e-3)
            tl=DataLoader(TensorDataset(feats[tr],torch.tensor(labels[tr])),batch_size=a.batch,shuffle=True)
            for _ in range(a.epochs):
                head.train()
                for fx,fy in tl: opt.zero_grad(); crit(head(fx.to(DEVICE)),fy.to(DEVICE)).backward(); opt.step()
            head.eval()
            with torch.no_grad(): preds=torch.softmax(head(feats[va].to(DEVICE)),1).cpu().numpy().argmax(1)
            accs.append(accuracy_score(to_bin(labels[va]),to_bin(preds)))
        pt={"lambda":lm,"avg_bytes":round(avg_bytes,1),"avg_kb":round(avg_bytes/1024,2),
            "binary_accuracy_mean":round(float(np.mean(accs)),4),"binary_accuracy_std":round(float(np.std(accs)),4)}
        points.append(pt); print(pt)
    json.dump({"note":"benign/malignant accuracy from the compressed latent vs transmitted size",
               "points":points}, open("results/latent_rate_curve.json","w"), indent=2)
    print("saved results/latent_rate_curve.json")

if __name__=="__main__": main()
