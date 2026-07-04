"""
diagnose_vs_bytes.py — the paper's thesis as a CURVE: benign/malignant accuracy read from
the compressed latent, as a function of how many bytes were transmitted.

For each trained FactorizedCodec (lambda in {0.002,0.01,0.05}) we extract the quantized latent
and train the same lightweight head under the official patient-wise 5-fold protocol, recording
benign/malignant accuracy. The x-axis (KB) is the codec's measured avg size from
learned_codec_results.json. This shows diagnosis degrading GRACEFULLY as bandwidth shrinks —
something no JPEG pipeline can offer, since JPEG's bitstream is not diagnosable.

    python diagnose_vs_bytes.py --data "path/to/BreakHis_8class" \
        --folds "path/to/archive/Folds.csv" --epochs 20

Output: results/diagnose_vs_bytes.json  and  results/figures/fig_diagnose_vs_bytes.png
Needs models/learned_codec_l{0.002,0.01,0.05}.pth (already trained).
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from learned_codec import FactorizedCodec
from train_utils import folds_from_csv, get_weighted_loss_function

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
TF=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
BENIGN={"Adenosis","Fibroadenoma","Phyllodes Tumor","Tubular Adenoma"}

class LatentHead(nn.Module):
    def __init__(self,in_ch,num_classes=8):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_ch,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),
            nn.Conv2d(256,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(0.3),nn.Linear(128,num_classes))
    def forward(self,x): return self.net(x)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--folds",required=True)
    ap.add_argument("--lambdas",type=float,nargs="+",default=[0.002,0.01,0.05])
    ap.add_argument("--epochs",type=int,default=20); ap.add_argument("--batch",type=int,default=64)
    ap.add_argument("--workers",type=int,default=4,help="data-loading workers (set 0 if Windows errors)")
    a=ap.parse_args(); os.makedirs("results/figures",exist_ok=True)

    ds=datasets.ImageFolder(a.data,transform=TF)
    labels=np.array(ds.targets); classes=ds.classes
    benign_idx={i for i,c in enumerate(classes) if c in BENIGN}
    to_bin=lambda arr:np.array([0 if x in benign_idx else 1 for x in arr])
    paths=[p for p,_ in ds.samples]
    sizes={}
    try: sizes={float(k):v["avg_kb"] for k,v in json.load(open("results/learned_codec_results.json"))["points"].items()}
    except Exception: pass

    curve=[]
    for lm in a.lambdas:
        sd=f"models/learned_codec_l{lm}.pth"
        if not os.path.exists(sd): print("skip (missing)",sd); continue
        codec=FactorizedCodec().to(DEVICE); codec.load_state_dict(torch.load(sd,map_location=DEVICE)); codec.eval()
        for p in codec.parameters(): p.requires_grad_(False)
        feats=[]; done=0; t0=time.time()
        loader=DataLoader(ds,batch_size=64,shuffle=False,num_workers=a.workers)
        print(f"[lambda {lm}] extracting latents from {len(ds)} images (workers={a.workers})...", flush=True)
        with torch.no_grad():
            for x,_ in loader:
                feats.append(codec.encode_latent(x.to(DEVICE)).cpu())
                done+=x.size(0)
                if done % 640 < 64 or done==len(ds):
                    rate=done/max(time.time()-t0,1e-6)
                    eta=(len(ds)-done)/max(rate,1e-6)
                    print(f"    {done}/{len(ds)} images  ({rate:.0f} img/s, ~{eta:.0f}s left)", flush=True)
        feats=torch.cat(feats); in_ch=feats.shape[1]
        print(f"[lambda {lm}] latents ready ({time.time()-t0:.0f}s); training head 5-fold...", flush=True)
        accs=[]
        for fi,(tr,va) in enumerate(folds_from_csv(a.folds,paths,mag=None)):
            torch.manual_seed(0)                     # fair, reproducible init across rates
            head=LatentHead(in_ch,len(classes)).to(DEVICE)
            crit=get_weighted_loss_function(labels[tr],len(classes),DEVICE)
            opt=torch.optim.Adam(head.parameters(),lr=1e-3)
            sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)  # MATCH classify_from_latent.py
            tl=DataLoader(TensorDataset(feats[tr],torch.tensor(labels[tr])),batch_size=a.batch,shuffle=True)
            for _ in range(a.epochs):
                head.train()
                for fx,fy in tl:
                    opt.zero_grad(); crit(head(fx.to(DEVICE)),fy.to(DEVICE)).backward(); opt.step()
                sch.step()
            head.eval()
            with torch.no_grad():
                pr=torch.softmax(head(feats[va].to(DEVICE)),1).cpu().numpy().argmax(1)
            accs.append(accuracy_score(to_bin(labels[va]),to_bin(pr)))
            print(f"    fold {fi+1}/5: benign/malignant acc {accs[-1]*100:.1f}%", flush=True)
        kb=sizes.get(lm)
        curve.append({"lambda":lm,"kb":kb,"binary_acc_mean":round(float(np.mean(accs)),4),
                      "binary_acc_std":round(float(np.std(accs)),4)})
        print(f"lambda {lm}: {kb} KB -> B/M acc {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}")
    json.dump({"curve":curve},open("results/diagnose_vs_bytes.json","w"),indent=2)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        pts=[c for c in curve if c["kb"] is not None]; pts.sort(key=lambda c:c["kb"])
        xs=[c["kb"] for c in pts]; ys=[c["binary_acc_mean"]*100 for c in pts]; es=[c["binary_acc_std"]*100 for c in pts]
        fig,ax=plt.subplots(figsize=(6,4.2))
        ax.errorbar(xs,ys,yerr=es,marker="o",capsize=4,linewidth=2,color="#7040c0")
        ax.set_xlabel("Transmitted size (KB per image)"); ax.set_ylabel("Benign/malignant accuracy from latent (%)")
        ax.set_title("Diagnosis from the compressed latent vs. transmitted bytes"); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig("results/figures/fig_diagnose_vs_bytes.png",dpi=160,bbox_inches="tight")
        print("figure saved: results/figures/fig_diagnose_vs_bytes.png")
    except Exception as e:
        print("plot skipped:",e)
    print("saved results/diagnose_vs_bytes.json")

if __name__=="__main__": main()
