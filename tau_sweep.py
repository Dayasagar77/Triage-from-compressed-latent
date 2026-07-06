"""
tau_sweep.py — reviewer Q2: empirically find the promotion threshold τ that balances
malignant sensitivity against fast-path queue load. Trains the latent triage head
(same patient-wise protocol as emit_latent_confusion.py), pools per-sample malignant
probabilities, then sweeps τ: for each τ it computes the promoted fraction, malignant
sensitivity, and specificity, and feeds the operating point into the Cobham
non-preemptive M/D/1 priority model (as in queue_sim.py) to get the missed-malignant
queue delay at a chosen load ρ.

Run: python tau_sweep.py --data "D:/DAYA PHD/PHD WORK/Journal paper 3/BreakHis_8class" \
        --folds "D:/DAYA PHD/PHD WORK/Journal paper 3/archive/Folds.csv" --rho 0.6
Output: results/tau_sweep.json , results/figures/fig_tau_sweep.png  (send me the json)
"""
import argparse, json, os
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from learned_codec import FactorizedCodec
from train_utils import folds_from_csv, get_weighted_loss_function
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
TF=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
BENIGN={"Adenosis","Fibroadenoma","Phyllodes Tumor","Tubular Adenoma"}
D_SVC=49.0*8/500.0   # deterministic service time (s), 49 KB @ 500 kb/s (matches queue_sim.py)

class LatentHead(nn.Module):
    def __init__(self,in_ch,n=8):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(in_ch,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),
            nn.Conv2d(256,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(0.3),nn.Linear(128,n))
    def forward(self,x): return self.net(x)

def cobham_missed_delay(rho, frac_high):
    """mean queue delay (s) of a LOW-priority (missed-malignant) case, M/D/1 non-preempt."""
    lam=rho/D_SVC; Res=lam*D_SVC*D_SVC/2.0
    rho_h=frac_high*rho
    return Res/((1-rho_h)*(1-rho))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--folds",required=True)
    ap.add_argument("--codec-lambda",type=float,default=0.01); ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--rho",type=float,default=0.6,help="link utilisation for the delay column")
    args=ap.parse_args(); os.makedirs("results/figures",exist_ok=True)

    ds=datasets.ImageFolder(args.data,transform=TF); labels=np.array(ds.targets); classes=ds.classes
    benign_idx={i for i,c in enumerate(classes) if c in BENIGN}
    mal_cols=[i for i in range(len(classes)) if i not in benign_idx]
    to_bin=lambda a:np.array([0 if x in benign_idx else 1 for x in a])
    codec=FactorizedCodec().to(DEVICE)
    codec.load_state_dict(torch.load(os.path.join("models",f"learned_codec_l{args.codec_lambda}.pth"),map_location=DEVICE))
    codec.eval()
    for p in codec.parameters(): p.requires_grad_(False)
    feats=[]
    with torch.no_grad():
        for x,_ in DataLoader(ds,batch_size=32,shuffle=False,num_workers=0):
            feats.append(codec.encode_latent(x.to(DEVICE)).cpu())
    feats=torch.cat(feats); in_ch=feats.shape[1]
    paths=[p for p,_ in ds.samples]
    y_true=[]; p_mal=[]
    for fold,(tr,va) in enumerate(folds_from_csv(args.folds,paths)):
        head=LatentHead(in_ch,len(classes)).to(DEVICE)
        crit=get_weighted_loss_function(labels[tr],len(classes),DEVICE)
        opt=torch.optim.Adam(head.parameters(),lr=1e-3)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
        tl=DataLoader(TensorDataset(feats[tr],torch.tensor(labels[tr])),batch_size=64,shuffle=True)
        for ep in range(args.epochs):
            head.train()
            for fx,fy in tl:
                opt.zero_grad(); crit(head(fx.to(DEVICE)),fy.to(DEVICE)).backward(); opt.step()
            sch.step()
        head.eval()
        with torch.no_grad():
            pr=torch.softmax(head(feats[va].to(DEVICE)),1).cpu().numpy()
        y_true.append(to_bin(labels[va])); p_mal.append(pr[:,mal_cols].sum(1))
        print(f"  fold {fold+1} done")
    y=np.concatenate(y_true); sc=np.concatenate(p_mal)
    npos=max((y==1).sum(),1); nneg=max((y==0).sum(),1)
    rows=[]
    for tau in [0.5,0.6,0.7,0.8,0.9,0.95]:
        prom=sc>=tau
        sens=((prom)&(y==1)).sum()/npos                # captured malignant
        spec=((~prom)&(y==0)).sum()/nneg
        frac_high=prom.mean()
        d_missed=cobham_missed_delay(args.rho,frac_high)
        rows.append(dict(tau=tau,promoted_fraction=round(float(frac_high),4),
                         sensitivity=round(float(sens),4),specificity=round(float(spec),4),
                         missed_malignant_delay_s=round(float(d_missed),4)))
        print(f"tau={tau}: promoted={frac_high:.3f} sens={sens:.3f} spec={spec:.3f} missed_delay={d_missed:.3f}s")
    # simple suggestion: highest tau keeping sensitivity >= 0.85
    ok=[r for r in rows if r["sensitivity"]>=0.85]
    suggest=max(ok,key=lambda r:r["tau"])["tau"] if ok else 0.5
    out=dict(rho=args.rho,service_s=round(D_SVC,4),rows=rows,
             suggested_tau=suggest,
             note="promoted_fraction feeds Cobham M/D/1; higher tau lowers fast-path load but reduces captured sensitivity")
    json.dump(out,open("results/tau_sweep.json","w"),indent=2)
    fig,ax1=plt.subplots(figsize=(6.2,4.2)); ax2=ax1.twinx()
    t=[r["tau"] for r in rows]
    l1,=ax1.plot(t,[r["sensitivity"] for r in rows],"o-",color="tab:red",label="malignant sensitivity (left)")
    l2,=ax1.plot(t,[r["promoted_fraction"] for r in rows],"s--",color="tab:orange",label="promoted fraction (left)")
    l3,=ax2.plot(t,[r["missed_malignant_delay_s"] for r in rows],"^-",color="tab:blue",label="missed-malignant delay, s (right)")
    ax1.set_xlabel("promotion threshold τ"); ax1.set_ylabel("sensitivity / promoted fraction"); ax2.set_ylabel("missed-malignant queue delay (s)")
    ax1.set_title(f"τ trade-off at ρ = {args.rho}"); ax1.legend(handles=[l1,l2,l3],loc="lower left",fontsize=8.5,framealpha=0.92)
    plt.tight_layout(); plt.savefig("results/figures/fig_tau_sweep.png",dpi=140)
    print(json.dumps(out,indent=2)); print("saved results/tau_sweep.json + fig_tau_sweep.png")

if __name__=="__main__": main()
