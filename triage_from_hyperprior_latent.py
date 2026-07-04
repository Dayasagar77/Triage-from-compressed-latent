"""
triage_from_hyperprior_latent.py — reviewer comment C.
Report benign/malignant TRIAGE accuracy read directly from the *hyperprior* codec's
latent (models/hyperprior_l0.01.pth), under the identical patient-wise protocol used for
the v2 FactorizedCodec latent (emit_latent_confusion.py). Lets us state whether the
richer hyperprior latent improves triage over the lightweight v2 latent, or not.

Run (same args as emit_latent_confusion.py):
    python triage_from_hyperprior_latent.py \
        --data "path/to/BreakHis_8class" \
        --folds "path/to/archive/Folds.csv" \
        --codec-lambda 0.01 --epochs 20
Output: results/hyperprior_latent_triage.json   (send me this file)
"""
import argparse, json, os
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from train_hyperprior_pure import ScaleHyperprior
from train_utils import folds_from_csv, get_weighted_loss_function
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TF = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
BENIGN = {"Adenosis","Fibroadenoma","Phyllodes Tumor","Tubular Adenoma"}

class LatentHead(nn.Module):
    def __init__(self, in_ch, num_classes=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128,num_classes))
    def forward(self,x): return self.net(x)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--folds",required=True)
    ap.add_argument("--codec-lambda",type=float,default=0.01)
    ap.add_argument("--epochs",type=int,default=20); ap.add_argument("--batch",type=int,default=64)
    ap.add_argument("--mag",default=None)
    args=ap.parse_args(); os.makedirs("results",exist_ok=True)

    ds=datasets.ImageFolder(args.data,transform=TF)
    labels=np.array(ds.targets); classes=ds.classes
    benign_idx={i for i,c in enumerate(classes) if c in BENIGN}
    mal_idx=[i for i in range(len(classes)) if i not in benign_idx]
    to_bin=lambda a: np.array([0 if x in benign_idx else 1 for x in a])

    codec=ScaleHyperprior().to(DEVICE)
    sd=os.path.join("models",f"hyperprior_l{args.codec_lambda}.pth")
    if not os.path.exists(sd): raise SystemExit(f"{sd} not found. Train hyperprior first.")
    codec.load_state_dict(torch.load(sd,map_location=DEVICE)); codec.eval()
    for p in codec.parameters(): p.requires_grad_(False)

    # encode: quantized main latent y_hat = round(g_a(x))  (same code that is transmitted)
    loader=DataLoader(ds,batch_size=32,shuffle=False,num_workers=0)
    feats=[]
    with torch.no_grad():
        for x,_ in loader:
            y=codec.g_a(x.to(DEVICE)); feats.append(torch.round(y).cpu())
    feats=torch.cat(feats); in_ch=feats.shape[1]
    print(f"hyperprior latent: {tuple(feats.shape[1:])} (M={in_ch})")

    sample_paths=[p for p,_ in ds.samples]
    all_tb, all_pb, all_score=[],[],[]
    for fold,(tr,va) in enumerate(folds_from_csv(args.folds,sample_paths,mag=args.mag)):
        head=LatentHead(in_ch,len(classes)).to(DEVICE)
        crit=get_weighted_loss_function(labels[tr],len(classes),DEVICE)
        opt=torch.optim.Adam(head.parameters(),lr=1e-3)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
        tl=DataLoader(TensorDataset(feats[tr],torch.tensor(labels[tr])),batch_size=args.batch,shuffle=True)
        for ep in range(args.epochs):
            head.train()
            for fx,fy in tl:
                opt.zero_grad(); crit(head(fx.to(DEVICE)),fy.to(DEVICE)).backward(); opt.step()
            sch.step()
        head.eval()
        with torch.no_grad():
            prob=torch.softmax(head(feats[va].to(DEVICE)),1).cpu().numpy()
        preds=prob.argmax(1); pmal=prob[:,mal_idx].sum(1)
        tb,pb=to_bin(labels[va]),to_bin(preds)
        all_tb.append(tb); all_pb.append(pb); all_score.append(pmal)
        print(f"  fold {fold+1}: B/M acc={(tb==pb).mean():.4f}")

    tb=np.concatenate(all_tb); pb=np.concatenate(all_pb); sc=np.concatenate(all_score)
    tp=int(((pb==1)&(tb==1)).sum()); tn=int(((pb==0)&(tb==0)).sum())
    fp=int(((pb==1)&(tb==0)).sum()); fn=int(((pb==0)&(tb==1)).sum())
    acc=(tp+tn)/len(tb); sens=tp/max(tp+fn,1); spec=tn/max(tn+fp,1)
    auc=float(roc_auc_score(tb,sc)) if len(set(tb))>1 else None
    out={"codec":"ScaleHyperprior","codec_lambda":args.codec_lambda,"latent_channels":in_ch,
         "binary_accuracy":round(acc,4),"malignant_sensitivity_r":round(sens,4),
         "specificity":round(spec,4),"binary_auc":round(auc,4) if auc else None,
         "confusion":{"tp":tp,"tn":tn,"fp":fp,"fn":fn},
         "compare_v2_factorized":{"binary_accuracy":0.823,"sensitivity_r":0.8919,"specificity":0.6707}}
    json.dump(out,open("results/hyperprior_latent_triage.json","w"),indent=2)
    print(json.dumps(out,indent=2)); print("saved results/hyperprior_latent_triage.json")

if __name__=="__main__": main()
