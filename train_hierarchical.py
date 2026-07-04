"""
train_hierarchical.py — two-stage classifier to (try to) lift 8-class accuracy.
Stage 1: benign vs malignant.  Stage 2: subtype WITHIN the predicted group (4-way each).
Evaluated under the SAME official patient-wise folds as the flat model, so numbers are
comparable to Table 2/3. Reports 8-class accuracy + macro-F1 for the hierarchical head and,
for reference, re-computes the flat model's numbers if kfold_results.json exists.

This wraps your existing training utilities; it does not change any reported flat-model
number. Run:
    python train_hierarchical.py --data "path/to/BreakHis_8class" \
        --folds "path/to/archive/Folds.csv" --epochs 25 --batch 16

NOTE: this is a research script to try an idea; report the result honestly whether or not it
beats the flat 49.2%. If it does not help, that is a legitimate negative result worth a sentence.
"""
import argparse, json, os, numpy as np, torch, torch.nn as nn
from collections import defaultdict
# Reuse the project's data/model plumbing:
from ai_engine import CancerNet, SUBTYPES, BENIGN_SUBTYPES
from train_utils import folds_from_csv                      # canonical location
from train import IMG_TF_TRAIN, IMG_TF_EVAL                   # image transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
MAL=[i for i,n in enumerate(SUBTYPES) if n not in BENIGN_SUBTYPES]
BEN=[i for i,n in enumerate(SUBTYPES) if n in BENIGN_SUBTYPES]

def remap(labels, group):  # map global subtype idx -> within-group idx
    idx={g:k for k,g in enumerate(group)}
    return np.array([idx[l] for l in labels])

def train_head(model, loader, epochs, lr, num_classes, weights=None):
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    crit=nn.CrossEntropyLoss(weight=weights.to(DEVICE) if weights is not None else None)
    model.train()
    for ep in range(epochs):
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad(); loss=crit(model(x),y); loss.backward(); opt.step()
    return model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--folds",required=True)
    ap.add_argument("--epochs",type=int,default=25); ap.add_argument("--batch",type=int,default=16)
    ap.add_argument("--lr",type=float,default=1e-4); ap.add_argument("--out",default="results/hierarchical_results.json")
    a=ap.parse_args()
    base=datasets.ImageFolder(a.data,transform=IMG_TF_TRAIN)
    evalds=datasets.ImageFolder(a.data,transform=IMG_TF_EVAL)
    paths=[p for p,_ in base.samples]; labels=np.array([l for _,l in base.samples])
    accs,f1s=[],[]
    from sklearn.metrics import accuracy_score, f1_score
    for fold,(tr,va) in enumerate(folds_from_csv(a.folds, paths, mag=None)):
        # stage 1: binary
        bin_lab=np.array([0 if l in BEN else 1 for l in labels])
        s1=CancerNet(num_classes=2,pretrained=True).to(DEVICE)
        dl=DataLoader(Subset(_relabel(base,bin_lab),list(tr)),batch_size=a.batch,shuffle=True)
        train_head(s1,dl,a.epochs,a.lr,2)
        # stage 2a: benign subtypes, 2b: malignant subtypes
        heads={}
        for name,grp in (("ben",BEN),("mal",MAL)):
            mask=[i for i in tr if labels[i] in grp]
            sub=CancerNet(num_classes=len(grp),pretrained=True).to(DEVICE)
            dl=DataLoader(Subset(_relabel(base,_grp_labels(labels,grp)),mask),batch_size=a.batch,shuffle=True)
            train_head(sub,dl,a.epochs,a.lr,len(grp)); heads[name]=sub
        # evaluate hierarchical on val
        yt,yp=[],[]
        s1.eval(); heads["ben"].eval(); heads["mal"].eval()
        with torch.no_grad():
            for i in va:
                x,_=evalds[i]; x=x.unsqueeze(0).to(DEVICE)
                g=int(s1(x).argmax(1)); grp=BEN if g==0 else MAL; head=heads["ben"] if g==0 else heads["mal"]
                sub=int(head(x).argmax(1)); pred=grp[sub]
                yt.append(int(labels[i])); yp.append(int(pred))
        accs.append(accuracy_score(yt,yp)); f1s.append(f1_score(yt,yp,average="macro"))
        print(f"fold {fold+1}: 8-class acc={accs[-1]:.4f} macroF1={f1s[-1]:.4f}")
    out={"model":"hierarchical (binary -> subtype)","folds":len(accs),
         "accuracy_mean":round(float(np.mean(accs)),4),"accuracy_std":round(float(np.std(accs)),4),
         "macro_f1_mean":round(float(np.mean(f1s)),4),"macro_f1_std":round(float(np.std(f1s)),4)}
    os.makedirs("results",exist_ok=True); json.dump(out,open(a.out,"w"),indent=2)
    print(json.dumps(out,indent=2)); print("saved",a.out)

def _relabel(ds, new_labels):
    ds2=datasets.ImageFolder(ds.root,transform=ds.transform)
    ds2.samples=[(p,int(new_labels[i])) for i,(p,_) in enumerate(ds2.samples)]
    ds2.targets=[s[1] for s in ds2.samples]; return ds2
def _grp_labels(labels, grp):
    idx={g:k for k,g in enumerate(grp)}
    return np.array([idx.get(l,0) for l in labels])

if __name__=="__main__": main()
