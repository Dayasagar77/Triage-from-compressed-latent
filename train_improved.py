"""
train_improved.py — attempt to lift patient-wise 8-class accuracy above 49.2% honestly.

Same model (ResNet-50 + SE, ImageNet transfer) and same STRICT patient-wise folds, but with:
  * stain/colour augmentation (ColorJitter) — the key missing augmentation for H&E images,
  * RandomResizedCrop for scale robustness,
  * more epochs (default 50) with cosine schedule (already in train_one_fold).
Saves results/kfold_results_improved.json (does NOT overwrite the reported kfold_results.json).
Report whatever it yields — including if it does not beat 49.2% (a legitimate negative result).

    python train_improved.py --data "D:/.../BreakHis_8class" --folds "D:/.../archive/Folds.csv" --epochs 50 --batch 16
"""
import os
os.environ["AI_ENGINE_LITE"]="1"   # skip loading unused app models in workers
import argparse, json, os
import numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from train import train_one_fold, IMG_TF_EVAL
from train_utils import folds_from_csv
from ai_engine import BENIGN_SUBTYPES

STRONG_TF = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # stain variation
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--folds",required=True)
    ap.add_argument("--epochs",type=int,default=50); ap.add_argument("--batch",type=int,default=16)
    ap.add_argument("--lr",type=float,default=1e-4); ap.add_argument("--out",default="results/kfold_results_improved.json")
    a=ap.parse_args(); os.makedirs("results",exist_ok=True)

    base=datasets.ImageFolder(a.data,transform=STRONG_TF)
    ev=datasets.ImageFolder(a.data,transform=IMG_TF_EVAL)
    labels=np.array(base.targets); classes=base.classes; ncls=len(classes)
    paths=[p for p,_ in base.samples]
    ben={i for i,c in enumerate(classes) if c in BENIGN_SUBTYPES}
    tobin=lambda arr:[0 if x in ben else 1 for x in arr]

    accs,f1s,bins=[],[],[]
    for fi,(tr,va) in enumerate(folds_from_csv(a.folds,paths,mag=None)):
        print(f"  [fold {fi+1}/{a.k if hasattr(a,'k') else 5}] training...", flush=True)
        m,(gts,preds,_),_=train_one_fold(base,ev,labels,tr,va,ncls,a.epochs,a.batch,a.lr,"proposed",num_workers=0,verbose=True)
        b=accuracy_score(tobin(gts),tobin(preds))
        accs.append(m["accuracy"]); f1s.append(m["macro_f1"]); bins.append(b)
        print(f"  fold {fi+1}/5: 8-class {m['accuracy']*100:.2f}% | macroF1 {m['macro_f1']:.3f} | B/M {b*100:.2f}%")

    out={"protocol":"patient-wise five-fold, stronger augmentation","epochs":a.epochs,
         "accuracy_mean":round(float(np.mean(accs)),4),"accuracy_std":round(float(np.std(accs)),4),
         "macro_f1_mean":round(float(np.mean(f1s)),4),"macro_f1_std":round(float(np.std(f1s)),4),
         "binary_accuracy_mean":round(float(np.mean(bins)),4)}
    json.dump(out,open(a.out,"w"),indent=2)
    print("\n==== IMPROVED TRAINING (patient-wise) ====")
    print(f"8-class: {out['accuracy_mean']*100:.2f}% ± {out['accuracy_std']*100:.2f}  (baseline reported: 49.24%)")
    print(f"macro-F1: {out['macro_f1_mean']:.3f} | B/M: {out['binary_accuracy_mean']*100:.2f}%")
    print("saved",a.out)

if __name__=="__main__": main()
