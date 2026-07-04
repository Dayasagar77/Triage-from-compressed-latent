"""
imagewise_ablation.py — quantify the DATA-LEAKAGE effect the paper claims.

Runs the SAME CancerNet, same epochs/settings, under IMAGE-WISE (generic stratified k-fold,
which lets a patient's images appear in both train and test = leakage) and compares to the
PATIENT-WISE result already in results/kfold_results.json. The gap is the honest measure of
how much patient leakage inflates BreakHis accuracy — the justification for our strict protocol.

Writes results/imagewise_results.json (does NOT touch the patient-wise file).

    python imagewise_ablation.py --data "path/to/BreakHis_8class" --epochs 25 --batch 16 --k 5
"""
import os
os.environ["AI_ENGINE_LITE"]="1"   # skip loading unused app models in workers
import argparse, json, os
import numpy as np
from torchvision import datasets
from sklearn.metrics import accuracy_score
from train import train_one_fold, IMG_TF_TRAIN, IMG_TF_EVAL
from train_utils import stratified_kfold_indices
from ai_engine import BENIGN_SUBTYPES

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--epochs",type=int,default=25)
    ap.add_argument("--batch",type=int,default=16); ap.add_argument("--k",type=int,default=5)
    ap.add_argument("--lr",type=float,default=1e-4); ap.add_argument("--out",default="results/imagewise_results.json")
    a=ap.parse_args(); os.makedirs("results",exist_ok=True)

    base=datasets.ImageFolder(a.data,transform=IMG_TF_TRAIN)
    ev=datasets.ImageFolder(a.data,transform=IMG_TF_EVAL)
    labels=np.array(base.targets); classes=base.classes; ncls=len(classes)
    ben={i for i,c in enumerate(classes) if c in BENIGN_SUBTYPES}
    tobin=lambda arr:[0 if x in ben else 1 for x in arr]

    accs,f1s,bins=[],[],[]
    for fi,(tr,va) in enumerate(stratified_kfold_indices(labels,k=a.k)):
        print(f"  [fold {fi+1}/{a.k if hasattr(a,'k') else 5}] training...", flush=True)
        m,(gts,preds,_),_=train_one_fold(base,ev,labels,tr,va,ncls,a.epochs,a.batch,a.lr,"proposed",num_workers=4,verbose=True)
        b=accuracy_score(tobin(gts),tobin(preds))
        accs.append(m["accuracy"]); f1s.append(m["macro_f1"]); bins.append(b)
        print(f"  image-wise fold {fi+1}/{a.k}: 8-class {m['accuracy']*100:.2f}% | macroF1 {m['macro_f1']:.3f} | B/M {b*100:.2f}%")

    out={"protocol":"image-wise (generic stratified k-fold; leaky)","k":a.k,
         "accuracy_mean":round(float(np.mean(accs)),4),"accuracy_std":round(float(np.std(accs)),4),
         "macro_f1_mean":round(float(np.mean(f1s)),4),"binary_accuracy_mean":round(float(np.mean(bins)),4)}
    json.dump(out,open(a.out,"w"),indent=2)
    # compare to patient-wise
    pw=None
    try: pw=json.load(open("results/kfold_results.json"))["summary"]["accuracy_mean"]
    except Exception: pass
    print("\n==== LEAKAGE ABLATION ====")
    print(f"image-wise 8-class:  {out['accuracy_mean']*100:.2f}% ± {out['accuracy_std']*100:.2f}")
    if pw is not None:
        print(f"patient-wise 8-class: {pw*100:.2f}%  (results/kfold_results.json)")
        print(f"leakage inflation:   +{(out['accuracy_mean']-pw)*100:.1f} points")
    print("saved",a.out)

if __name__=="__main__": main()
