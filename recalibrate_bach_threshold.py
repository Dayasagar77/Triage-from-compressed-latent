"""
recalibrate_bach_threshold.py — reviewer minor 3.
Re-runs the ZERO-SHOT BACH inference (no fitting) exactly like eval_external_bach.py, but
saves the per-image malignancy score and finds the decision threshold that maximizes
BALANCED ACCURACY (the Youden-optimal operating point). Reports the recalibrated threshold
and the balanced accuracy it achieves, versus the default 0.5 threshold. This shows how
much of the domain-shift gap a cheap per-site threshold shift can recover.

Run:
    python recalibrate_bach_threshold.py --bach "D:/path/to/BACH/Photos" \
        --model models/final_cancer_model.pth
Output: results/bach_threshold_recal.json   (send me this file)
"""
import argparse, os, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ai_engine import CancerNet, SUBTYPES, BENIGN_SUBTYPES
from sklearn.metrics import roc_auc_score

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
def bach_binary(name):
    n=name.strip().lower().replace("-"," ").replace("_"," ")
    if n in ("normal","n","benign","b"): return 0
    if n in ("in situ","insitu","is","in situ carcinoma","invasive","iv","invasive carcinoma"): return 1
    return None

def bal_acc(tb,pred):
    tp=((pred==1)&(tb==1)).sum(); tn=((pred==0)&(tb==0)).sum()
    fp=((pred==1)&(tb==0)).sum(); fn=((pred==0)&(tb==1)).sum()
    sens=tp/max(tp+fn,1); spec=tn/max(tn+fp,1)
    return 0.5*(sens+spec), sens, spec

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bach",required=True); ap.add_argument("--model",default="models/final_cancer_model.pth")
    ap.add_argument("--out",default="results/bach_threshold_recal.json")
    a=ap.parse_args()
    net=CancerNet(num_classes=len(SUBTYPES),pretrained=False).to(DEVICE)
    net.load_state_dict(torch.load(a.model,map_location=DEVICE)); net.eval()
    mal={i for i,n in enumerate(SUBTYPES) if n not in BENIGN_SUBTYPES}
    tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    yt,ys=[],[]; exts=(".tif",".tiff",".png",".jpg",".jpeg",".bmp")
    for root,_,files in os.walk(a.bach):
        lab=bach_binary(os.path.basename(root))
        if lab is None: continue
        for f in files:
            if not f.lower().endswith(exts): continue
            x=tf(Image.open(os.path.join(root,f)).convert("RGB")).unsqueeze(0).to(DEVICE)
            with torch.no_grad(): p=F.softmax(net(x),1)[0]
            yt.append(lab); ys.append(float(sum(p[i] for i in mal)))
    if not yt:
        print("No BACH images matched. Folders seen under --bach (looking for Normal/Benign/InSitu/Invasive):")
        seen=set()
        for root,dirs,files in os.walk(a.bach):
            b=os.path.basename(root)
            n_imgs=sum(1 for f in files if f.lower().endswith(exts))
            if b and (dirs or n_imgs): 
                print(f"   {b}/   (subdirs={len(dirs)}, images={n_imgs})")
        raise SystemExit("Fix the --bach path to point at the folder whose subfolders are the 4 classes.")
    tb=np.array(yt); sc=np.array(ys)
    # default 0.5
    ba0,se0,sp0=bal_acc(tb,(sc>=0.5).astype(int))
    # sweep thresholds -> Youden-optimal balanced accuracy
    best=(-1,0.5,0,0)
    for thr in np.unique(np.round(sc,4)):
        ba,se,sp=bal_acc(tb,(sc>=thr).astype(int))
        if ba>best[0]: best=(ba,float(thr),float(se),float(sp))
    auc=float(roc_auc_score(tb,sc)) if len(set(tb))>1 else None
    out={"dataset":"BACH ICIAR-2018 (zero-shot, threshold recalibration)","n":len(tb),
         "auc":round(auc,4) if auc else None,
         "default_threshold_0.5":{"balanced_accuracy":round(float(ba0),4),
             "sensitivity":round(float(se0),4),"specificity":round(float(sp0),4)},
         "recalibrated":{"threshold":round(best[1],4),"balanced_accuracy":round(best[0],4),
             "sensitivity":round(best[2],4),"specificity":round(best[3],4)}}
    os.makedirs("results",exist_ok=True); json.dump(out,open(a.out,"w"),indent=2)
    print(json.dumps(out,indent=2)); print("saved",a.out)

if __name__=="__main__": main()
