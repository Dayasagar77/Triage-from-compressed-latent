"""
eval_external_bach.py — EXTERNAL validation of the BreakHis-trained classifier on BACH
(ICIAR-2018). This is the single most important experiment for a Q1 submission: it tests
whether the model generalizes to a different dataset/scanner/stain without any retraining.

BACH microscopy (Part A) has 4 classes: Normal, Benign, InSitu, Invasive. We map to the
paper's BINARY task:  Normal + Benign -> benign ;  InSitu + Invasive -> malignant.
We load your trained model (models/final_cancer_model.pth) and run inference ONLY — no
fitting on BACH — then report benign/malignant accuracy, sensitivity, specificity, AUC.

Download BACH (free, registration): https://iciar2018-challenge.grand-challenge.org/Dataset/
Put the Photos folder so that images live under:  <bach_root>/{Normal,Benign,InSitu,Invasive}/*.tif

Run:
    python eval_external_bach.py --bach "D:/path/to/BACH/Photos" --model models/final_cancer_model.pth
"""
import argparse, os, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ai_engine import CancerNet, SUBTYPES, BENIGN_SUBTYPES   # your model + label maps
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def bach_binary(name):
    """Map a BACH class folder name to 0=benign(normal/benign) or 1=malignant(insitu/invasive).
    Robust to spaces, case, and common code variants; returns None if not a class folder."""
    n = name.strip().lower().replace("-", " ").replace("_", " ")
    if n in ("normal", "n"): return 0
    if n in ("benign", "b"): return 0
    if n in ("in situ", "insitu", "is", "in situ carcinoma"): return 1
    if n in ("invasive", "iv", "invasive carcinoma"): return 1
    return None

def malignant_index_set():
    # indices in SUBTYPES that are malignant (not in BENIGN_SUBTYPES)
    return {i for i,name in enumerate(SUBTYPES) if name not in BENIGN_SUBTYPES}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bach",required=True,help="BACH Photos root with class subfolders")
    ap.add_argument("--model",default="models/final_cancer_model.pth")
    ap.add_argument("--out",default="results/external_bach_results.json")
    a=ap.parse_args()

    net=CancerNet(num_classes=len(SUBTYPES),pretrained=False).to(DEVICE)
    net.load_state_dict(torch.load(a.model,map_location=DEVICE)); net.eval()
    mal=malignant_index_set()
    tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

    y_true,y_score,y_pred=[],[],[]
    exts=(".tif",".tiff",".png",".jpg",".jpeg",".bmp")
    for root,_,files in os.walk(a.bach):
        lab=bach_binary(os.path.basename(root))
        if lab is None: continue
        for f in files:
            if not f.lower().endswith(exts): continue
            x=tf(Image.open(os.path.join(root,f)).convert("RGB")).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                p=F.softmax(net(x),dim=1)[0]
            p_mal=float(sum(p[i] for i in mal))
            y_true.append(lab); y_score.append(p_mal); y_pred.append(int(p_mal>=0.5))
    if not y_true:
        raise SystemExit("No BACH class images found. Expected folders named "
                         "Normal/Benign/InSitu(or 'in situ')/Invasive somewhere under --bach.")
    from collections import Counter
    print("images found by binary label (0=benign,1=malignant):", dict(Counter(y_true)))
    y_true=np.array(y_true); y_pred=np.array(y_pred); y_score=np.array(y_score)
    tp=int(((y_pred==1)&(y_true==1)).sum()); tn=int(((y_pred==0)&(y_true==0)).sum())
    fp=int(((y_pred==1)&(y_true==0)).sum()); fn=int(((y_pred==0)&(y_true==1)).sum())
    acc=(tp+tn)/len(y_true); sens=tp/max(tp+fn,1); spec=tn/max(tn+fp,1)
    auc=float(roc_auc_score(y_true,y_score)) if len(set(y_true))>1 else None
    out={"dataset":"BACH ICIAR-2018 (external, inference only)","n":len(y_true),
         "binary_accuracy":round(acc,4),"sensitivity":round(sens,4),"specificity":round(spec,4),
         "auc":round(auc,4) if auc else None,"confusion":{"tp":tp,"tn":tn,"fp":fp,"fn":fn}}
    os.makedirs("results",exist_ok=True); json.dump(out,open(a.out,"w"),indent=2)
    print(json.dumps(out,indent=2)); print("saved",a.out)

if __name__=="__main__": main()
