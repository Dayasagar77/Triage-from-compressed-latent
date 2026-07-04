"""
bach_stain_macenko.py — reviewer W4: does cheap Macenko stain normalization recover any of
the BACH zero-shot gap? Applies standard Macenko normalization to each BACH image (to a
canonical H&E target), then runs the SAME zero-shot BreakHis-trained classifier
(eval_external_bach.py protocol) and reports benign/malignant metrics with vs without
normalization on the same 400 images.

Run: python bach_stain_macenko.py \
      --bach "path/to/BACH/ICIAR2018_BACH_Challenge/ICIAR2018_BACH_Challenge/Photos" \
      --model models/final_cancer_model.pth
Output: results/bach_stain_macenko.json  (send me this)
"""
import argparse, os, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ai_engine import CancerNet, SUBTYPES, BENIGN_SUBTYPES
from sklearn.metrics import roc_auc_score

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
# canonical target stain matrix + max concentrations (Macenko et al., common defaults)
TARGET_HE=np.array([[0.5626,0.2159],[0.7201,0.8012],[0.4062,0.5581]])
MAXC_REF=np.array([1.9705,1.0308])

def macenko(img, Io=240, beta=0.15, alpha=1):
    """Return a Macenko-normalized RGB uint8 image; fall back to input on failure."""
    I=np.asarray(img.convert("RGB")).astype(float).reshape(-1,3)
    try:
        OD=-np.log((I+1)/Io); OD=np.clip(OD,0,None)
        ODhat=OD[np.all(OD>=beta,axis=1)]
        if len(ODhat)<50: return img
        _,V=np.linalg.eigh(np.cov(ODhat.T)); V=V[:,[2,1]]
        if V[0,0]<0: V[:,0]*=-1
        if V[0,1]<0: V[:,1]*=-1
        T=ODhat@V; phi=np.arctan2(T[:,1],T[:,0])
        mn,mx=np.percentile(phi,alpha),np.percentile(phi,100-alpha)
        v1=V@np.array([np.cos(mn),np.sin(mn)]); v2=V@np.array([np.cos(mx),np.sin(mx)])
        HE=np.array([v1,v2]).T if v1[0]>v2[0] else np.array([v2,v1]).T
        C=np.linalg.lstsq(HE,OD.T,rcond=None)[0]
        maxC=np.percentile(C,99,axis=1); maxC[maxC<1e-6]=1e-6
        C2=C*(MAXC_REF/maxC)[:,None]
        ODn=TARGET_HE@C2; In=Io*np.exp(-ODn); In=np.clip(In,0,255)
        return Image.fromarray(In.T.reshape(img.size[1],img.size[0],3).astype(np.uint8))
    except Exception:
        return img

def bach_binary(name):
    n=name.strip().lower().replace("-"," ").replace("_"," ")
    if n in ("normal","n","benign","b"): return 0
    if n in ("in situ","insitu","is","invasive","iv"): return 1
    return None

def evaluate(net,mal,tf,items,normalize):
    yt,ys=[],[]
    for path,lab in items:
        im=Image.open(path).convert("RGB")
        if normalize: im=macenko(im)
        x=tf(im).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): p=F.softmax(net(x),1)[0]
        yt.append(lab); ys.append(float(sum(p[i] for i in mal)))
    yt=np.array(yt); ys=np.array(ys); yp=(ys>=0.5).astype(int)
    tp=int(((yp==1)&(yt==1)).sum()); tn=int(((yp==0)&(yt==0)).sum())
    fp=int(((yp==1)&(yt==0)).sum()); fn=int(((yp==0)&(yt==1)).sum())
    sens=tp/max(tp+fn,1); spec=tn/max(tn+fp,1)
    return {"accuracy":round((tp+tn)/len(yt),4),"sensitivity":round(sens,4),
            "specificity":round(spec,4),"balanced_accuracy":round(0.5*(sens+spec),4),
            "auc":round(float(roc_auc_score(yt,ys)),4) if len(set(yt))>1 else None,
            "confusion":{"tp":tp,"tn":tn,"fp":fp,"fn":fn}}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bach",required=True); ap.add_argument("--model",default="models/final_cancer_model.pth")
    ap.add_argument("--out",default="results/bach_stain_macenko.json")
    a=ap.parse_args(); os.makedirs("results",exist_ok=True)
    net=CancerNet(num_classes=len(SUBTYPES),pretrained=False).to(DEVICE)
    net.load_state_dict(torch.load(a.model,map_location=DEVICE)); net.eval()
    mal={i for i,n in enumerate(SUBTYPES) if n not in BENIGN_SUBTYPES}
    tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    exts=(".tif",".tiff",".png",".jpg",".jpeg",".bmp"); items=[]
    for root,_,files in os.walk(a.bach):
        lab=bach_binary(os.path.basename(root))
        if lab is None: continue
        for f in files:
            if f.lower().endswith(exts): items.append((os.path.join(root,f),lab))
    if not items: raise SystemExit("No BACH images found under --bach.")
    print(f"{len(items)} images; evaluating raw then Macenko-normalized...")
    raw=evaluate(net,mal,tf,items,False); print("raw:",raw)
    norm=evaluate(net,mal,tf,items,True); print("macenko:",norm)
    out={"dataset":"BACH ICIAR-2018 zero-shot","n":len(items),
         "raw":raw,"macenko_normalized":norm,
         "note":"if macenko balanced_accuracy > raw, part of the shift is color variance"}
    json.dump(out,open(a.out,"w"),indent=2); print("saved",a.out)

if __name__=="__main__": main()
