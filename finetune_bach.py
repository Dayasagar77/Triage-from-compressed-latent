"""
finetune_bach.py — domain-adaptation test: does a SMALL amount of BACH fine-tuning close the
zero-shot generalization gap found in Section 5.7?

Protocol (fair before/after on the SAME held-out test set):
  1. Split BACH into a small fine-tune set (--train-per-class images per benign/malignant class)
     and a held-out test set (everything else), stratified, fixed seed.
  2. ZERO-SHOT: evaluate the original BreakHis-trained 8-class model on the test set
     (sum malignant-subtype probabilities -> binary). This reproduces Section 5.7 on this split.
  3. FINE-TUNE: replace the final layer with a 2-way head and fine-tune on the small train set.
  4. Evaluate the fine-tuned model on the SAME test set.
Reports benign/malignant accuracy, sensitivity, specificity, AUC — before vs after.

    python finetune_bach.py --bach "D:/.../BACH/ICIAR2018_BACH_Challenge/ICIAR2018_BACH_Challenge/Photos" \
        --model models/final_cancer_model.pth --train-per-class 50 --epochs 15

Output: results/bach_finetune_results.json  (send me this)
Honest note: report whatever it shows. If fine-tuning barely helps, that is itself an important
finding about how large the domain gap is; if it closes most of the gap, that is the good news.
"""
import argparse, os, json, random
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from ai_engine import CancerNet, SUBTYPES, BENIGN_SUBTYPES

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
TF=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
MAL_IDX=[i for i,n in enumerate(SUBTYPES) if n not in BENIGN_SUBTYPES]

def bach_binary(name):
    n=name.strip().lower().replace("-"," ").replace("_"," ")
    if n in ("normal","n","benign","b"): return 0
    if n in ("in situ","insitu","is","invasive","iv"): return 1
    return None

def gather(bach):
    exts=(".tif",".tiff",".png",".jpg",".jpeg",".bmp"); items=[]
    for root,_,files in os.walk(bach):
        lab=bach_binary(os.path.basename(root))
        if lab is None: continue
        for f in files:
            if f.lower().endswith(exts): items.append((os.path.join(root,f),lab))
    return items

def metrics(y,p,s):
    y,p,s=np.array(y),np.array(p),np.array(s)
    tp=int(((p==1)&(y==1)).sum()); tn=int(((p==0)&(y==0)).sum())
    fp=int(((p==1)&(y==0)).sum()); fn=int(((p==0)&(y==1)).sum())
    return {"accuracy":round((tp+tn)/len(y),4),"sensitivity":round(tp/max(tp+fn,1),4),
            "specificity":round(tn/max(tn+fp,1),4),
            "auc":round(float(roc_auc_score(y,s)),4) if len(set(y))>1 else None,
            "confusion":{"tp":tp,"tn":tn,"fp":fp,"fn":fn}}

def load_batch(paths):
    return torch.stack([TF(Image.open(p).convert("RGB")) for p in paths])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bach",required=True); ap.add_argument("--model",default="models/final_cancer_model.pth")
    ap.add_argument("--train-per-class",type=int,default=50); ap.add_argument("--epochs",type=int,default=15)
    ap.add_argument("--lr",type=float,default=1e-4); ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--out",default="results/bach_finetune_results.json")
    a=ap.parse_args(); os.makedirs("results",exist_ok=True)
    items=gather(a.bach)
    if not items: raise SystemExit("No BACH class images found under --bach.")
    rng=random.Random(a.seed)
    by={0:[],1:[]}
    for p,l in items: by[l].append(p)
    for l in by: rng.shuffle(by[l])
    train=[(p,l) for l in (0,1) for p in by[l][:a.train_per_class]]
    test =[(p,l) for l in (0,1) for p in by[l][a.train_per_class:]]
    print(f"train {len(train)} | test {len(test)}")

    # ZERO-SHOT with original 8-class model
    net=CancerNet(num_classes=len(SUBTYPES),pretrained=False).to(DEVICE)
    net.load_state_dict(torch.load(a.model,map_location=DEVICE)); net.eval()
    yt=[l for _,l in test]; zs_p=[]; zs_s=[]
    with torch.no_grad():
        for i in range(0,len(test),16):
            xb=load_batch([p for p,_ in test[i:i+16]]).to(DEVICE)
            pr=torch.softmax(net(xb),1)
            pm=pr[:,MAL_IDX].sum(1).cpu().numpy()
            zs_s+=list(pm); zs_p+=list((pm>=0.5).astype(int))
    zero=metrics(yt,zs_p,zs_s)
    print("zero-shot:",zero)

    # FINE-TUNE a binary head
    net2=CancerNet(num_classes=len(SUBTYPES),pretrained=False).to(DEVICE)
    net2.load_state_dict(torch.load(a.model,map_location=DEVICE))
    net2.fc[3]=nn.Linear(512,2).to(DEVICE)      # replace final layer (512->2)
    opt=torch.optim.Adam(net2.parameters(),lr=a.lr); crit=nn.CrossEntropyLoss()
    net2.train()
    for ep in range(a.epochs):
        rng.shuffle(train); tot=0
        for i in range(0,len(train),16):
            xb=load_batch([p for p,_ in train[i:i+16]]).to(DEVICE)
            yb=torch.tensor([l for _,l in train[i:i+16]]).to(DEVICE)
            opt.zero_grad(); loss=crit(net2(xb),yb); loss.backward(); opt.step(); tot+=loss.item()
        print(f"  finetune epoch {ep+1}/{a.epochs} loss={tot:.3f}")
    net2.eval(); ft_p=[]; ft_s=[]
    with torch.no_grad():
        for i in range(0,len(test),16):
            xb=load_batch([p for p,_ in test[i:i+16]]).to(DEVICE)
            pr=torch.softmax(net2(xb),1)[:,1].cpu().numpy()
            ft_s+=list(pr); ft_p+=list((pr>=0.5).astype(int))
    fine=metrics(yt,ft_p,ft_s)
    print("fine-tuned:",fine)
    out={"dataset":"BACH ICIAR-2018","train_per_class":a.train_per_class,"n_train":len(train),"n_test":len(test),
         "zero_shot":zero,"fine_tuned":fine}
    json.dump(out,open(a.out,"w"),indent=2); print("saved",a.out)

if __name__=="__main__": main()
