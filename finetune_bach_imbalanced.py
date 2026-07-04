"""
finetune_bach_imbalanced.py — real-world domain adaptation rarely has
balanced classes. Repeats the BACH fine-tuning but with an IMBALANCED fine-tune set
(default 75 benign / 25 malignant, reflecting a benign-heavy screening prevalence), while
evaluating on the SAME fixed BALANCED held-out test set so results are comparable to the
50/50 fine-tune in Table 16.

Run:  python finetune_bach_imbalanced.py \
        --bach "path/to/BACH/ICIAR2018_BACH_Challenge/ICIAR2018_BACH_Challenge/Photos" \
        --model models/final_cancer_model.pth --benign-train 75 --malignant-train 25 --test-per-class 125 --epochs 15
Output: results/bach_finetune_imbalanced.json  (send me this)
"""
import argparse, os, json, random
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from ai_engine import CancerNet, SUBTYPES, BENIGN_SUBTYPES

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
TF=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
MAL_IDX=[i for i,n in enumerate(SUBTYPES) if n not in BENIGN_SUBTYPES]
def bach_binary(name):
    n=name.strip().lower().replace("-"," ").replace("_"," ")
    if n in ("normal","n","benign","b"): return 0
    if n in ("in situ","insitu","is","invasive","iv"): return 1
    return None
def gather(bach):
    exts=(".tif",".tiff",".png",".jpg",".jpeg",".bmp"); items={0:[],1:[]}
    for root,_,files in os.walk(bach):
        lab=bach_binary(os.path.basename(root))
        if lab is None: continue
        for f in files:
            if f.lower().endswith(exts): items[lab].append(os.path.join(root,f))
    return items
def metrics(y,p,s):
    y,p,s=np.array(y),np.array(p),np.array(s)
    tp=int(((p==1)&(y==1)).sum()); tn=int(((p==0)&(y==0)).sum())
    fp=int(((p==1)&(y==0)).sum()); fn=int(((p==0)&(y==1)).sum())
    return {"accuracy":round((tp+tn)/len(y),4),"sensitivity":round(tp/max(tp+fn,1),4),
            "specificity":round(tn/max(tn+fp,1),4),
            "auc":round(float(roc_auc_score(y,s)),4) if len(set(y))>1 else None,
            "balanced_accuracy":round(0.5*(tp/max(tp+fn,1)+tn/max(tn+fp,1)),4),
            "confusion":{"tp":tp,"tn":tn,"fp":fp,"fn":fn}}
def load_batch(paths): return torch.stack([TF(Image.open(p).convert("RGB")) for p in paths])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bach",required=True); ap.add_argument("--model",default="models/final_cancer_model.pth")
    ap.add_argument("--benign-train",type=int,default=75); ap.add_argument("--malignant-train",type=int,default=25)
    ap.add_argument("--test-per-class",type=int,default=125); ap.add_argument("--epochs",type=int,default=15)
    ap.add_argument("--lr",type=float,default=1e-4); ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--out",default="results/bach_finetune_imbalanced.json")
    a=ap.parse_args(); os.makedirs("results",exist_ok=True)
    by=gather(a.bach)
    if not by[0] or not by[1]: raise SystemExit("No BACH class images found under --bach.")
    rng=random.Random(a.seed)
    for l in by: rng.shuffle(by[l])
    # reserve a BALANCED test set first (last test-per-class of each), train from the front
    T=a.test_per_class
    test=[(p,0) for p in by[0][-T:]]+[(p,1) for p in by[1][-T:]]
    train=[(p,0) for p in by[0][:a.benign_train]]+[(p,1) for p in by[1][:a.malignant_train]]
    assert a.benign_train+T<=len(by[0]) and a.malignant_train+T<=len(by[1]), "not enough images; lower test-per-class"
    print(f"train {len(train)} (benign {a.benign_train}/malignant {a.malignant_train}) | test {len(test)} (balanced {T}/{T})")
    yt=[l for _,l in test]
    # zero-shot 8-class
    net=CancerNet(num_classes=len(SUBTYPES),pretrained=False).to(DEVICE)
    net.load_state_dict(torch.load(a.model,map_location=DEVICE)); net.eval()
    zs_p=[]; zs_s=[]
    with torch.no_grad():
        for i in range(0,len(test),16):
            xb=load_batch([p for p,_ in test[i:i+16]]).to(DEVICE)
            pm=torch.softmax(net(xb),1)[:,MAL_IDX].sum(1).cpu().numpy()
            zs_s+=list(pm); zs_p+=list((pm>=0.5).astype(int))
    zero=metrics(yt,zs_p,zs_s); print("zero-shot:",zero)
    # fine-tune binary head on IMBALANCED train (class-weighted loss to be fair)
    net2=CancerNet(num_classes=len(SUBTYPES),pretrained=False).to(DEVICE)
    net2.load_state_dict(torch.load(a.model,map_location=DEVICE)); net2.fc[3]=nn.Linear(512,2).to(DEVICE)
    w=torch.tensor([1.0/max(a.benign_train,1),1.0/max(a.malignant_train,1)],device=DEVICE); w=w/w.sum()*2
    opt=torch.optim.Adam(net2.parameters(),lr=a.lr); crit=nn.CrossEntropyLoss(weight=w)
    net2.train()
    for ep in range(a.epochs):
        rng.shuffle(train); tot=0
        for i in range(0,len(train),16):
            xb=load_batch([p for p,_ in train[i:i+16]]).to(DEVICE)
            yb=torch.tensor([l for _,l in train[i:i+16]]).to(DEVICE)
            opt.zero_grad(); loss=crit(net2(xb),yb); loss.backward(); opt.step(); tot+=loss.item()
        print(f"  epoch {ep+1}/{a.epochs} loss={tot:.3f}")
    net2.eval(); ft_p=[]; ft_s=[]
    with torch.no_grad():
        for i in range(0,len(test),16):
            xb=load_batch([p for p,_ in test[i:i+16]]).to(DEVICE)
            pr=torch.softmax(net2(xb),1)[:,1].cpu().numpy()
            ft_s+=list(pr); ft_p+=list((pr>=0.5).astype(int))
    fine=metrics(yt,ft_p,ft_s); print("fine-tuned (imbalanced train):",fine)
    out={"dataset":"BACH ICIAR-2018 (imbalanced fine-tune)","train_benign":a.benign_train,
         "train_malignant":a.malignant_train,"n_test_balanced":len(test),
         "class_weighted_loss":True,"zero_shot":zero,"fine_tuned_imbalanced":fine,
         "compare_balanced_5050":{"accuracy":0.8133,"sensitivity":0.90,"specificity":0.7267,"auc":0.9125}}
    json.dump(out,open(a.out,"w"),indent=2); print("saved",a.out)

if __name__=="__main__": main()
