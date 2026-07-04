"""
latent_umap.py — interpretability of the ~5KB compressed latent (reviewer request).
Encodes BreakHis with the v2 codec, global-average-pools each 14x14xM latent to an
M-vector, projects to 2-D (UMAP if available, else t-SNE), and plots three panels:
benign/malignant, 8 subtypes, and magnification. Also reports the silhouette score of
the benign/malignant split in latent space (a number for 'does the latent cluster by
morphology?'). If clusters follow magnification/stain more than class, that supports the
BACH domain-shift hypothesis.

Run:  python latent_umap.py --data "path/to/BreakHis_8class" --codec-lambda 0.01
Output: results/figures/fig_latent_umap.png , results/latent_umap.json   (send me the json)
"""
import argparse, os, json, re
import numpy as np, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from learned_codec import FactorizedCodec
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
TF=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
BENIGN={"Adenosis","Fibroadenoma","Phyllodes Tumor","Tubular Adenoma"}
def mag_of(path):
    m=re.search(r"-(40|100|200|400)-",os.path.basename(path))
    return m.group(1)+"X" if m else "?"

def project(X):
    try:
        import umap
        return umap.UMAP(n_neighbors=25,min_dist=0.1,random_state=0).fit_transform(X),"UMAP"
    except Exception as e:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2,init="pca",random_state=0,perplexity=30).fit_transform(X),"t-SNE"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--codec-lambda",type=float,default=0.01)
    ap.add_argument("--max",type=int,default=4000,help="subsample for speed")
    a=ap.parse_args(); os.makedirs("results/figures",exist_ok=True)
    ds=datasets.ImageFolder(a.data,transform=TF); classes=ds.classes
    paths=[p for p,_ in ds.samples]; labels=np.array(ds.targets)
    idx=np.arange(len(ds))
    if len(ds)>a.max:
        rng=np.random.RandomState(0); idx=rng.choice(idx,a.max,replace=False)
    codec=FactorizedCodec().to(DEVICE)
    sd=os.path.join("models",f"learned_codec_l{a.codec_lambda}.pth")
    codec.load_state_dict(torch.load(sd,map_location=DEVICE)); codec.eval()
    for p in codec.parameters(): p.requires_grad_(False)
    # encode selected images -> GAP to M-vector
    feats=[]; sel=sorted(idx.tolist())
    sub=torch.utils.data.Subset(ds,sel)
    with torch.no_grad():
        for x,_ in DataLoader(sub,batch_size=32,shuffle=False,num_workers=0):
            y=codec.encode_latent(x.to(DEVICE))          # B,M,14,14 quantized
            feats.append(y.mean(dim=(2,3)).cpu().numpy())  # global avg pool -> B,M
    X=np.concatenate(feats); 
    y_sub=labels[sel]; bm=np.array([0 if classes[c] in BENIGN else 1 for c in y_sub])
    mags=np.array([mag_of(paths[i]) for i in sel])
    sil_bm=float(silhouette_score(X,bm))
    sil_mag=float(silhouette_score(X,mags)) if len(set(mags))>1 else None
    Z,method=project(X)
    fig,ax=plt.subplots(1,3,figsize=(16,5))
    # panel 1 B/M
    for lab,col,name in [(0,"tab:green","Benign"),(1,"tab:red","Malignant")]:
        m=bm==lab; ax[0].scatter(Z[m,0],Z[m,1],s=6,c=col,alpha=0.5,label=name)
    ax[0].set_title(f"Benign vs Malignant (silhouette={sil_bm:.3f})"); ax[0].legend()
    # panel 2 subtype
    for c in range(len(classes)):
        m=y_sub==c; ax[1].scatter(Z[m,0],Z[m,1],s=6,alpha=0.5,label=classes[c])
    ax[1].set_title("8 subtypes"); ax[1].legend(fontsize=6,markerscale=2)
    # panel 3 magnification
    for mg in sorted(set(mags)):
        m=mags==mg; ax[2].scatter(Z[m,0],Z[m,1],s=6,alpha=0.5,label=mg)
    ax[2].set_title(f"Magnification (silhouette={sil_mag:.3f})" if sil_mag else "Magnification"); ax[2].legend()
    for a_ in ax: a_.set_xticks([]); a_.set_yticks([])
    plt.suptitle(f"v2 latent ({method} projection, n={len(X)})")
    plt.tight_layout(); plt.savefig("results/figures/fig_latent_umap.png",dpi=140); 
    out={"method":method,"n":len(X),"latent_dim_pooled":X.shape[1],
         "silhouette_benign_malignant":round(sil_bm,4),
         "silhouette_magnification":round(sil_mag,4) if sil_mag else None,
         "interpretation":"higher B/M silhouette than magnification silhouette => latent clusters by morphology more than by scanner/stain proxy"}
    json.dump(out,open("results/latent_umap.json","w"),indent=2)
    print(json.dumps(out,indent=2)); print("saved fig_latent_umap.png + latent_umap.json")

if __name__=="__main__": main()
