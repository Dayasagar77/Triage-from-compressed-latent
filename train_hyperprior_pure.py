"""
train_hyperprior_pure.py — a Balle-2018 SCALE-HYPERPRIOR codec in PURE PyTorch.

Why this file exists: CompressAI needs a C++ toolchain (the `ans` range-coder) that is
painful to install on Windows. This implementation needs ONLY torch/torchvision/PIL/skimage
— the same stack you already used to train the classifier — and reports rate as the
entropy-model's estimated bits, EXACTLY like our own learned_codec.py (v2). That makes it a
fair, matched learned-codec baseline for Table 4.

It reuses GDN/conv/deconv from learned_codec.py, adds a hyper-analysis/synthesis and a
Gaussian conditional entropy model, trains on BreakHis at a few lambdas, and writes
results/hyperprior_results.json in the same shape as learned_codec_results.json.

Run on your GPU machine:
    python train_hyperprior_pure.py --data "path/to/BreakHis_8class" \
        --epochs 40 --lambdas 0.002 0.01 0.05 --eval-n 400
Then send me results/hyperprior_results.json and I will insert the rows into Table 4.
"""
import argparse, json, os, math, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from learned_codec import GDN, conv, deconv   # reuse the same building blocks as v2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ScaleHyperprior(nn.Module):
    """224x224 -> y(14x14xM); hyperprior z gives per-element scales for a Gaussian on y."""
    def __init__(self, N=128, M=192):
        super().__init__()
        self.M = M
        self.g_a = nn.Sequential(conv(3,N),GDN(N),conv(N,N),GDN(N),conv(N,N),GDN(N),conv(N,M))
        self.g_s = nn.Sequential(deconv(M,N),GDN(N,inverse=True),deconv(N,N),GDN(N,inverse=True),
                                 deconv(N,N),GDN(N,inverse=True),deconv(N,3),nn.Sigmoid())
        # hyper analysis/synthesis (operate on |y| -> z -> scales)
        self.h_a = nn.Sequential(nn.Conv2d(M,N,3,1,1),nn.ReLU(inplace=True),
                                 nn.Conv2d(N,N,5,2,2),nn.ReLU(inplace=True),nn.Conv2d(N,N,5,2,2))
        self.h_s = nn.Sequential(nn.ConvTranspose2d(N,N,5,2,2,output_padding=1),nn.ReLU(inplace=True),
                                 nn.ConvTranspose2d(N,N,5,2,2,output_padding=1),nn.ReLU(inplace=True),
                                 nn.Conv2d(N,M,3,1,1),nn.Softplus())
        # factorized prior for the hyper-latent z
        self.z_mean = nn.Parameter(torch.zeros(N)); self.z_logscale = nn.Parameter(torch.zeros(N))

    def quantize(self, v):
        return v + (torch.rand_like(v)-0.5) if self.training else torch.round(v)

    @staticmethod
    def _gauss_bits(v, scale, mean=None):
        if mean is None: mean = torch.zeros_like(v)
        d = torch.distributions.Normal(mean, scale)
        p = (d.cdf(v+0.5)-d.cdf(v-0.5)).clamp(min=1e-9)
        return (-torch.log2(p)).sum()

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat = self.quantize(z)
        scales = self.h_s(z_hat)
        if scales.shape[-2:] != y.shape[-2:]:
            scales = F.interpolate(scales, size=y.shape[-2:], mode="nearest")
        scales = scales.clamp(min=0.11)
        y_hat = self.quantize(y)
        x_hat = self.g_s(y_hat).clamp(0,1)
        zscale = F.softplus(self.z_logscale).clamp(min=0.11).view(1,-1,1,1)
        zmean  = self.z_mean.view(1,-1,1,1)
        bits = self._gauss_bits(y_hat, scales) + self._gauss_bits(z_hat, zscale, zmean)
        return {"x_hat":x_hat, "bits":bits}

def psnr_ssim(o,r):
    o8=(o.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
    r8=(r.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
    mse=float(((o8.astype(float)-r8.astype(float))**2).mean())
    p=99.0 if mse==0 else 10*np.log10(255.0**2/mse)
    og=np.asarray(Image.fromarray(o8).convert("L")); rg=np.asarray(Image.fromarray(r8).convert("L"))
    return p, float(ssim(og,rg))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--epochs",type=int,default=40)
    ap.add_argument("--lambdas",type=float,nargs="+",default=[0.002,0.01,0.05])
    ap.add_argument("--batch",type=int,default=16); ap.add_argument("--lr",type=float,default=1e-4)
    ap.add_argument("--eval-n",type=int,default=400); ap.add_argument("--out",default="results/hyperprior_results.json")
    a=ap.parse_args()
    tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    ds=datasets.ImageFolder(a.data,transform=tf)
    n=len(ds); idx=list(range(n)); random.Random(43).shuffle(idx)
    eval_idx=idx[:a.eval_n]; train_idx=idx[a.eval_n:]
    out={"resolution":"224x224","model":"scale-hyperprior (pure PyTorch), estimated-bpp rate",
         "rate_note":"bytes = entropy-model bits / 8 (matched to our v2 codec)","points":{}}
    for lm in a.lambdas:
        model=ScaleHyperprior().to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=a.lr)
        loader=DataLoader(torch.utils.data.Subset(ds,train_idx),batch_size=a.batch,shuffle=True,num_workers=2)
        model.train()
        for ep in range(a.epochs):
            tot=0
            for x,_ in loader:
                x=x.to(DEVICE); o=model(x)
                npix=x.shape[0]*x.shape[2]*x.shape[3]
                loss=lm*(255**2)*F.mse_loss(o["x_hat"],x)+o["bits"]/npix
                opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
            print(f"  lambda={lm} epoch {ep+1}/{a.epochs} loss={tot/len(loader):.3f}")
        model.eval(); bts,ps,ss=[],[],[]
        with torch.no_grad():
            for i in eval_idx:
                x,_=ds[i]; x=x.unsqueeze(0).to(DEVICE); o=model(x)
                bts.append(float(o["bits"])/8.0)
                p,s=psnr_ssim(x[0],o["x_hat"][0]); ps.append(p); ss.append(s)
        out["points"][str(lm)]={"avg_bytes":round(float(np.mean(bts)),1),
            "avg_kb":round(float(np.mean(bts))/1024,2),
            "avg_psnr":round(float(np.mean(ps)),2),"avg_ssim":round(float(np.mean(ss)),4)}
        print(f"lambda {lm}: {out['points'][str(lm)]}")
        os.makedirs("models",exist_ok=True)
        torch.save(model.state_dict(),f"models/hyperprior_l{lm}.pth")
    os.makedirs(os.path.dirname(a.out),exist_ok=True); json.dump(out,open(a.out,"w"),indent=2)
    print("saved",a.out)

if __name__=="__main__":
    main()
