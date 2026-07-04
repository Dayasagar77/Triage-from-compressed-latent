"""
edge_latency.py — measure encoder inference latency to ground the edge-compute discussion
(the ResNet-50 dominates edge compute; is the lightweight codec really 'cheaper at the edge'?). Reports parameter
counts and ms/patch for the source classifier (CancerNet), the v2 codec encoder, and the
hyperprior encoder, on GPU and CPU (CPU is the honest edge proxy).

Run:  python edge_latency.py
Output: results/edge_latency.json   (send me this)
"""
import json, os, time, numpy as np, torch
from learned_codec import FactorizedCodec
from train_hyperprior_pure import ScaleHyperprior
from ai_engine import CancerNet, SUBTYPES

def nparams(m): return sum(p.numel() for p in m.parameters())

def bench(fn, device, n=50, warm=10):
    x=torch.randn(1,3,224,224,device=device)
    for _ in range(warm): fn(x)
    if device=="cuda": torch.cuda.synchronize()
    ts=[]
    for _ in range(n):
        t0=time.perf_counter(); fn(x)
        if device=="cuda": torch.cuda.synchronize()
        ts.append((time.perf_counter()-t0)*1000)
    return round(float(np.mean(ts)),2), round(float(np.std(ts)),2)

def main():
    os.makedirs("results",exist_ok=True)
    out={"note":"ms per 224x224 patch, batch=1; CPU is the edge proxy","models":{}}
    v2=FactorizedCodec().eval(); hp=ScaleHyperprior().eval()
    net=CancerNet(num_classes=len(SUBTYPES),pretrained=False).eval()
    specs=[("CancerNet_ResNet50_SE", net, lambda m,x: m(x)),
           ("v2_encoder", v2, lambda m,x: m.encode_latent(x)),
           ("hyperprior_encoder", hp, lambda m,x: m.g_a(x))]
    for dev in (["cuda"] if torch.cuda.is_available() else [])+["cpu"]:
        for name,m,call in specs:
            m.to(dev)
            with torch.no_grad():
                mean,std=bench(lambda x: call(m,x), dev)
            out["models"].setdefault(name,{"params_millions":round(nparams(m)/1e6,2)})
            out["models"][name][f"{dev}_ms_mean"]=mean
            out["models"][name][f"{dev}_ms_std"]=std
            print(f"{dev:4s} {name:26s} {nparams(m)/1e6:6.2f}M  {mean:8.2f} ± {std:.2f} ms")
    json.dump(out,open("results/edge_latency.json","w"),indent=2)
    print("saved results/edge_latency.json")

if __name__=="__main__": main()
