"""
eval_hyperprior_baseline.py — matched SOTA learned-codec baseline for Table 4.

Runs a PRETRAINED scale-hyperprior (Balle et al., 2018 / CompressAI "bmshj2018-hyperprior")
on the SAME held-out BreakHis images used for our codec, at several quality levels, and
records size (bytes), PSNR, and SSIM in the SAME JSON shape as compression_comparison.json
so the numbers drop straight into Table 4.

NOTE ON INTEGRITY: this evaluates a pretrained model out-of-the-box (a fair, reproducible
SOTA reference). For a fully matched comparison you may FINE-TUNE it on BreakHis first with
--finetune-epochs > 0; report clearly in the paper whichever you use. Do NOT mix the two.

Requires a machine with PyTorch + CompressAI (needs a C++ build toolchain):
    pip install compressai            # or: pip install torch torchvision compressai
Run (GPU recommended):
    python eval_hyperprior_baseline.py --data "path/to/BreakHis_8class" \
        --eval-n 400 --qualities 1 3 5 --out results/hyperprior_results.json
"""
import argparse, json, os, random
import numpy as np

def main():
    import torch
    from PIL import Image
    import torchvision.transforms as T
    from skimage.metrics import structural_similarity as ssim
    from compressai.zoo import bmshj2018_hyperprior

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval-n", type=int, default=400)
    ap.add_argument("--qualities", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--finetune-epochs", type=int, default=0, help=">0 to fine-tune on BreakHis")
    ap.add_argument("--out", default="results/hyperprior_results.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # gather images (same 224x224 eval transform as our codec)
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    paths = []
    for r, _, fs in os.walk(a.data):
        for f in fs:
            if f.lower().endswith(exts): paths.append(os.path.join(r, f))
    random.seed(43); random.shuffle(paths); paths = paths[: a.eval_n]
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    def psnr_ssim(o, r):
        o8 = (o.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
        r8 = (r.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
        mse = float(((o8.astype(float)-r8.astype(float))**2).mean())
        p = 99.0 if mse == 0 else 10*np.log10(255.0**2/mse)
        import PIL.Image as I
        og = np.asarray(I.fromarray(o8).convert("L")); rg = np.asarray(I.fromarray(r8).convert("L"))
        return p, float(ssim(og, rg))

    out = {"resolution":"224x224","model":"bmshj2018-hyperprior (CompressAI, pretrained)",
           "finetune_epochs":a.finetune_epochs,"n":len(paths),"points":{}}
    for q in a.qualities:
        net = bmshj2018_hyperprior(quality=q, pretrained=True).eval().to(dev)
        # (optional) fine-tuning left as an exercise: see CompressAI examples/train.py
        byts, ps, ss = [], [], []
        with torch.no_grad():
            for p in paths:
                x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
                res = net.compress(x)                        # real entropy-coded bytes
                nbytes = sum(len(s[0]) for s in res["strings"])
                xh = net.decompress(res["strings"], res["shape"])["x_hat"].clamp(0,1)[0]
                pv, sv = psnr_ssim(x[0], xh)
                byts.append(nbytes); ps.append(pv); ss.append(sv)
        out["points"][str(q)] = {"avg_bytes":round(float(np.mean(byts)),1),
            "avg_kb":round(float(np.mean(byts))/1024,2),
            "avg_psnr":round(float(np.mean(ps)),2),"avg_ssim":round(float(np.mean(ss)),4)}
        print(f"quality {q}: {out['points'][str(q)]}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out,"w"), indent=2)
    print("saved", a.out)

if __name__ == "__main__":
    main()
