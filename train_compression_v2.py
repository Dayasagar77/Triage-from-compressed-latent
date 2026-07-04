"""
train_compression_v2.py — train the self-contained learned codec (learned_codec.py)
with a rate-distortion loss, at several lambda points, to trace an R-D curve that can
beat JPEG. Pure PyTorch, no CompressAI / no C++ build required.

    python train_compression_v2.py --data "D:\\...\\BreakHis_8class" --epochs 40 --lambdas 0.002 0.01 0.05

Rate is the entropy model's estimated bits (bytes = bits/8), the standard rate metric
in learned compression. Evaluated on the SAME 224x224 held-out split as the JPEG test.

Output: models/learned_codec_l{lambda}.pth , results/learned_codec_results.json
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from skimage.metrics import structural_similarity as ssim

from learned_codec import FactorizedCodec

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


def psnr_ssim(o_t, r_t):
    o = (o_t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    r = (r_t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    mse = float(np.mean((o.astype(np.float32) - r.astype(np.float32)) ** 2))
    psnr = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    og = np.dot(o[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    rg = np.dot(r[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return psnr, float(ssim(og, rg))


def train_point(lmbda, train_loader, ds, eval_idx, epochs, lr):
    model = FactorizedCodec().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train(); tot = 0.0
        for x, _ in train_loader:
            x = x.to(DEVICE); n_pix = x.shape[0] * x.shape[2] * x.shape[3]
            opt.zero_grad()
            out = model(x)
            mse = F.mse_loss(out["x_hat"], x)
            bpp = out["bits"] / n_pix
            loss = lmbda * (255 ** 2) * mse + bpp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item()
        sched.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"    lambda={lmbda} epoch {ep+1}/{epochs} loss={tot/len(train_loader):.3f}")
    # eval (rounding quantization -> real integer latent -> estimated entropy bytes)
    model.eval(); bts, ps, ss = [], [], []
    with torch.no_grad():
        for i in eval_idx:
            x, _ = ds[int(i)]; x = x.unsqueeze(0).to(DEVICE)
            out = model(x)
            bts.append(float(out["bits"].item()) / 8.0)   # bits -> bytes
            p, s = psnr_ssim(x[0], out["x_hat"][0]); ps.append(p); ss.append(s)
    torch.save(model.state_dict(), f"models/learned_codec_l{lmbda}.pth")
    return {"avg_bytes": round(float(np.mean(bts)), 1), "avg_kb": round(float(np.mean(bts))/1024, 2),
            "avg_psnr": round(float(np.mean(ps)), 2), "avg_ssim": round(float(np.mean(ss)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.002, 0.01, 0.05])
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--eval-n", type=int, default=400)
    args = ap.parse_args()
    os.makedirs("models", exist_ok=True); os.makedirs("results", exist_ok=True)

    ds = datasets.ImageFolder(args.data, transform=TF)
    n = len(ds); perm = np.random.default_rng(42).permutation(n)
    val_all = perm[:int(n * args.val_frac)]; train_idx = perm[int(n * args.val_frac):]
    eval_idx = np.random.default_rng(43).choice(val_all, size=min(args.eval_n, len(val_all)), replace=False)
    train_loader = DataLoader(Subset(ds, train_idx.tolist()), batch_size=args.batch, shuffle=True, num_workers=0)
    print(f"Images {n} | train {len(train_idx)} | eval {len(eval_idx)} | device {DEVICE}")

    res = {"resolution": "224x224", "model": "FactorizedCodec (GDN + factorized entropy), rate-distortion trained",
           "rate_note": "bytes = entropy-model bits / 8 (near-optimal arithmetic-coding rate)", "points": {}}
    for lm in args.lambdas:
        print(f"  === lambda {lm} ===")
        res["points"][str(lm)] = train_point(lm, train_loader, ds, eval_idx, args.epochs, args.lr)
        print(f"    -> {res['points'][str(lm)]}")
    json.dump(res, open("results/learned_codec_results.json", "w"), indent=2)
    print("\n==== LEARNED CODEC v2 (rate-distortion) ====")
    for lm, r in res["points"].items():
        print(f"  lambda {lm}: {r['avg_kb']} KB | PSNR {r['avg_psnr']} dB | SSIM {r['avg_ssim']}")
    print("Compare vs JPEG in results/compression_comparison.json (same val set). make_figures.py overlays both.")


if __name__ == "__main__":
    main()
