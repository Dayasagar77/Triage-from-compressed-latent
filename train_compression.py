"""
train_compression.py — train the neural compression autoencoder
(ClinicalEncoder + ClinicalDecoder) on BreakHis and measure REAL reconstruction
quality (PSNR / SSIM / MSE) and compression ratio. This produces genuine numbers
for the paper's compression section — nothing is hard-coded or simulated.

USAGE:
    python train_compression.py --data "D:\\...\\BreakHis_8class" --epochs 30 --batch 16

OUTPUTS:
    models/clinical_encoder.pth, models/clinical_decoder.pth
    results/compression_results.json   (PSNR/SSIM/MSE mean+/-std, compression ratio)

NOTES ON HONESTY:
    - Reconstruction quality (PSNR/SSIM/MSE) is measured on a held-out split the
      autoencoder never saw during training.
    - Compression ratio is reported two ways so the reader can judge it fairly:
        (a) latent bytes vs the 224x224 RGB pixel buffer actually fed to the model;
        (b) latent bytes vs the original image file on disk (this also includes the
            effect of resizing the image to 224x224, so it is an upper bound).
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from skimage.metrics import structural_similarity as ssim

from ai_engine import ClinicalEncoder, ClinicalDecoder, DEVICE, _pack_latent, _quality_to_mode

# Encoder/decoder operate on [0,1] tensors (decoder ends in Sigmoid). No normalization.
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
RAW_PIXEL_BYTES = 3 * 224 * 224  # uint8 RGB buffer size the model actually ingests


def psnr_ssim_mse(orig_t, recon_t):
    """orig_t, recon_t: (3,H,W) tensors in [0,1]. Returns (psnr, ssim, mse)."""
    o = (orig_t.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    r = (recon_t.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    mse = float(np.mean((o.astype(np.float32) - r.astype(np.float32)) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * np.log10((255.0 ** 2) / mse)
    og = np.dot(o[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    rg = np.dot(r[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    s = ssim(og, rg)
    return psnr, float(s), mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="BreakHis ImageFolder root (8 subtype dirs)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--quality", type=int, default=85, help="Latent quantization quality for the reported ratio")
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    ds = datasets.ImageFolder(args.data, transform=TF)
    paths = [p for p, _ in ds.samples]
    n = len(ds)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_val = int(n * args.val_frac)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    print(f"Images: {n}  train: {len(train_idx)}  val: {len(val_idx)}")

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch, shuffle=False)

    enc = ClinicalEncoder().to(DEVICE)
    dec = ClinicalDecoder().to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    for ep in range(args.epochs):
        enc.train(); dec.train()
        running = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.to(DEVICE)
            opt.zero_grad()
            recon = dec(enc(imgs))
            loss = loss_fn(recon, imgs)
            loss.backward()
            opt.step()
            running += loss.item() * imgs.size(0)
        sched.step()
        print(f"  epoch {ep+1}/{args.epochs}  train MSE={running/len(train_idx):.6f}")

    # ---- Evaluation on held-out val set ----
    enc.eval(); dec.eval()
    psnrs, ssims, mses = [], [], []
    latent_bytes_total, raw_bytes_total, file_bytes_total = 0, 0, 0
    mode = _quality_to_mode(args.quality)
    with torch.no_grad():
        vi = 0
        for imgs, _ in val_loader:
            imgs = imgs.to(DEVICE)
            latent = enc(imgs)
            recon = dec(latent)
            for b in range(imgs.size(0)):
                p, s, m = psnr_ssim_mse(imgs[b], recon[b])
                if np.isfinite(p):
                    psnrs.append(p)
                ssims.append(s); mses.append(m)
                # compression ratio bookkeeping
                lat_np = latent[b].cpu().numpy().astype(np.float32)
                latent_bytes_total += len(_pack_latent(lat_np, mode))
                raw_bytes_total += RAW_PIXEL_BYTES
                file_bytes_total += os.path.getsize(paths[val_idx[vi]])
                vi += 1

    def ms(a):
        a = np.array(a, dtype=np.float64)
        return float(a.mean()), float(a.std())

    psnr_m, psnr_s = ms(psnrs)
    ssim_m, ssim_s = ms(ssims)
    mse_m, mse_s = ms(mses)
    savings_vs_raw = 100.0 * (1 - latent_bytes_total / raw_bytes_total)
    savings_vs_file = 100.0 * (1 - latent_bytes_total / file_bytes_total)

    results = {
        "n_val": len(val_idx),
        "quality": args.quality,
        "PSNR_dB_mean": round(psnr_m, 2), "PSNR_dB_std": round(psnr_s, 2),
        "SSIM_mean": round(ssim_m, 4), "SSIM_std": round(ssim_s, 4),
        "MSE_mean": round(mse_m, 6), "MSE_std": round(mse_s, 6),
        "compression_savings_vs_raw_pixels_pct": round(savings_vs_raw, 2),
        "compression_savings_vs_original_file_pct": round(savings_vs_file, 2),
        "avg_latent_bytes": round(latent_bytes_total / len(val_idx), 1),
    }
    with open("results/compression_results.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(enc.state_dict(), "models/clinical_encoder.pth")
    torch.save(dec.state_dict(), "models/clinical_decoder.pth")

    print("\n==== COMPRESSION RESULTS (held-out val) ====")
    print(f"PSNR : {psnr_m:.2f} +/- {psnr_s:.2f} dB")
    print(f"SSIM : {ssim_m:.4f} +/- {ssim_s:.4f}")
    print(f"MSE  : {mse_m:.6f} +/- {mse_s:.6f}")
    print(f"Compression savings vs raw 224x224 pixels : {savings_vs_raw:.2f}%")
    print(f"Compression savings vs original file       : {savings_vs_file:.2f}%")
    print(f"Avg latent size: {results['avg_latent_bytes']} bytes (quality={args.quality})")
    print("Saved models/clinical_encoder.pth, models/clinical_decoder.pth, results/compression_results.json")


if __name__ == "__main__":
    main()
