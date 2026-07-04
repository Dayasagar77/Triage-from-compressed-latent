"""
compare_compression.py — FAIR rate-distortion comparison of the neural compressor
against JPEG and JPEG2000, all at the SAME 224x224 resolution.

This separates "resize" from "neural coding": every method compresses the identical
224x224 image, so differences reflect the codec, not resolution. This is the honest
test of whether the neural compressor beats standard codecs.

    python compare_compression.py --data "D:\\...\\BreakHis_8class" --n 400

Output: results/compression_comparison.json  (bytes vs PSNR/SSIM per method/operating point)
Needs trained models/clinical_encoder.pth + clinical_decoder.pth (run train_compression.py first).
"""
import argparse, io, json, os
import numpy as np
import torch
from torchvision import datasets, transforms
from skimage.metrics import structural_similarity as ssim
from PIL import Image

from ai_engine import (ClinicalEncoder, ClinicalDecoder, DEVICE,
                       _pack_latent, reconstruct_image)

TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


def metrics(orig_u8, rec_u8):
    mse = float(np.mean((orig_u8.astype(np.float32) - rec_u8.astype(np.float32)) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * np.log10((255.0 ** 2) / mse)
    og = np.dot(orig_u8[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    rg = np.dot(rec_u8[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return psnr, float(ssim(og, rg))


def jpeg_bytes(pil224, q):
    b = io.BytesIO(); pil224.save(b, format="JPEG", quality=q)
    data = b.getvalue()
    rec = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    return len(data), rec


def jp2_bytes(pil224, ratio):
    try:
        b = io.BytesIO()
        pil224.save(b, format="JPEG2000", quality_mode="rates", quality_layers=[ratio])
        data = b.getvalue()
        rec = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        return len(data), rec
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    enc = ClinicalEncoder().to(DEVICE).eval()
    dec = ClinicalDecoder().to(DEVICE).eval()
    enc.load_state_dict(torch.load("models/clinical_encoder.pth", map_location=DEVICE))
    dec.load_state_dict(torch.load("models/clinical_decoder.pth", map_location=DEVICE))

    ds = datasets.ImageFolder(args.data, transform=TF)
    n = len(ds); rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n); val_idx = perm[:int(n*args.val_frac)]
    rng2 = np.random.default_rng(args.seed + 1)
    pick = rng2.choice(val_idx, size=min(args.n, len(val_idx)), replace=False)

    JPEG_Q = [10, 20, 30, 50, 70, 85, 95]
    JP2_R = [80, 40, 20, 10, 5]   # compression ratios (higher = smaller)
    NEURAL_Q = [90, 50, 25]        # -> float16 / int8 / int8+downsample latent

    acc = {"neural": {q: {"bytes": [], "psnr": [], "ssim": []} for q in NEURAL_Q},
           "jpeg":   {q: {"bytes": [], "psnr": [], "ssim": []} for q in JPEG_Q},
           "jpeg2000": {r: {"bytes": [], "psnr": [], "ssim": []} for r in JP2_R}}

    for idx in pick:
        img_t, _ = ds[int(idx)]
        orig_u8 = (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(orig_u8)

        # Neural (encode once, pack at each mode, decode)
        with torch.no_grad():
            latent = enc(img_t.unsqueeze(0).to(DEVICE)).squeeze(0).cpu().numpy().astype(np.float32)
        for q in NEURAL_Q:
            from ai_engine import _quality_to_mode
            packed = _pack_latent(latent, _quality_to_mode(q))
            rec = np.array(Image.open(io.BytesIO(reconstruct_image(packed))).convert("RGB"))
            if rec.shape[:2] != (224, 224):
                rec = np.array(Image.fromarray(rec).resize((224, 224)))
            p, s = metrics(orig_u8, rec)
            acc["neural"][q]["bytes"].append(len(packed)); acc["neural"][q]["psnr"].append(p); acc["neural"][q]["ssim"].append(s)

        for q in JPEG_Q:
            nb, rec = jpeg_bytes(pil, q); p, s = metrics(orig_u8, rec)
            acc["jpeg"][q]["bytes"].append(nb); acc["jpeg"][q]["psnr"].append(p); acc["jpeg"][q]["ssim"].append(s)

        for r in JP2_R:
            nb, rec = jp2_bytes(pil, r)
            if nb is not None:
                p, s = metrics(orig_u8, rec)
                acc["jpeg2000"][r]["bytes"].append(nb); acc["jpeg2000"][r]["psnr"].append(p); acc["jpeg2000"][r]["ssim"].append(s)

    def summarize(d):
        out = {}
        for k, v in d.items():
            if not v["bytes"]:
                continue
            out[str(k)] = {"avg_bytes": round(float(np.mean(v["bytes"])), 1),
                           "avg_psnr": round(float(np.mean([x for x in v["psnr"] if np.isfinite(x)])), 2),
                           "avg_ssim": round(float(np.mean(v["ssim"])), 4)}
        return out

    result = {"n": int(len(pick)), "resolution": "224x224",
              "neural": summarize(acc["neural"]),
              "jpeg": summarize(acc["jpeg"]),
              "jpeg2000": summarize(acc["jpeg2000"])}
    json.dump(result, open("results/compression_comparison.json", "w"), indent=2)

    # Honest headline: at the neural operating point (q=90), what JPEG quality matches its SSIM?
    nq90 = result["neural"].get("90")
    print("\n==== FAIR COMPRESSION COMPARISON (224x224) ====")
    print(f"Neural (float16 latent): {nq90['avg_bytes']} B, PSNR {nq90['avg_psnr']} dB, SSIM {nq90['avg_ssim']}")
    for q in JPEG_Q:
        j = result["jpeg"].get(str(q))
        if j: print(f"JPEG q={q:>2}: {j['avg_bytes']:>8.1f} B, PSNR {j['avg_psnr']:>5.2f} dB, SSIM {j['avg_ssim']}")
    print("\nInterpretation is written to results/compression_comparison.json for the paper.")


if __name__ == "__main__":
    main()
