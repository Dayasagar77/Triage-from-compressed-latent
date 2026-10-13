"""
r4_actual_bitstream.py - Reviewer-4 comments #3 and #4 (actual encoded bytes).

Until now the learned-codec sizes in the paper were ENTROPY-MODEL ESTIMATES (bits/8), while the JPEG and
JPEG2000 sizes were actual file bytes. This script range-codes the quantized latents with the codecs' own
entropy models (constriction, a range coder), decodes them back, asserts the round trip is lossless, and
reports the ACTUAL bitstream bytes next to the estimate and next to JPEG / JPEG2000 bytes on the SAME images.

Bitstream layout (everything counted):
  factorized-prior codec : 2-byte y_min + 2-byte y_max + range-coded y (per-channel Gaussian, as in training)
  scale-hyperprior codec : 2+2 bytes z range + range-coded z (factorized Gaussian) + 2+2 bytes y range +
                           range-coded y (zero-mean Gaussian with scales from h_s(z_hat), as in training)
  Range-coder output is a sequence of 32-bit words, so each coded part is rounded up to a multiple of 4 bytes.
The latent shape is fixed for 224x224 input, so no shape header is needed.

Held-out images (each codec is evaluated only on images it did not train on):
  F-set: the 400-image evaluation set of train_compression_v2.py / compare_compression.py
         (numpy default_rng(42).permutation -> first 20% held out; default_rng(43).choice 400).
  H-set: the 400-image evaluation set of train_hyperprior_pure.py (random.Random(43).shuffle -> first 400).
JPEG and JPEG2000 are computed on both sets with the settings of compare_compression.py.
PSNR (RGB) and SSIM (luma) follow train_compression_v2.py / compare_compression.py.

    python r4_actual_bitstream.py --data "<BreakHis_8class>" [--n 400]
Output: results/actual_bitstream_results.json   (MEASURED bytes; images and codecs as stated)
"""
import argparse, io, json, os, random, time
import numpy as np, torch, torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import constriction
from learned_codec import FactorizedCodec
from train_hyperprior_pure import ScaleHyperprior

DEVICE = "cpu"                        # CPU keeps the entropy-model parameters bit-identical at encode and decode
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
QG = constriction.stream.model.QuantizedGaussian


def rng_bytes(sym, means, stds, fixed=None):
    """Range-code integer symbols with a discretized Gaussian per symbol; return (nbytes, decoded, support).
    fixed=None : support = this image's [min, max] (sent as two int16, 4 bytes of header)
    fixed=(a,b): a fixed support known to the decoder (no header); symbols must lie inside it."""
    if fixed is None:
        lo, hi = int(sym.min()), int(sym.max()); hi = max(hi, lo + 1); header = 4
    else:
        lo, hi = fixed; header = 0
        assert sym.min() >= lo and sym.max() <= hi, f"symbol outside fixed support {fixed}: {sym.min()}..{sym.max()}"
    model = QG(lo, hi)
    enc = constriction.stream.queue.RangeEncoder()
    enc.encode(sym.astype(np.int32).ravel(), model, means.astype(np.float64).ravel(), stds.astype(np.float64).ravel())
    words = enc.get_compressed()
    dec = constriction.stream.queue.RangeDecoder(words)
    out = dec.decode(model, means.astype(np.float64).ravel(), stds.astype(np.float64).ravel())
    return header + 4 * len(words), np.asarray(out, dtype=np.int64).reshape(sym.shape), (lo, hi)


FIXED = (-255, 255)     # wide fixed support for the no-header variant


def psnr_ssim(o_t, r_t):
    o = (o_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    r = (r_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return rgb_metrics(o, r)


def rgb_metrics(o, r):
    mse = float(np.mean((o.astype(np.float32) - r.astype(np.float32)) ** 2))
    psnr = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    og = np.dot(o[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    rg = np.dot(r[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return psnr, float(ssim(og, rg))


@torch.no_grad()
def code_factorized(m, x):
    y = m.g_a(x); y_hat = torch.round(y)
    mean = m.prior_mean.view(1, -1, 1, 1).expand_as(y_hat)
    scale = F.softplus(m.prior_logscale).clamp(min=0.11).view(1, -1, 1, 1).expand_as(y_hat)
    est_bits = float(m.rate_bits(y_hat))
    nbytes, dec, rng = rng_bytes(y_hat.numpy().astype(np.int64), mean.numpy(), scale.numpy())
    assert np.array_equal(dec, y_hat.numpy().astype(np.int64)), "factorized round trip failed"
    nfix, dec2, _ = rng_bytes(y_hat.numpy().astype(np.int64), mean.numpy(), scale.numpy(), fixed=FIXED)
    assert np.array_equal(dec2, dec), "factorized fixed-support round trip failed"
    x_hat = m.g_s(torch.from_numpy(dec.astype(np.float32))).clamp(0, 1)
    return nbytes, est_bits / 8.0, x_hat[0], {"fixed_support_bytes": nfix, "y_range": rng}


@torch.no_grad()
def code_hyperprior(m, x):
    y = m.g_a(x); z = m.h_a(torch.abs(y)); z_hat = torch.round(z)
    zmean = m.z_mean.view(1, -1, 1, 1).expand_as(z_hat)
    zscale = F.softplus(m.z_logscale).clamp(min=0.11).view(1, -1, 1, 1).expand_as(z_hat)
    zb, z_dec, zr = rng_bytes(z_hat.numpy().astype(np.int64), zmean.numpy(), zscale.numpy())
    assert np.array_equal(z_dec, z_hat.numpy().astype(np.int64)), "z round trip failed"

    def scales_from(zh):
        s = m.h_s(zh)
        if s.shape[-2:] != y.shape[-2:]:
            s = F.interpolate(s, size=y.shape[-2:], mode="nearest")
        return s.clamp(min=0.11)
    sc_enc = scales_from(z_hat)
    sc_dec = scales_from(torch.from_numpy(z_dec.astype(np.float32)))      # decoder recomputes from decoded z
    assert torch.equal(sc_enc, sc_dec), "decoder-side scales differ from encoder-side scales"
    y_hat = torch.round(y)
    yb, y_dec, yr = rng_bytes(y_hat.numpy().astype(np.int64), np.zeros(y_hat.shape), sc_dec.numpy())
    assert np.array_equal(y_dec, y_hat.numpy().astype(np.int64)), "y round trip failed"
    zf, zd2, _ = rng_bytes(z_hat.numpy().astype(np.int64), zmean.numpy(), zscale.numpy(), fixed=FIXED)
    yf, yd2, _ = rng_bytes(y_hat.numpy().astype(np.int64), np.zeros(y_hat.shape), sc_dec.numpy(), fixed=FIXED)
    assert np.array_equal(zd2, z_dec) and np.array_equal(yd2, y_dec), "hyperprior fixed-support round trip failed"
    est_bits = float(m._gauss_bits(y_hat, sc_enc) + m._gauss_bits(z_hat, zscale, zmean))
    x_hat = m.g_s(torch.from_numpy(y_dec.astype(np.float32))).clamp(0, 1)
    return zb + yb, est_bits / 8.0, x_hat[0], {"fixed_support_bytes": zf + yf, "z_bytes": zb, "y_bytes": yb, "z_range": zr, "y_range": yr}


def jpeg_bytes(pil, q):
    b = io.BytesIO(); pil.save(b, format="JPEG", quality=q); d = b.getvalue()
    return len(d), np.array(Image.open(io.BytesIO(d)).convert("RGB"))


def jp2_bytes(pil, ratio):
    b = io.BytesIO(); pil.save(b, format="JPEG2000", quality_mode="rates", quality_layers=[ratio]); d = b.getvalue()
    return len(d), np.array(Image.open(io.BytesIO(d)).convert("RGB"))


def summarize(rows, keys):
    return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys} | \
           {k + "_sd": round(float(np.std([r[k] for r in rows])), 4) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--fcodec", default=os.path.join("models", "learned_codec_l0.01.pth"))
    ap.add_argument("--hcodec", default=os.path.join("models", "hyperprior_l0.01.pth"))
    ap.add_argument("--out", default=os.path.join("results", "actual_bitstream_results.json"))
    a = ap.parse_args()
    torch.set_num_threads(1)
    ds = datasets.ImageFolder(a.data, transform=TF); n = len(ds)
    perm = np.random.default_rng(42).permutation(n); val_all = perm[:int(n * 0.2)]
    fset = np.random.default_rng(43).choice(val_all, size=min(a.n, len(val_all)), replace=False).tolist()
    idx = list(range(n)); random.Random(43).shuffle(idx); hset = idx[:a.n]
    fm = FactorizedCodec(); fm.load_state_dict(torch.load(a.fcodec, map_location="cpu", weights_only=True)); fm.eval()
    hm = ScaleHyperprior(); hm.load_state_dict(torch.load(a.hcodec, map_location="cpu", weights_only=True)); hm.eval()
    JPEG_Q = [10, 20, 30, 50]; JP2_R = [80, 40, 20, 10]
    res = {"units": "bytes (1 KB = 1,024 bytes)", "n_images": n,
           "sets": {"F": {"definition": "train_compression_v2.py eval set (held out from the factorized codec)", "size": len(fset)},
                    "H": {"definition": "train_hyperprior_pure.py eval set (held out from the hyperprior)", "size": len(hset)}},
           "overlap_F_H": len(set(fset) & set(hset)), "codecs": {}}
    t0 = time.time()
    for set_name, ids, codec_name in (("F", fset, "factorized_l0.01"), ("H", hset, "hyperprior_l0.01")):
        rows = []
        for k, i in enumerate(ids):
            x, _ = ds[int(i)]; xb = x.unsqueeze(0)
            if codec_name.startswith("factorized"):
                nb, est, xh, extra = code_factorized(fm, xb)
            else:
                nb, est, xh, extra = code_hyperprior(hm, xb)
            p, s = psnr_ssim(x, xh)
            o8 = (x.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8); pil = Image.fromarray(o8)
            row = {"actual_bytes": nb, "actual_bytes_fixed_support": extra["fixed_support_bytes"], "estimate_bytes": est,
                   "psnr": p, "ssim": s}
            for q in JPEG_Q:
                jb, rec = jpeg_bytes(pil, q); jp, js = rgb_metrics(o8, rec)
                row |= {f"jpeg{q}_bytes": jb, f"jpeg{q}_psnr": jp, f"jpeg{q}_ssim": js}
            for r in JP2_R:
                jb, rec = jp2_bytes(pil, r); jp, js = rgb_metrics(o8, rec)
                row |= {f"jp2_{r}_bytes": jb, f"jp2_{r}_psnr": jp, f"jp2_{r}_ssim": js}
            rows.append(row)
            if (k + 1) % 50 == 0:
                print(f"  {codec_name}: {k+1}/{len(ids)}  {time.time()-t0:.0f}s", flush=True)
        keys = list(rows[0].keys())
        res["codecs"][codec_name] = {"eval_set": set_name, "all_round_trips_lossless": True,
                                     "mean_actual_over_estimate": round(float(np.mean([r['actual_bytes'] / r['estimate_bytes'] for r in rows])), 4),
                                     **summarize(rows, keys)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: {kk: v[kk] for kk in ('actual_bytes', 'actual_bytes_fixed_support', 'estimate_bytes', 'psnr', 'ssim', 'mean_actual_over_estimate')} for k, v in res["codecs"].items()}, indent=1))
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
