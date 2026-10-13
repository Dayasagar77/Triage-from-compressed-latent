"""
make_reconstruction_figure_heldout.py  (29 Sep 2026, Reviewer-4 revision)
Same as make_reconstruction_figure.py -- the same trained FactorizedCodec (lambda = 0.01), the same pure-NumPy forward
pass (GDN analysis -> round -> GDN synthesis), the same PSNR/SSIM/entropy-bytes computation and the same layout --
with ONE change: the two images are taken from the codec's held-out evaluation set instead of the first file of each
class folder. The original script used the first Adenosis and the first Ductal Carcinoma file, and both of those
images are in the codec's TRAINING split (train_compression_v2.py: numpy default_rng(42) permutation, 20% held out).
Here the images are the first Adenosis and the first Ductal Carcinoma image, in file order, of the 400-image
evaluation set used for the rate-distortion results (default_rng(43) sample of the held-out 20%). Nothing is tuned or
hand-picked. Writes results/figures/fig_reconstruction_heldout.png and results/fig_reconstruction_heldout.json.

    python make_reconstruction_figure_heldout.py "<BreakHis_8class>"
"""
import zipfile, io, pickle, os, json, sys
import numpy as np
from PIL import Image
from scipy.stats import norm as _norm
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "models", "learned_codec_l0.01.pth")
DATA = os.environ.get("BREAKHIS_DIR", sys.argv[1] if len(sys.argv) > 1 else "BreakHis_8class")
OUT = os.path.join(HERE, "results", "figures", "fig_reconstruction_heldout.png")
OUTJ = os.path.join(HERE, "results", "fig_reconstruction_heldout.json")

def load_state_dict(path):
    z = zipfile.ZipFile(path)
    base = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    pref = base[:-len("data.pkl")]
    def rebuild(storage, off, size, stride, *r):
        size = tuple(size); n = 1
        for s in size: n *= s
        return storage[off:off+n].reshape(size).copy()
    class U(pickle.Unpickler):
        def find_class(self, m, n):
            if m == "torch._utils" and n.startswith("_rebuild_tensor"): return rebuild
            if m == "torch" and n.endswith("Storage"): return n
            try: return super().find_class(m, n)
            except Exception: return lambda *a, **k: None
        def persistent_load(self, pid):
            _, st, key, loc, numel = pid
            raw = z.read(pref + "data/" + str(key))
            dt = np.float32 if "Float" in str(st) else np.int64
            return np.frombuffer(raw, dtype=dt)
    return U(io.BytesIO(z.read(base))).load()

sd = {k: np.asarray(v, dtype=np.float64) for k, v in load_state_dict(WEIGHTS).items()}

def conv2d(x, W, b, stride=2, pad=2):
    C, H, Wd = x.shape
    xp = np.pad(x, ((0,0),(pad,pad),(pad,pad)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (5,5), axis=(1,2))
    win = win[:, ::stride, ::stride, :, :]
    oH, oW = win.shape[1], win.shape[2]
    cols = win.transpose(1,2,0,3,4).reshape(oH*oW, C*25)
    out = cols @ W.reshape(W.shape[0], C*25).T + b
    return out.reshape(oH, oW, -1).transpose(2,0,1)

def deconv2d(x, W, b, stride=2, pad=2, outpad=1):
    inC, H, Wd = x.shape
    outC = W.shape[1]
    Hf = (H-1)*stride + 5
    Wf = (Wd-1)*stride + 5
    out = np.zeros((outC, Hf, Wf))
    Wr = W.reshape(inC, outC*25)
    for i in range(H):
        for j in range(Wd):
            patch = (x[:, i, j] @ Wr).reshape(outC, 5, 5)
            out[:, i*stride:i*stride+5, j*stride:j*stride+5] += patch
    out = out[:, pad:Hf-pad+outpad, pad:Wf-pad+outpad]
    return out + b[:, None, None]

def gdn(x, gamma, beta, inverse):
    g = np.abs(gamma); bb = np.abs(beta)
    val = np.tensordot(g, x*x, axes=([1],[0])) + bb[:, None, None]
    n = np.sqrt(val + 1e-6)
    return x*n if inverse else x/n

def g_a(x):
    x = conv2d(x, sd["g_a.0.weight"], sd["g_a.0.bias"]); x = gdn(x, sd["g_a.1.gamma"], sd["g_a.1.beta"], False)
    x = conv2d(x, sd["g_a.2.weight"], sd["g_a.2.bias"]); x = gdn(x, sd["g_a.3.gamma"], sd["g_a.3.beta"], False)
    x = conv2d(x, sd["g_a.4.weight"], sd["g_a.4.bias"]); x = gdn(x, sd["g_a.5.gamma"], sd["g_a.5.beta"], False)
    return conv2d(x, sd["g_a.6.weight"], sd["g_a.6.bias"])

def g_s(y):
    x = deconv2d(y, sd["g_s.0.weight"], sd["g_s.0.bias"]); x = gdn(x, sd["g_s.1.gamma"], sd["g_s.1.beta"], True)
    x = deconv2d(x, sd["g_s.2.weight"], sd["g_s.2.bias"]); x = gdn(x, sd["g_s.3.gamma"], sd["g_s.3.beta"], True)
    x = deconv2d(x, sd["g_s.4.weight"], sd["g_s.4.bias"]); x = gdn(x, sd["g_s.5.gamma"], sd["g_s.5.beta"], True)
    x = deconv2d(x, sd["g_s.6.weight"], sd["g_s.6.bias"])
    return 1.0/(1.0+np.exp(-x))

def rate_bytes(y_hat):
    mean = sd["prior_mean"][:, None, None]
    scale = np.log1p(np.exp(sd["prior_logscale"]))
    scale = np.clip(scale, 0.11, None)[:, None, None]
    hi = _norm.cdf((y_hat + 0.5 - mean)/scale); lo = _norm.cdf((y_hat - 0.5 - mean)/scale)
    p = np.clip(hi - lo, 1e-9, None)
    return np.sum(-np.log2(p))/8.0

def load_img(path):
    im = Image.open(path).convert("RGB").resize((224,224), Image.BILINEAR)
    return np.asarray(im, dtype=np.float64).transpose(2,0,1)/255.0

def reconstruct(path):
    x = load_img(path)
    y = np.round(g_a(x))
    xr = np.clip(g_s(y), 0, 1)
    o8 = (x.transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
    r8 = (xr.transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
    mse = np.mean((o8.astype(float)-r8.astype(float))**2)
    psnr = 99.0 if mse == 0 else 10*np.log10(255.0**2/mse)
    og = np.asarray(Image.fromarray(o8).convert("L")); rg = np.asarray(Image.fromarray(r8).convert("L"))
    return o8, r8, psnr, float(ssim(og, rg)), rate_bytes(y)/1024.0

# ---- the codec's held-out evaluation set, in the ImageFolder order used by train_compression_v2.py ----
classes = sorted(d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d)))
files = [(c, f) for c in classes for f in sorted(os.listdir(os.path.join(DATA, c)))]
n = len(files)
perm = np.random.default_rng(42).permutation(n); val_all = perm[:int(n * 0.2)]
eval_idx = set(np.random.default_rng(43).choice(val_all, size=min(400, len(val_all)), replace=False).tolist())
pick = {}
for cls in ("Adenosis", "Carcinoma"):
    pick[cls] = next(i for i, (c, f) in enumerate(files) if c == cls and i in eval_idx)
rows = [("Benign (Adenosis)", pick["Adenosis"]), ("Malignant (Ductal Carcinoma)", pick["Carcinoma"])]

CROP = (84, 84, 56)
def add_zoom(a, img, cr):
    y0, x0, sz = cr
    axins = a.inset_axes([0.60, 0.03, 0.37, 0.37])
    axins.imshow(img[y0:y0+sz, x0:x0+sz]); axins.set_xticks([]); axins.set_yticks([])
    for sp in axins.spines.values(): sp.set_edgecolor("yellow"); sp.set_linewidth(1.6)
    a.add_patch(mpatches.Rectangle((x0, y0), sz, sz, fill=False, edgecolor="yellow", linewidth=1.3))

record = {"codec": "FactorizedCodec lambda=0.01 (models/learned_codec_l0.01.pth)", "selection":
          "first Adenosis and first Ductal Carcinoma image, in file order, of the codec's 400-image held-out evaluation set",
          "kb_is": "entropy-model estimate of the latent (as in make_reconstruction_figure.py)", "images": []}
fig, ax = plt.subplots(2, 2, figsize=(6.8, 7.2))
for r, (label, i) in enumerate(rows):
    c, f = files[i]; path = os.path.join(DATA, c, f)
    o8, r8, psnr, s, kb = reconstruct(path)
    ax[r,0].imshow(o8); ax[r,0].set_title(f"{label}\nOriginal (224x224)", fontsize=10); add_zoom(ax[r,0], o8, CROP)
    ax[r,1].imshow(r8); ax[r,1].set_title(f"Reconstruction ~{kb:.1f} KB\nPSNR {psnr:.2f} dB   SSIM {s:.3f}", fontsize=10); add_zoom(ax[r,1], r8, CROP)
    for cc in (0, 1): ax[r,cc].set_xticks([]); ax[r,cc].set_yticks([])
    record["images"].append({"label": label, "file": f, "index": i, "in_codec_training_split": False,
                             "kb_estimate": round(kb, 3), "psnr_db": round(psnr, 2), "ssim": round(s, 4)})
    print(f"{label}: {f}  {kb:.2f} KB  PSNR {psnr:.2f}  SSIM {s:.3f}")
fig.text(0.5, 0.005, "Yellow inset: the boxed 56 x 56-pixel region, enlarged about 1.5x (nuclear boundaries and chromatin detail).", ha="center", fontsize=8)
plt.tight_layout(rect=[0, 0.02, 1, 1])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
json.dump(record, open(OUTJ, "w"), indent=1)
print("saved", OUT, "and", OUTJ)
