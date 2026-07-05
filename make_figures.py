"""
make_figures.py — generate publication figures from the saved results/ files.
Run after train.py (--per-mag), train_compression.py, benchmark_network.py, and
(optionally) the baseline run. Figures are written to results/figures/*.png (300 dpi).

    python make_figures.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
FIG = os.path.join(R, "figures")
os.makedirs(FIG, exist_ok=True)
MAGS = ["40", "100", "200", "400"]


def load(fn):
    p = os.path.join(R, fn)
    return json.load(open(p)) if os.path.exists(p) else None


def fig_per_mag():
    accs, accs_e, bms, bms_e, f1s, f1s_e = [], [], [], [], [], []
    for m in MAGS:
        d = load(f"kfold_results_{m}X.json")
        if not d: return
        s = d["summary"]
        accs.append(s["accuracy_mean"]*100); accs_e.append(s["accuracy_std"]*100)
        bms.append(s["binary_accuracy_mean"]*100); bms_e.append(s["binary_accuracy_std"]*100)
        f1s.append(s["macro_f1_mean"]); f1s_e.append(s["macro_f1_std"])
    x = np.arange(len(MAGS)); w = 0.35
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.bar(x-w/2, bms, w, yerr=bms_e, capsize=4, label="Benign/Malignant acc.", color="#2c7fb8")
    ax1.bar(x+w/2, accs, w, yerr=accs_e, capsize=4, label="8-class acc.", color="#7fcdbb")
    ax1.set_ylabel("Accuracy (%)"); ax1.set_xlabel("Magnification")
    ax1.set_xticks(x); ax1.set_xticklabels([f"{m}X" for m in MAGS]); ax1.set_ylim(0, 100)
    ax1.legend(loc="upper right"); ax1.set_title("Classification accuracy per magnification (patient-wise 5-fold)")
    for i,v in enumerate(bms): ax1.text(i-w/2, v+2, f"{v:.1f}", ha="center", fontsize=8)
    for i,v in enumerate(accs): ax1.text(i+w/2, v+2, f"{v:.1f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_per_magnification.png"), dpi=300); plt.close()
    print("wrote fig_per_magnification.png")


def fig_confusion(mag="40"):
    p = os.path.join(R, f"confusion_matrix_{mag}X.csv")
    if not os.path.exists(p): p = os.path.join(R, "confusion_matrix.csv")
    if not os.path.exists(p): return
    with open(p) as f:
        header = f.readline().strip().split(",")
        cm = np.loadtxt(f, delimiter=",", dtype=int)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(header))); ax.set_yticks(range(len(header)))
    short=[h[:10] for h in header]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7); ax.set_yticklabels(short, fontsize=7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cmn[i,j]*100:.0f}", ha="center", va="center",
                    color="white" if cmn[i,j]>0.5 else "black", fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (row-normalized %), {mag}X")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(os.path.join(FIG,f"fig_confusion_{mag}X.png"), dpi=300); plt.close()
    print(f"wrote fig_confusion_{mag}X.png")


def fig_protocol():
    d = load("network_results.json")
    if not d: return
    pc = d["protocol_comparison"]
    names = {"TCP_IP":"TCP/IP","Epidemic":"Epidemic","PRoPHET":"PRoPHET","Neural_DTN":"Codec-Aware DTN\n(single custody)"}
    keys = ["TCP_IP","Epidemic","PRoPHET","Neural_DTN"]
    oh = [pc[k]["avg_overhead_kb"] for k in keys]
    colors = ["#bdbdbd","#fdae6b","#a1d99b","#3182bd"]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    bars = ax.bar([names[k] for k in keys], oh, color=colors)
    ax.set_ylabel("Avg overhead per image (KB, log scale)"); ax.set_yscale("log")
    ax.set_title("Protocol transmission overhead (lower is better)")
    for b,v in zip(bars,oh): ax.text(b.get_x()+b.get_width()/2, v, f"{v} KB", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_protocol_overhead.png"), dpi=300); plt.close()
    print("wrote fig_protocol_overhead.png")


def fig_baseline():
    prop = load("kfold_results.json"); base = load("kfold_results_baseline.json")
    if not (prop and base): 
        print("(skipping baseline figure — run: python train.py ... --model baseline)"); return
    labels = ["8-class Acc (%)","8-class Macro-F1 (x100)","Benign/Malignant (%)"]
    def vals(d):
        s=d["summary"]; return [s["accuracy_mean"]*100, s["macro_f1_mean"]*100, s["binary_accuracy_mean"]*100]
    pv, bv = vals(prop), vals(base)
    x=np.arange(len(labels)); w=0.35
    fig,ax=plt.subplots(figsize=(7,4.2))
    ax.bar(x-w/2, bv, w, label="Baseline CNN", color="#bdbdbd")
    ax.bar(x+w/2, pv, w, label="Proposed", color="#3182bd")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("Score")
    ax.legend(); ax.set_title("Baseline vs Proposed (pooled, patient-wise 5-fold)")
    for i,v in enumerate(bv): ax.text(i-w/2, v+1, f"{v:.1f}", ha="center", fontsize=8)
    for i,v in enumerate(pv): ax.text(i+w/2, v+1, f"{v:.1f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_baseline_vs_proposed.png"), dpi=300); plt.close()
    print("wrote fig_baseline_vs_proposed.png")




def fig_compression_rd():
    d = load("compression_comparison.json")
    if not d:
        print("(skip R-D fig - run compare_compression.py)"); return
    def curve(dct):
        pts = sorted((v["avg_bytes"]/1024, v["avg_ssim"]) for v in dct.values())
        return [x for x, _ in pts], [y for _, y in pts]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name, style, lab in [("jpeg", "-o", "JPEG"), ("jpeg2000", "-s", "JPEG2000"),
                             ("neural", "-^", "Neural v1 (raw latent)")]:
        x, y = curve(d[name]); ax.plot(x, y, style, label=lab, markersize=6)
    lc = load("learned_codec_results.json")   # v2 CompressAI codec, if present
    if lc and lc.get("points"):
        pts = sorted((v["avg_kb"], v["avg_ssim"]) for v in lc["points"].values())
        ax.plot([x for x, _ in pts], [y for _, y in pts], "-D", color="crimson",
                label="Learned codec v2 (proposed)", markersize=7, linewidth=2)
    ax.set_xlabel("Compressed size (KB per 224x224 image)")
    ax.set_ylabel("SSIM (reconstruction quality)")
    ax.set_title("Rate-distortion at 224x224")
    ax.axhline(0.98, ls="--", color="gray", lw=0.8); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_compression_rd.png"), dpi=300); plt.close()
    print("wrote fig_compression_rd.png")


if __name__ == "__main__":
    fig_per_mag(); fig_confusion("40"); fig_protocol(); fig_baseline(); fig_compression_rd()
    print("Figures in", FIG)
