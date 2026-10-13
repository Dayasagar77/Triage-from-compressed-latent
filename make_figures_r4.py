"""make_figures_r4.py (29 Sep 2026, Reviewer-4 revision): one script for every plotted data figure in the paper.

Why this script exists
  Figures 3, 4, 9, 10 and 11 of the first submission were drawn by versions of make_figures.py, diagnose_vs_bytes.py
  and tau_sweep.py whose plot settings (figure size, font size, legend position, value labels) were changed after the
  saved copies. The saved scripts therefore draw slightly different pictures from the same numbers. The settings below
  were recovered by redrawing each figure and comparing it pixel by pixel with the image in the submitted manuscript;
  `--check` repeats that comparison. The numbers plotted come only from the result files listed per figure.

  Paper figure  function            data (results/)                                   submitted image (results/figures/)
  Figure 3      fig_per_mag         kfold_results_{40,100,200,400}X.json              fig_per_magnification.png
  Figure 4      fig_confusion       confusion_matrix_40X.csv                          fig_confusion_40X.png
  Figure 5      fig_baseline        kfold_results.json, kfold_results_baseline.json   fig_baseline_vs_proposed.png
  Figure 6      fig_rd              compression_comparison.json, learned_codec_results.json,
                                    hyperprior_results.json                           fig_compression_rd.png
  Figure 9      fig_diagnose        diagnose_vs_bytes.json, kfold_results.json        fig_diagnose_vs_bytes.png
  Figure 10     fig_overhead        network_results.json                              fig_protocol_overhead.png
  Figure 11     fig_tau             tau_sweep.json                                    fig_tau_sweep.png
  (Figure 1: make_architecture.py. Figure 7: make_reconstruction_figure_heldout.py. Figure 8: latent_umap.py.
   Figure 2 shows two BreakHis images, SOB_B_A-14-22549AB-100-001.png and SOB_M_DC-14-10926-100-001.png.)

Usage (from the repository folder)
  python make_figures_r4.py --check      draw every figure exactly as submitted and compare it pixel by pixel with
                                         results/figures/<name>.png; prints IDENTICAL or the number of differing pixels
  python make_figures_r4.py              write the revised figures to results/figures/r4/ (the submitted files are not
                                         touched). Changes from the submitted figures, and nothing else:
                                           Figures 3 and 5: title "patient-wise 5-fold" -> "official patient-wise splits"
                                             (the official BreakHis protocol is five overlapping splits, not five folds);
                                           Figure 9: value labels moved beside the points so the error bars no longer
                                             cross them.
                                         Figures 4, 6, 10 and 11 are written unchanged.
Tested with matplotlib 3.10.9 (the version that drew the submitted figures); another version may shift pixels.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
MAGS = ["40", "100", "200", "400"]


def load(fn):
    with open(os.path.join(R, fn)) as f:
        return json.load(f)


def fig_per_mag(out, revised):
    accs, accs_e, bms, bms_e = [], [], [], []
    for m in MAGS:
        s = load(f"kfold_results_{m}X.json")["summary"]
        accs.append(s["accuracy_mean"] * 100); accs_e.append(s["accuracy_std"] * 100)
        bms.append(s["binary_accuracy_mean"] * 100); bms_e.append(s["binary_accuracy_std"] * 100)
    x = np.arange(len(MAGS)); w = 0.35
    fig, ax1 = plt.subplots(figsize=(7, 4.3))
    ax1.bar(x - w / 2, bms, w, yerr=bms_e, capsize=4, label="Benign/Malignant acc.", color="#2c7fb8")
    ax1.bar(x + w / 2, accs, w, yerr=accs_e, capsize=4, label="8-class acc.", color="#7fcdbb")
    ax1.set_ylabel("Accuracy (%)"); ax1.set_xlabel("Magnification")
    ax1.set_xticks(x); ax1.set_xticklabels([f"{m}X" for m in MAGS]); ax1.set_ylim(0, 106)
    ax1.legend(loc="upper center", ncol=2)
    split = "official patient-wise splits" if revised else "patient-wise 5-fold"
    ax1.set_title(f"Classification accuracy per magnification ({split})")
    for i, v in enumerate(bms): ax1.text(i - w / 2, v + bms_e[i] + 1.5, f"{v:.1f}", ha="center", fontsize=8)
    for i, v in enumerate(accs): ax1.text(i + w / 2, v + accs_e[i] + 1.5, f"{v:.1f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()


def fig_confusion(out, revised):
    with open(os.path.join(R, "confusion_matrix_40X.csv")) as f:
        header = f.readline().strip().split(",")
        cm = np.loadtxt(f, delimiter=",", dtype=int)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn * 100, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(header))); ax.set_yticks(range(len(header)))
    short = [h[:10] for h in header]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(short, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cmn[i, j] * 100:.0f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized %), 40X")
    fig.colorbar(im, fraction=0.046, pad=0.04).set_label("% of true class")
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()


def fig_baseline(out, revised):
    prop, base = load("kfold_results.json"), load("kfold_results_baseline.json")
    labels = ["8-class Acc (%)", "8-class Macro-F1 (x100)", "Benign/Malignant (%)"]

    def vals(d):
        s = d["summary"]; return [s["accuracy_mean"] * 100, s["macro_f1_mean"] * 100, s["binary_accuracy_mean"] * 100]
    pv, bv = vals(prop), vals(base)
    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w / 2, bv, w, label="Baseline CNN", color="#bdbdbd")
    ax.bar(x + w / 2, pv, w, label="Proposed", color="#3182bd")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("Score")
    split = "official patient-wise splits" if revised else "patient-wise 5-fold"
    ax.legend(); ax.set_title(f"Baseline vs Proposed (pooled, {split})")
    for i, v in enumerate(bv): ax.text(i - w / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    for i, v in enumerate(pv): ax.text(i + w / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()


def fig_rd(out, revised):
    # JPEG, JPEG2000 and the factorized-prior codec: the same 400 held-out images (set F);
    # scale-hyperprior: its own 400 held-out images (set H). Learned-codec sizes are entropy-model estimates.
    cc = load("compression_comparison.json")
    lc = load("learned_codec_results.json")["points"]
    hp = load("hyperprior_results.json")["points"]

    def curve(d): return sorted((v["avg_bytes"] / 1024, v["avg_ssim"]) for v in d.values())
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, d, style in [("JPEG", cc["jpeg"], "s-"), ("JPEG2000", cc["jpeg2000"], "^-"),
                           ("Factorized-prior codec (lightweight)", lc, "D-"),
                           ("Scale-hyperprior (ours, trained)", hp, "o-")]:
        pts = curve(d); xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, style, label=name, linewidth=1.8, markersize=5)
    ax.set_xlabel("Compressed size (KB per 224×224 image)"); ax.set_ylabel("SSIM (reconstruction quality)")
    ax.set_title("Rate–distortion at 224×224"); ax.set_xlim(0, 16); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=160, bbox_inches="tight"); plt.close()


def fig_diagnose(out, revised):
    # sizes are the factorized-prior codec's entropy-model estimates; accuracy is mean and SD over the five splits
    curve = sorted(load("diagnose_vs_bytes.json")["curve"], key=lambda c: c["kb"])
    xs = [c["kb"] for c in curve]
    ys = [c["binary_acc_mean"] * 100 for c in curve]
    es = [c["binary_acc_std"] * 100 for c in curve]
    full = load("kfold_results.json")["summary"]["binary_accuracy_mean"] * 100
    fig, ax = plt.subplots(figsize=(6.1, 4.15))
    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, linewidth=2, markersize=7, color="#6a3d9a",
                label="Triage from latent")
    ax.axhline(round(full, 1), ls="--", color="gray", label=f"Full-image classifier ({full:.1f}%)")
    for x, y in zip(xs, ys):
        if revised:   # beside the point, clear of the error bar
            ax.annotate(f"{y:.1f}%", (x, y), xytext=(9, -13), textcoords="offset points", ha="left", va="center",
                        fontsize=9)
        else:         # as submitted: centred just above the point (the error bar crosses the label)
            ax.text(x, y + 2.2, f"{y:.1f}%", ha="center", fontsize=9)
    ax.set_xlim(0, 10); ax.set_ylim(50, 100)
    ax.set_xlabel("Transmitted size (KB per image)"); ax.set_ylabel("Benign/malignant accuracy (%)")
    ax.set_title("Triage from the compressed latent is rate-robust"); ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out, dpi=160, bbox_inches="tight"); plt.close()


def fig_overhead(out, revised):
    # application-layer overhead model (Section 3.4) for the earlier 49 KB payload
    pc = load("network_results.json")["protocol_comparison"]
    names = {"TCP_IP": "TCP/IP", "Epidemic": "Epidemic", "PRoPHET": "PRoPHET",
             "Neural_DTN": "Codec-Aware DTN\n(single custody)"}
    keys = ["TCP_IP", "Epidemic", "PRoPHET", "Neural_DTN"]
    oh = [pc[k]["avg_overhead_kb"] for k in keys]
    colors = ["#bdbdbd", "#fdae6b", "#a1d99b", "#3182bd"]
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    bars = ax.bar([names[k] for k in keys], oh, color=colors)
    ax.set_ylabel("Avg overhead per image (KB, log scale)"); ax.set_yscale("log")
    ax.set_title("Protocol transmission overhead (lower is better)")
    for b, v in zip(bars, oh):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v} KB", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()


def fig_tau(out, revised):
    # service time in tau_sweep.json: 49.0 * 8 / 500 = 0.784 s (49,000 bytes at 500 kb/s)
    d = load("tau_sweep.json"); rows = d["rows"]
    fig, ax1 = plt.subplots(figsize=(6.5, 4.3)); ax2 = ax1.twinx()
    t = [r["tau"] for r in rows]
    l1, = ax1.plot(t, [r["sensitivity"] for r in rows], "o-", color="tab:red", label="malignant sensitivity (left)")
    l2, = ax1.plot(t, [r["promoted_fraction"] for r in rows], "s--", color="tab:orange", label="promoted fraction (left)")
    l3, = ax2.plot(t, [r["missed_malignant_delay_s"] for r in rows], "^-", color="tab:blue",
                   label="missed-malignant delay, s (right)")
    ax1.set_xlabel("promotion threshold τ"); ax1.set_ylabel("sensitivity / promoted fraction")
    ax2.set_ylabel("missed-malignant queue delay (s)")
    ax1.set_title(f"τ trade-off at ρ = {d['rho']}")
    ax1.legend(handles=[l1, l2, l3], loc="lower left", fontsize=8.5, framealpha=0.92)
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()


FIGS = [("Figure 3", fig_per_mag, "fig_per_magnification.png"),
        ("Figure 4", fig_confusion, "fig_confusion_40X.png"),
        ("Figure 5", fig_baseline, "fig_baseline_vs_proposed.png"),
        ("Figure 6", fig_rd, "fig_compression_rd.png"),
        ("Figure 9", fig_diagnose, "fig_diagnose_vs_bytes.png"),
        ("Figure 10", fig_overhead, "fig_protocol_overhead.png"),
        ("Figure 11", fig_tau, "fig_tau_sweep.png")]


def pixels(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGBA"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="redraw as submitted and compare with the submitted images")
    ap.add_argument("--submitted", default=os.path.join(R, "figures"), help="folder holding the submitted images")
    a = ap.parse_args()
    if a.check:
        tmp = os.path.join(R, "figures", "_check_as_submitted"); os.makedirs(tmp, exist_ok=True)
        ok = True
        for label, fn, name in FIGS:
            out = os.path.join(tmp, name); fn(out, revised=False)
            A, B = pixels(out), pixels(os.path.join(a.submitted, name))
            if A.shape == B.shape and np.array_equal(A, B):
                print(f"{label:9s} {name:30s} IDENTICAL ({A.shape[1]}x{A.shape[0]} px)")
            else:
                ok = False
                diff = int(np.any(A != B, axis=2).sum()) if A.shape == B.shape else -1
                print(f"{label:9s} {name:30s} DIFFERENT ({'shape ' + str(A.shape) if diff < 0 else str(diff) + ' px'})")
        print("ALL IDENTICAL" if ok else "NOT ALL IDENTICAL")
        return 0 if ok else 1
    outdir = os.path.join(R, "figures", "r4"); os.makedirs(outdir, exist_ok=True)
    for label, fn, name in FIGS:
        out = os.path.join(outdir, name); fn(out, revised=True)
        print(f"{label:9s} wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
