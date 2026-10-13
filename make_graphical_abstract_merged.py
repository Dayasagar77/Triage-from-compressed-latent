"""make_graphical_abstract_merged.py (13 Oct 2026, combined revision) -- the script-drawn graphical abstract.
Based on make_graphical_abstract_r4.py (29 Sep 2026). It shows the paper's exact title, the pipeline from the rural
clinic to the diagnostic centre, the two priority decisions (transmission priority at the clinic, review priority set
by the latent read at the centre), the learned-codec bitstream size (about 5 KB per field) and no performance claims.
Drawn with matplotlib only; no generative image tool.
    python make_graphical_abstract_merged.py   ->  graphical_abstract_merged.png
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(11, 4.4)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.4); ax.axis("off")
def box(x, y, w, h, text, fc, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec="#333", lw=1.3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9.5, color="#111", fontweight="bold" if bold else "normal")
def arrow(x1, y1, x2, y2, color="#444", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, color=color, lw=1.8, linestyle=ls))
ax.text(5.5, 4.15, "Triage from the Compressed Latent:", ha="center", fontsize=13, fontweight="bold", color="#1a1a1a")
ax.text(5.5, 3.8, "Bandwidth-Constrained Telepathology with Delay-Tolerant Delivery", ha="center", fontsize=12, fontweight="bold", color="#1a1a1a")
ax.text(1.85, 3.3, "Rural clinic (source)", ha="center", fontsize=10, color="#1f6b35", fontweight="bold")
box(0.35, 2.35, 3.0, 0.7, "Biopsy field → ResNet-50 + SE\nbenign/malignant flag", "#cfe8d6")
box(0.35, 1.45, 3.0, 0.7, "Learned codec\nbitstream ≈ 5 KB per field", "#ffe0b3")
box(0.35, 0.55, 3.0, 0.7, "Flagged cases sent first\n(transmission priority)", "#d9e4ff")
ax.text(5.5, 3.3, "Intermittent link", ha="center", fontsize=10, color="#444", fontweight="bold")
box(4.1, 1.45, 2.8, 1.0, "Store-and-forward delivery\n(outages tolerated)", "#ededed")
ax.text(9.15, 3.3, "Diagnostic centre (receiver)", ha="center", fontsize=10, color="#9c5a00", fontweight="bold")
box(7.65, 2.35, 3.0, 0.7, "Triage read from the latent,\nno image decode", "#e4d6ff", bold=True)
box(7.65, 1.45, 3.0, 0.7, "Reconstruct the image\nonly when a full read is needed", "#ffd6cc")
box(7.65, 0.55, 3.0, 0.7, "Specialist review\n(review priority set by the latent read)", "#cfe0f5")
arrow(1.85, 2.35, 1.85, 2.15); arrow(1.85, 1.45, 1.85, 1.25)
arrow(3.35, 0.9, 4.1, 1.75); arrow(6.9, 1.95, 7.65, 2.7)
arrow(9.15, 2.35, 9.15, 2.15); arrow(9.15, 1.45, 9.15, 1.25)
ax.text(5.5, 0.2, "Computational study on selected BreakHis fields; clinical benefit not evaluated.", ha="center", fontsize=8.5, style="italic", color="#555")
plt.tight_layout()
plt.savefig("graphical_abstract_merged.png", dpi=170, bbox_inches="tight")
print("saved graphical_abstract_merged.png")
