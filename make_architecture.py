"""Regenerate Figure 1 (system architecture) with a corrected receiver-side flow:
network -> (5) triage from latent (immediate, no decode) -> (6) reconstruct when needed -> (7) specialist review.
Removes the earlier incorrect 'reconstruct -> classify-from-latent' arrow."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
fig, ax = plt.subplots(figsize=(13.2, 6.2)); ax.set_xlim(0, 16); ax.set_ylim(0, 8); ax.axis("off")

def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.14",
                                fc=fc, ec="#333", lw=1.5))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=11, color="#111", fontweight="bold")
def arrow(x1, y1, x2, y2, style="-|>", color="#333", lw=2.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=18,
                                 color=color, lw=lw, linestyle=ls, shrinkA=2, shrinkB=2))

# headers
ax.text(2.15, 7.6, "Rural clinic (source)", ha="center", fontsize=13, color="#1f5fb0", fontweight="bold")
ax.text(13.85, 7.6, "Diagnostic centre (receiver)", ha="center", fontsize=13, color="#1a8a3a", fontweight="bold")

# left column (source)
box(0.35, 5.95, 3.6, 1.05, "1. Biopsy image\ncapture", "#a6cee3")
box(0.35, 4.35, 3.6, 1.15, "2. AI classifier\n(8 subtypes → benign/\nmalignant triage)", "#cbb8e0")
box(0.35, 2.75, 3.6, 1.15, "3. Adaptive neural\ncompression\n(bandwidth-aware)", "#fdbf6f")
box(0.35, 1.05, 3.6, 1.25, "4. Priority queue\n(malignant first) +\nDTN store-and-forward", "#b2df8a")

# middle (network)
ax.text(7.75, 5.35, "compressed bundles", ha="center", fontsize=10, style="italic", color="#666")
box(6.15, 3.3, 3.2, 1.55, "Intermittent network\n(store-carry-forward;\noutages tolerated)", "#ededed")

# right column (receiver) — corrected order/flow: triage -> reconstruct -> review
box(12.0, 5.95, 3.65, 1.05, "5. Triage from latent\n(immediate, no decode)", "#cbb8e0")
box(12.0, 4.15, 3.65, 1.25, "6. Reconstruct full image\n(neural decoder,\nwhen a full read is needed)", "#ffd6cc")
box(12.0, 2.55, 3.65, 1.05, "7. Specialist review\n& diagnosis", "#a6cee3")

# arrows — source chain
arrow(2.15, 5.95, 2.15, 5.52)
arrow(2.15, 4.35, 2.15, 3.92)
arrow(2.15, 2.75, 2.15, 2.32)
# source -> network
arrow(3.95, 1.65, 6.15, 3.55)
# network -> triage (5): the code is read directly for immediate triage
arrow(9.35, 4.25, 12.0, 6.2)
# triage (5) -> reconstruct (6): reconstruct only when a full read is needed
arrow(13.82, 5.95, 13.82, 5.42)
# reconstruct (6) -> specialist review (7)
arrow(13.82, 4.15, 13.82, 3.62)
# triage (5) also prioritises the specialist queue (dashed, down the right side)
arrow(15.72, 6.35, 15.72, 3.1, color="#888", lw=1.6, ls=(0,(4,3)))
ax.text(15.9, 4.7, "triage\nprioritises\nreview", ha="left", va="center", fontsize=7.5, color="#888")

plt.tight_layout()
os.makedirs("results/paper_assets", exist_ok=True)
plt.savefig("results/paper_assets/fig_architecture.png", dpi=250, bbox_inches="tight")
print("regenerated fig_architecture.png with corrected receiver flow")
