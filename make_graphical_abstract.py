import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")

def box(x, y, w, h, text, fc, ec="#333"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                 fc=fc, ec=ec, lw=1.3))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=9.5, color="#111")

def arrow(x1,y1,x2,y2,style="-|>",color="#444",lw=1.8):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=16,color=color,lw=lw))

# Title
ax.text(5.5, 4.35, "Triage from the Compressed Latent — Telepathology for Bandwidth-Constrained Networks",
        ha="center", va="center", fontsize=12.5, fontweight="bold", color="#1a1a1a")

# Source cluster
ax.text(1.85, 3.75, "Rural clinic (source)", ha="center", fontsize=9.5, color="#2b5", fontweight="bold")
box(0.35, 2.55, 3.0, 0.62, "Biopsy → ResNet-50 + SE\nclassify (8 subtypes) & triage", "#cfe8d6")
box(0.35, 1.7, 3.0, 0.62, "Learned codec\ncompress ~5 KB (diagnosable latent)", "#ffe0b3")
box(0.35, 0.85, 3.0, 0.62, "Priority queue\n(malignant first) → DTN", "#d9e4ff")

# Network
box(4.05, 1.7, 2.4, 0.9, "Intermittent network\nstore–carry–forward\n(outages tolerated)", "#ededed")
ax.text(5.25, 1.4, "compressed bundles", ha="center", fontsize=7.5, style="italic", color="#666")

# Receiver cluster
ax.text(9.1, 3.75, "Diagnostic centre (receiver)", ha="center", fontsize=9.5, color="#c60", fontweight="bold")
box(7.6, 2.55, 3.0, 0.62, "Reconstruct image\n(neural decoder)", "#ffd6cc")
box(7.6, 1.7, 3.0, 0.62, "Diagnose from latent\n(immediate triage, no decode)", "#e4d6ff")
box(7.6, 0.85, 3.0, 0.62, "Specialist review\n& diagnosis", "#cfe0f5")

# arrows
arrow(1.85,2.55,1.85,2.32); arrow(1.85,1.7,1.85,1.47)
arrow(3.35,1.15,4.05,1.9)
arrow(6.45,1.95,7.6,2.7); arrow(6.45,2.0,7.6,2.0)
arrow(9.1,2.55,9.1,2.32); arrow(9.1,1.7,9.1,1.47)

# key results strip
ax.add_patch(FancyBboxPatch((0.35,0.05),10.25,0.6,boxstyle="round,pad=0.03,rounding_size=0.1",
             fc="#f7f7f2", ec="#bbb", lw=1))
ax.text(5.5,0.35,
 "Patient-wise 5-fold  •  87.6% benign/malignant (AUC 0.94)  •  82.3% from the 5 KB latent alone  •  "
 "100% DTN delivery where TCP/IP fails  •  every number reproducible",
 ha="center", va="center", fontsize=8.8, color="#222")

plt.tight_layout()
plt.savefig("results/paper_assets/graphical_abstract.png", dpi=170, bbox_inches="tight")
print("saved results/paper_assets/graphical_abstract.png")
