import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
cc=json.load(open("results/compression_comparison.json"))
lc=json.load(open("results/learned_codec_results.json"))["points"]
hp=json.load(open("results/hyperprior_results.json"))["points"]
def curve(d): return sorted((v["avg_bytes"]/1024, v["avg_ssim"]) for v in d.values())
fig,ax=plt.subplots(figsize=(6.4,4.4))
for name,d,style in [("JPEG",cc["jpeg"],"s-"),("JPEG2000",cc["jpeg2000"],"^-"),
                     ("Factorized-prior codec (lightweight)",lc,"D-"),
                     ("Scale-hyperprior (ours, trained)",hp,"o-")]:
    pts=curve(d); xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    ax.plot(xs,ys,style,label=name,linewidth=1.8,markersize=5)
ax.set_xlabel("Compressed size (KB per 224×224 image)"); ax.set_ylabel("SSIM (reconstruction quality)")
ax.set_title("Rate–distortion at 224×224"); ax.set_xlim(0,16); ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/figures/fig_compression_rd.png",dpi=160,bbox_inches="tight")
print("RD figure regenerated with hyperprior series")
