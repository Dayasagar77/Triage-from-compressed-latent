"""make_fig11_service_units.py - Figure 11 redrawn from results/tau_sweep.json with the
missed-malignant delay in units of the service time per case (delay_s / service_s), so the
figure reads as a review-queue model. Same data, markers and colours as tau_sweep.py.
    python make_fig11_service_units.py RESULTS_DIR OUT.png
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
res, out = sys.argv[1], sys.argv[2]
d = json.load(open(os.path.join(res, "tau_sweep.json")))
rows, D = d["rows"], d["service_s"]
t = [r["tau"] for r in rows]
fig, ax1 = plt.subplots(figsize=(6.2, 4.2)); ax2 = ax1.twinx()
l1, = ax1.plot(t, [r["sensitivity"] for r in rows], "o-", color="tab:red", label="malignant sensitivity (left)")
l2, = ax1.plot(t, [r["promoted_fraction"] for r in rows], "s--", color="tab:orange", label="promoted fraction (left)")
l3, = ax2.plot(t, [r["missed_malignant_delay_s"] / D for r in rows], "^-", color="tab:blue",
               label="missed-malignant delay (right)")
ax1.set_xlabel("promotion threshold τ"); ax1.set_ylabel("sensitivity / promoted fraction")
ax2.set_ylabel("missed-malignant delay (service times per case)")
ax1.set_title(f"τ trade-off at ρ = {d['rho']}")
ax1.legend(handles=[l1, l2, l3], loc="lower left", fontsize=8.5, framealpha=0.92)
plt.tight_layout(); plt.savefig(out, dpi=140)
print("saved", out, [round(r["missed_malignant_delay_s"] / D, 2) for r in rows])
