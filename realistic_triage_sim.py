"""realistic_triage_sim.py — realistic malignant delivery time under PROMOTE-ONLY
scheduling driven by the (imperfect) latent triage, instead of ground-truth labels.

Derivation from the two MEASURED oracle endpoints (results/triage_results.json):
  * FIFO (no triage / a missed malignant case): mean malignant delivery = t_fifo
  * Oracle priority (every malignant promoted):  mean malignant delivery = t_prio
Under promote-only with latent malignant sensitivity r (= fraction of malignant cases
the latent stage correctly flags), a promoted case gets t_prio and a MISSED malignant
case stays FIFO (t_fifo). Expected malignant delivery:
      E[t](r) = r * t_prio + (1 - r) * t_fifo
Net speed-up vs FIFO = (t_fifo - E[t]) / t_fifo = r * (t_fifo - t_prio)/t_fifo = r * S_oracle
i.e. the realistic speed-up scales linearly with latent sensitivity; r = 1 recovers the
oracle 31.8%. r itself must be read off the latent binary confusion matrix (not yet
computed — requires re-running the latent classifier), so we report the relation + a sweep."""
import json
tr=json.load(open("results/triage_results.json"))
t_fifo=tr["malignant_delivery_time_fifo_s"]["mean"]
t_prio=tr["malignant_delivery_time_priority_s"]["mean"]
S=(t_fifo-t_prio)/t_fifo
rows=[]
for r in [0.5,0.6,0.7,0.8,0.9,1.0]:
    Et=r*t_prio+(1-r)*t_fifo
    rows.append({"latent_sensitivity":r,"exp_malignant_delivery_s":round(Et,3),
                 "net_speedup_pct":round(100*(t_fifo-Et)/t_fifo,1)})
out={"t_fifo_s":t_fifo,"t_prio_s":t_prio,"oracle_speedup_pct":round(100*S,1),
     "model":"E[t](r)=r*t_prio+(1-r)*t_fifo ; net_speedup = r * oracle_speedup",
     "note":"r = latent malignant sensitivity; exact value needs the latent binary confusion matrix (future work).",
     "sweep":rows}
json.dump(out,open("results/realistic_triage_results.json","w"),indent=2)
print("oracle (r=1) speedup:",out["oracle_speedup_pct"],"%")
print(f"{'sensitivity r':>13} {'E[malignant delivery] s':>24} {'net speedup %':>14}")
for x in rows: print(f"{x['latent_sensitivity']:>13} {x['exp_malignant_delivery_s']:>24} {x['net_speedup_pct']:>14}")
