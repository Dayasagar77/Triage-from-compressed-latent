"""
r12b_queue_ops.py - Reviewer-4 comments #1 and #2: promote-only priority queue for each decision.

Uses the queue model of r12_network_units_fixed.py unchanged (non-preemptive M/D/1, two classes,
Poisson arrivals; Cobham's formula cross-checked by a discrete-event simulation with 300,000 jobs),
for these operating points, each taken from COUNTS in a result file (no rounding before use):

  perfect_flags_patient : sens = spec = 1 (upper bound), malignant share of the 68 tested patients
  latent_patient        : receiver latent classifier, patient level (R2: 44 TP, 3 FN, 4 FP, 17 TN)
  perfect_flags_field   : sens = spec = 1, malignant share of the 13,665 held-out field predictions
  latent_field          : receiver latent classifier, field level (R2: 8,144 TP, 1,006 FN, 1,392 FP, 3,123 TN)
  latent_hyperprior_field : receiver latent classifier on the RECOMMENDED codec's (scale-hyperprior) latent,
                          field level (hyperprior_latent_triage.json: 8,042 TP, 1,108 FN, 1,471 FP, 3,044 TN)

Drawing a case's class and promotion from (malignant share, sens, spec) is the same as resampling
the empirical (true class, decision) pairs, because the three numbers are computed from the counts.
The source-classifier operating point (decision E1) is NOT here: it needs the source classifier's
saved held-out predictions (re-run on the GPU server).

Percent changes at a fixed utilisation rho do not depend on the service time D (nor, in Cobham's
formula, on the service-time distribution, as long as both classes share it); D only scales the
absolute waits. D is set to the recommended payload's transmission time for completeness.

    python r12b_queue_ops.py --caselevel results/latent_caselevel_results.json --hyperprior results/hyperprior_latent_triage.json --payloads payloads.json --out results/queue_ops.json
"""
import argparse, json
from r12_network_units_fixed import queue_table, tx_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caselevel", required=True)
    ap.add_argument("--payloads", required=True)
    ap.add_argument("--hyperprior", required=True, help="hyperprior_latent_triage.json (confusion counts)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cl = json.load(open(a.caselevel))
    cfg = json.load(open(a.payloads)); D = tx_time(cfg["payloads"][cfg["queue_payload"]])
    P, F = cl["patient_level"], cl["patch_level"]
    ops = {}
    for tag, c, unit in (("patient", P, "68 tested patients (mean score per patient, threshold 0.5)"),
                         ("field", F, "13,665 held-out field predictions pooled over the five official splits (threshold 0.5)")):
        n_mal, n_ben = c["tp"] + c["fn"], c["tn"] + c["fp"]
        p_mal = n_mal / (n_mal + n_ben)
        ops[f"perfect_flags_{tag}"] = dict(sens=1.0, spec=1.0, p_mal=p_mal,
                                          source=f"upper bound; malignant share of the {unit}")
        ops[f"latent_{tag}"] = dict(sens=c["tp"] / n_mal, spec=c["tn"] / n_ben, p_mal=p_mal,
                                    counts=dict(tp=c["tp"], fn=c["fn"], fp=c["fp"], tn=c["tn"]),
                                    source=f"latent_caselevel_results.json; {unit}")
    hp = json.load(open(a.hyperprior))["confusion"]          # triage from the recommended codec's latent (field level)
    n_mal, n_ben = hp["tp"] + hp["fn"], hp["tn"] + hp["fp"]
    ops["latent_hyperprior_field"] = dict(sens=hp["tp"] / n_mal, spec=hp["tn"] / n_ben, p_mal=n_mal / (n_mal + n_ben),
                                          counts=dict(tp=hp["tp"], fn=hp["fn"], fp=hp["fp"], tn=hp["tn"]),
                                          source="hyperprior_latent_triage.json; held-out field predictions pooled over the five official splits (argmax decision)")
    res = {"model": "promote-only non-preemptive M/D/1 (two classes), Cobham formula + discrete-event simulation (300,000 jobs, first 5% dropped)",
           "service_payload": cfg["queue_payload"], "service_time_s": round(D, 5),
           "note": "percent changes at fixed rho are independent of D; absolute waits scale with D",
           "operating_points": {}}
    for k, op in ops.items():
        q = queue_table(dict(sens=op["sens"], spec=op["spec"]), D, op["p_mal"])
        q["operating_point"] = op
        res["operating_points"][k] = q
        print(k, "promoted", q["rows"][0]["promoted_fraction"],
              [(r["rho"], r["malignant_wait_reduction_pct_theory"], r["malignant_wait_reduction_pct_sim"], r["missed_over_fifo_sim"]) for r in q["rows"]])
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
