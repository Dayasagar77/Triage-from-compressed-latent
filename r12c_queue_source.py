"""
r12c_queue_source.py - Reviewer-4 comment #1: the E1 (transmission priority) row of the queue table.

Uses the source classifier's saved held-out predictions (r13_source_predictions.py) at its deployed decision rule
(most probable subtype, grouped benign/malignant) and field level, and runs the promote-only queue model of
r12_network_units_fixed.py unchanged (Cobham's formula + discrete-event simulation), exactly as r12b_queue_ops.py
does for the latent classifier.

    python r12c_queue_source.py --source results/source_casewise_results.json --payloads payloads.json --out results/queue_ops_source.json
"""
import argparse, json
from r12_network_units_fixed import queue_table, tx_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True); ap.add_argument("--payloads", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = json.load(open(a.source)); assert not src.get("quick"), "refusing to use a --quick smoke-test result"
    cfg = json.load(open(a.payloads)); D = tx_time(cfg["payloads"][cfg["queue_payload"]])
    c = src["field_argmax"]; n_mal, n_ben = c["tp"] + c["fn"], c["tn"] + c["fp"]
    op = dict(sens=c["tp"] / n_mal, spec=c["tn"] / n_ben, p_mal=n_mal / (n_mal + n_ben), unit=f"field ({n_mal + n_ben:,})",
              counts=dict(tp=c["tp"], fn=c["fn"], fp=c["fp"], tn=c["tn"]),
              source="source_casewise_results.json field_argmax (deployed rule; held-out predictions pooled over the five official splits)")
    q = queue_table(dict(sens=op["sens"], spec=op["spec"]), D, op["p_mal"]); q["operating_point"] = op
    json.dump({"model": "as r12b_queue_ops.py", "service_payload": cfg["queue_payload"], "service_time_s": round(D, 5),
               "operating_points": {"source_field": q}}, open(a.out, "w"), indent=1)
    print("E1:", [(r["rho"], r["malignant_wait_reduction_pct_theory"], r["malignant_wait_reduction_pct_sim"]) for r in q["rows"]])


if __name__ == "__main__":
    main()
