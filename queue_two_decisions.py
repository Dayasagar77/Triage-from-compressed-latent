"""
queue_two_decisions.py - promote-only priority queue for the TWO triage decisions (Section 5.5);
Reviewer 4, comments 1 and 2.

Model (the same as queue_sim.py): one server, Poisson arrivals, deterministic service D (M/D/1),
non-preemptive two-class priority. A case is promoted (high class) if it is flagged malignant:
truly malignant cases with probability = sensitivity, benign cases with probability
1 - specificity (false-positive promotions). Missed malignant cases wait in the low class.
Baseline: a single FIFO queue fed by the SAME arrival stream. All waits are reported in units
of the service time D, so they hold for any D at the same utilisation rho.

Operating points (all from measured result files):
  transmission : the SOURCE classifier on held-out patients (source_casewise_results.json,
                 field_argmax: tp 8410, fn 740, tn 3557, fp 958) -> sens 0.9191, spec 0.7878,
                 malignant fraction 9150/13665.  SIMULATED here (seed 20261013) and solved
                 analytically (Cobham).
  review       : the LATENT classifier, per field (latent_confusion.json, image-wise codec;
                 sens 0.8919, spec 0.6707, malignant fraction 0.664 as in queue_sim.py). Its
                 simulation is queue_sim_results.json; only the analytic solution is added here.
  review_pt    : the LATENT classifier, leakage-free, per patient (patient_level_intervals.json:
                 43/47, 18/21; malignant fraction 47/68). Analytic only.

    python queue_two_decisions.py --results results
Output: results/queue_two_decisions.json
"""
import argparse, collections, heapq, json, os, random

RHOS = (0.3, 0.5, 0.7, 0.9)


def cobham(sens, spec, p_mal, rho):
    """Waits in units of D for the non-preemptive M/D/1 two-class priority queue."""
    f_high = p_mal * sens + (1 - p_mal) * (1 - spec)
    w0 = rho / 2.0                       # residual work, D = 1
    wh = w0 / (1 - f_high * rho)
    wl = w0 / ((1 - f_high * rho) * (1 - rho))
    wf = w0 / (1 - rho)
    wm = sens * wh + (1 - sens) * wl
    return dict(frac_promoted=f_high, W_promoted=wh, W_missed=wl, W_fifo=wf,
                malignant_reduction_pct=100 * (1 - wm / wf), missed_over_fifo=wl / wf)


def simulate(sens, spec, p_mal, rho, n=400_000, warm=20_000, rng=None):
    """Discrete-event simulation with D = 1; returns mean waits after a warm-up."""
    lam = rho
    t, jobs = 0.0, []
    for _ in range(n):
        t += rng.expovariate(lam)
        mal = rng.random() < p_mal
        prom = (rng.random() < sens) if mal else (rng.random() < 1 - spec)
        jobs.append((t, prom, mal))
    # priority server
    waits = [None] * n
    qh, ql = collections.deque(), collections.deque()
    free_at, i = 0.0, 0
    while i < n or qh or ql:
        if (qh or ql) and (i >= n or free_at <= jobs[i][0]):
            k = qh.popleft() if qh else ql.popleft()
            start = max(free_at, jobs[k][0])
            waits[k] = start - jobs[k][0]
            free_at = start + 1.0
        else:
            (qh if jobs[i][1] else ql).append(i)
            i += 1
    # FIFO on the same arrivals
    fifo, free = [], 0.0
    for (a, _p, _m) in jobs:
        s = max(a, free); fifo.append(s - a); free = s + 1.0
    idx = range(warm, n)
    mean = lambda xs: sum(xs) / len(xs)
    w_prom = mean([waits[k] for k in idx if jobs[k][1]])
    w_low = mean([waits[k] for k in idx if not jobs[k][1]])
    w_missed = mean([waits[k] for k in idx if jobs[k][2] and not jobs[k][1]])
    w_mal = mean([waits[k] for k in idx if jobs[k][2]])
    w_fifo = mean([fifo[k] for k in idx])
    w_fifo_mal = mean([fifo[k] for k in idx if jobs[k][2]])
    return dict(W_promoted=w_prom, W_low=w_low, W_missed=w_missed, W_fifo=w_fifo,
                malignant_reduction_pct=100 * (1 - w_mal / w_fifo_mal), missed_over_fifo=w_missed / w_fifo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    pl = json.load(open(os.path.join(a.results, "patient_level_intervals.json")))
    pts = {
        "transmission_source_classifier": (8410 / 9150, 3557 / 4515, 9150 / 13665),
        "review_latent_per_field_imagewise": (0.8919, 0.6707, 0.664),
        "review_latent_per_patient_leakfree": (pl["tp"] / pl["n_malignant"], pl["tn"] / pl["n_benign"],
                                               pl["n_malignant"] / pl["n_patients"]),
    }
    rng = random.Random(20261013)
    out = {"units": "waits in units of the deterministic service time D", "seed": 20261013, "points": {}}
    for name, (s, sp, p) in pts.items():
        rows = []
        for rho in RHOS:
            row = {"rho": rho, "analytic": {k: round(v, 4) for k, v in cobham(s, sp, p, rho).items()}}
            if name.startswith("transmission"):
                row["simulated"] = {k: round(v, 4) for k, v in simulate(s, sp, p, rho, rng=rng).items()}
            rows.append(row)
        out["points"][name] = {"sensitivity": round(s, 4), "specificity": round(sp, 4),
                               "malignant_fraction": round(p, 4), "rows": rows}
    json.dump(out, open(os.path.join(a.results, "queue_two_decisions.json"), "w"), indent=2)
    for name, v in out["points"].items():
        print(name, v["sensitivity"], v["specificity"], v["malignant_fraction"])
        for r in v["rows"]:
            print("  rho", r["rho"], "analytic", r["analytic"], "\n          sim", r.get("simulated"))


if __name__ == "__main__":
    main()
