"""
patient_level_intervals.py - 95% intervals for the leakage-free patient-level latent triage
(Section 5.3); Reviewer 4, comment 9.

Input: results/latent_leakagefree_results.json (r5_codec_perfold.py), which stores the
patient-level rates rounded to 4 decimals and n_patients = 68, but not the counts.

1. The counts are recovered by exhaustive search: every (n_malignant, tp, tn) with
   n_malignant + n_benign = 68 whose rates round to the stored sens, spec and acc. The search
   must return exactly one solution, otherwise the script stops.
2. The AUC numerator is recovered the same way: AUC = k / (n_mal * n_ben), with k a multiple
   of 0.5 (Mann-Whitney with ties counted half, as sklearn's roc_auc_score), whose value rounds
   to the stored AUC. Exactly one k must match.
3. Intervals: Wilson 95% for sensitivity and specificity; Hanley-McNeil (1982) standard error
   for the AUC, with the 95% interval formed on the logit scale (keeps it inside [0, 1]).

    python patient_level_intervals.py --results results
Output: results/patient_level_intervals.json
"""
import argparse, json, math, os
from statistics import NormalDist


def wilson(k, n, z):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    d = json.load(open(os.path.join(a.results, "latent_leakagefree_results.json")))
    N, pl = d["n_patients"], d["patient_level"]
    sols = [(npos, tp, tn) for npos in range(1, N) for tp in range(npos + 1)
            if round(tp / npos, 4) == pl["sens"]
            for tn in range(N - npos + 1)
            if round(tn / (N - npos), 4) == pl["spec"] and round((tp + tn) / N, 4) == pl["acc"]]
    assert len(sols) == 1, f"counts not unique: {sols}"
    npos, tp, tn = sols[0]; nneg = N - npos; P = npos * nneg
    ks = [k / 2 for k in range(2 * P + 1) if round((k / 2) / P, 4) == pl["auc"]]
    assert len(ks) == 1, f"AUC numerator not unique: {ks}"
    A = ks[0] / P
    z = NormalDist().inv_cdf(0.975)
    q1, q2 = A / (2 - A), 2 * A * A / (1 + A)
    se = math.sqrt((A * (1 - A) + (npos - 1) * (q1 - A * A) + (nneg - 1) * (q2 - A * A)) / P)
    lg, sl = math.log(A / (1 - A)), se / (A * (1 - A))
    auc_ci = (1 / (1 + math.exp(-(lg - z * sl))), 1 / (1 + math.exp(-(lg + z * sl))))
    out = {"n_patients": N, "n_malignant": npos, "n_benign": nneg, "tp": tp, "tn": tn,
           "sensitivity": tp / npos, "sensitivity_wilson95": wilson(tp, npos, z),
           "specificity": tn / nneg, "specificity_wilson95": wilson(tn, nneg, z),
           "auc": A, "auc_numerator_pairs": ks[0], "auc_pairs": P, "auc_se_hanley_mcneil": se,
           "auc_ci95_logit": auc_ci}
    json.dump(out, open(os.path.join(a.results, "patient_level_intervals.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
