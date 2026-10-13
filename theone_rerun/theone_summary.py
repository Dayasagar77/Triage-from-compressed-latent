"""
theone_summary.py - collect The ONE MessageStatsReport files of the Table-12 re-run into one JSON.

    python theone_summary.py --reports the-one/reports --out theone_rerun_results.json
Reads tele_<Router>_seed<k>_MessageStatsReport.txt for every router and seed found, and writes the raw
per-seed values plus the mean, minimum and maximum over seeds. Nothing is edited or filtered.
"""
import argparse, glob, json, os, re, statistics

KEYS = ("created", "started", "relayed", "aborted", "dropped", "removed", "delivered", "delivery_prob",
        "overhead_ratio", "latency_avg", "latency_med", "hopcount_avg")


def parse(path):
    out = {}
    for line in open(path):
        m = re.match(r"(\w+): (\S+)", line.strip())
        if m and m.group(1) in KEYS:
            v = m.group(2); out[m.group(1)] = float(v) if v != "NaN" else None
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reports", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {}
    for p in sorted(glob.glob(os.path.join(a.reports, "tele_*_seed*_MessageStatsReport.txt"))):
        m = re.search(r"tele_(\w+?)_seed(\d+)_MessageStatsReport", os.path.basename(p))
        d = parse(p)
        assert len(d) == len(KEYS), f"incomplete report {p}: {sorted(d)}"
        runs.setdefault(m.group(1), {})[int(m.group(2))] = d
    summ = {}
    for r, seeds in runs.items():
        summ[r] = {"n_seeds": len(seeds), "seeds": sorted(seeds)}
        for k in KEYS:
            vals = [seeds[s][k] for s in sorted(seeds)]
            summ[r][k] = {"mean": statistics.mean(vals), "min": min(vals), "max": max(vals)}
    json.dump({"simulator": "The ONE (github.com/akeranen/the-one, commit c973bfb)",
               "settings": "tele_base.txt + Group.router line (tele_<Router>.txt); MovementModel.rngSeed = 1..5",
               "per_seed": runs, "summary": summ}, open(a.out, "w"), indent=1)
    for r, s in summ.items():
        print(r, s["n_seeds"], {k: round(s[k]["mean"], 3) for k in ("delivery_prob", "overhead_ratio", "latency_avg", "created", "delivered")})


if __name__ == "__main__":
    main()
