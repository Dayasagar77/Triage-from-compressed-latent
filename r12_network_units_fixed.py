"""
r12_network_units_fixed.py - Reviewer-4 comments #4, #7 and #12 (and the queue part of #1/#2).

Why this script exists
----------------------
The original transport scripts (benchmark_network.py, benchmark_triage.py, r7_resumable_tcp.py)
set the link speed as  speed = 500 * 1024  BYTES per second (500 KiB/s, about 4.1 Mb/s) and, in
benchmark_network.py / r7_resumable_tcp.py, charged a full 64 KiB chunk time for every chunk
whatever the payload size. queue_sim.py and r2_caselevel.py instead used 500 kb/s
(49 KB * 8 / 500 = 0.784 s per case). This script puts every transport and queue result on ONE
link definition and ONE timing rule:

    link rate R = 500 kb/s = 500,000 bit/s          (1 kb = 1,000 bit)
    1 KB = 1,024 bytes                               (payload sizes)
    time to send b bytes = 8*b / R                   (actual bytes, not a full chunk)

and repeats the comparisons for the payload actually recommended for deployment.

What it computes (all SIMULATION under stated parameters, not network measurements)
  A. Transport comparison under IDENTICAL outage traces (comments #4, #7): TCP (non-resumable),
     TCP with resumable upload (byte-range resume, handshake per reconnection) and single-custody
     store-and-forward (DTN), for each payload and each outage condition:
     delivery %, mean latency, mean overhead, mean reconnections.
  B. Application-layer overhead O = (kappa-1)*S + rho*S (the paper's Eq. 10) for each payload.
  C. Promote-only non-preemptive M/D/1 priority queue (Poisson arrivals), discrete-event
     simulation cross-checked against Cobham's formula, for a given case-level operating point
     (sensitivity, specificity) including false-positive promotions (comments #1, #2).

Outage model (unchanged from the original scripts): the link is up for a transmission slot with
probability p; otherwise an outage of duration U(a, b) seconds occurs.
  Real-Time p=0.98, U(0.1,0.5);  Delayed p=0.70, U(2,5);  Extreme p=0.30, U(5,10).
Transport parameters (stated, not tuned; as in r7_resumable_tcp.py): 64 KiB segments,
RTT 0.2 s, reconnection handshake 1.5*RTT = 0.30 s and 0.20 KB per reconnection,
DTN control overhead 2% of payload (the paper's modelling heuristic), non-resumable TCP fails if an
outage interrupts a transfer that has started.

    python r12_network_units_fixed.py --payloads payloads.json --out results/network_units_fixed.json
"""
import argparse, json, math, random, statistics, heapq, collections

LINK_BPS = 500_000                     # 500 kb/s
KIB = 1024
SEG = 64 * KIB                         # segment / bundle size in bytes
RTT = 0.2; HS_TIME = 1.5 * RTT; HS_KB = 0.20; DTN_CTRL = 0.02
CONDITIONS = {"Real-Time": (0.98, (0.1, 0.5)), "Delayed": (0.70, (2.0, 5.0)), "Extreme": (0.30, (5.0, 10.0))}
KAPPA_RHO = {"TCP/IP": (1, 0.05), "Epidemic": (5, 0.10), "PRoPHET": (2, 0.07), "Single-custody DTN": (1, 0.02)}


def tx_time(nbytes):
    return 8.0 * nbytes / LINK_BPS


def trace_stream(rng, p, wait):
    """Lazily drawn outage trace (up, outage_seconds). Each trial has its own seeded stream, so the
    same trial sees the same trace for every transport AND every payload."""
    while True:
        yield (True, 0.0) if rng.random() <= p else (False, rng.uniform(*wait))


def run_transfer(nbytes, trace, max_slots=1_000_000):
    """Walk the shared trace sending segments of the ACTUAL payload; return what each transport sees."""
    segs = [SEG] * (nbytes // SEG) + ([nbytes % SEG] if nbytes % SEG else [])
    t, sent, reconnections, started, in_gap, k = 0.0, 0, 0, False, False, 0
    while sent < len(segs) and k < max_slots:
        up, gap = next(trace); k += 1
        if up:
            if in_gap and started:
                reconnections += 1
            in_gap, started = False, True
            t += tx_time(segs[sent]); sent += 1
        else:
            if started:
                in_gap = True
            t += gap
    return dict(ok=sent == len(segs), t=t, rec=reconnections, nseg=len(segs))


def transport_table(payloads, trials, seed):
    out = {}
    for pname, nbytes in payloads.items():
        out[pname] = {}
        for cname, (p, wait) in CONDITIONS.items():
            acc = {k: {"n_ok": 0, "lat": [], "oh": [], "rec": []} for k in ("TCP_nonresumable", "TCP_resumable", "DTN_single_custody")}
            for trial in range(trials):
                rng = random.Random(f"{seed}-{cname}-{trial}")   # same trace per trial for every payload and transport
                r = run_transfer(nbytes, trace_stream(rng, p, wait))
                if r["ok"] and r["rec"] == 0:
                    a = acc["TCP_nonresumable"]; a["n_ok"] += 1; a["lat"].append(r["t"]); a["oh"].append(0.0); a["rec"].append(0)
                if r["ok"]:
                    a = acc["TCP_resumable"]; a["n_ok"] += 1; a["lat"].append(r["t"] + r["rec"] * HS_TIME)
                    a["oh"].append(r["rec"] * HS_KB); a["rec"].append(r["rec"])
                    a = acc["DTN_single_custody"]; a["n_ok"] += 1; a["lat"].append(r["t"])
                    a["oh"].append(nbytes / KIB * DTN_CTRL); a["rec"].append(0)
            out[pname][cname] = {k: {"delivery_pct": round(100.0 * v["n_ok"] / trials, 1),
                                     "mean_latency_s": round(statistics.mean(v["lat"]), 3) if v["lat"] else None,
                                     "mean_overhead_kb": round(statistics.mean(v["oh"]), 3) if v["oh"] else None,
                                     "mean_reconnections": round(statistics.mean(v["rec"]), 2) if v["rec"] else None}
                                 for k, v in acc.items()}
            out[pname][cname]["segments"] = math.ceil(nbytes / SEG)
            out[pname][cname]["ideal_tx_time_s"] = round(tx_time(nbytes), 4)
    return out


def overhead_table(payloads):
    return {pname: {proto: round(((k - 1) * nbytes + rho * nbytes) / KIB, 2) for proto, (k, rho) in KAPPA_RHO.items()}
            for pname, nbytes in payloads.items()}


# ---------- promote-only priority queue ----------
def cobham(rho, D, frac_high):
    """Non-preemptive M/D/1, 2 classes; mean waiting times (excl. service)."""
    lam = rho / D; R = lam * D * D / 2.0; rho_h = frac_high * rho
    return R / (1 - rho_h), R / ((1 - rho_h) * (1 - rho)), R / (1 - rho)


def des(rho, D, p_mal, sens, spec, n=300000, seed=0):
    rnd = random.Random(seed); lam = rho / D; t = 0.0; jobs = []
    for _ in range(n):
        t += rnd.expovariate(lam); mal = rnd.random() < p_mal
        prom = (rnd.random() < sens) if mal else (rnd.random() < (1 - spec))
        jobs.append((t, 0 if prom else 1, mal))
    qh, ql = collections.deque(), collections.deque(); ev = [(a[0], 0, i) for i, a in enumerate(jobs)]
    heapq.heapify(ev); busy = False; w = {"high": [], "low": [], "mal": [], "mal_missed": []}

    def start(job, now):
        wt = now - job[0]
        (w["high"] if job[1] == 0 else w["low"]).append(wt)
        if job[2]:
            w["mal"].append(wt)
            if job[1] == 1: w["mal_missed"].append(wt)
        return now + D
    while ev:
        now, typ, i = heapq.heappop(ev)
        if typ == 0:
            if not busy:
                busy = True; heapq.heappush(ev, (start(jobs[i], now), 1, -1))
            else:
                (qh if jobs[i][1] == 0 else ql).append(jobs[i])
        else:
            if qh: heapq.heappush(ev, (start(qh.popleft(), now), 1, -1))
            elif ql: heapq.heappush(ev, (start(ql.popleft(), now), 1, -1))
            else: busy = False
    cut = lambda x: x[len(x) // 20:]                    # drop the first 5% (warm-up)
    # FIFO baseline on an independent arrival stream with the same rate
    rnd2 = random.Random(seed + 1); t = 0.0; free = 0.0; wf = []
    for _ in range(n):
        t += rnd2.expovariate(lam); s = max(t, free); wf.append(s - t); free = s + D
    return {k: (statistics.mean(cut(v)) if cut(v) else None) for k, v in w.items()} | {"fifo": statistics.mean(cut(wf))}   # None: no job of that kind (e.g. no missed cases when sens = 1)


def queue_table(op, D, p_mal, rhos=(0.3, 0.5, 0.7, 0.9), seed=20260929):
    sens, spec = op["sens"], op["spec"]; fh = p_mal * sens + (1 - p_mal) * (1 - spec)
    rows = []
    for rho in rhos:
        wh, wl, wf = cobham(rho, D, fh)
        mal_th = sens * wh + (1 - sens) * wl
        sim = des(rho, D, p_mal, sens, spec, seed=seed + int(rho * 100))
        rows.append(dict(rho=rho, promoted_fraction=round(fh, 4),
                         W_promoted_theory_s=round(wh, 4), W_notpromoted_theory_s=round(wl, 4), W_fifo_theory_s=round(wf, 4),
                         W_malignant_theory_s=round(mal_th, 4),
                         malignant_wait_reduction_pct_theory=round(100 * (wf - mal_th) / wf, 1),
                         W_malignant_sim_s=round(sim["mal"], 4), W_missed_malignant_sim_s=None if sim["mal_missed"] is None else round(sim["mal_missed"], 4),
                         W_fifo_sim_s=round(sim["fifo"], 4),
                         malignant_wait_reduction_pct_sim=round(100 * (sim["fifo"] - sim["mal"]) / sim["fifo"], 1),
                         missed_over_fifo_sim=None if sim["mal_missed"] is None else round(sim["mal_missed"] / sim["fifo"], 3)))
    return dict(operating_point=op, malignant_fraction=p_mal, service_time_s=round(D, 5), rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payloads", required=True, help="json: {name: bytes, ...}; key 'queue_payload' names the one used for D")
    ap.add_argument("--ops", required=True, help="json: {name: {sens, spec, source}}")
    ap.add_argument("--p-mal", type=float, default=0.664)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.payloads)); payloads = cfg["payloads"]; qp = cfg["queue_payload"]
    ops = json.load(open(a.ops))
    D = tx_time(payloads[qp])
    res = {"units": {"link_rate": "500 kb/s = 500,000 bit/s", "KB": "1,024 bytes", "tx_time": "8*bytes/500,000 s",
                     "segment_bytes": SEG},
           "conditions": {k: {"p_up": v[0], "outage_s": v[1]} for k, v in CONDITIONS.items()},
           "payload_bytes": payloads, "payload_notes": cfg.get("notes", {}),
           "transport": transport_table(payloads, a.trials, a.seed),
           "overhead_eq10_kb": overhead_table(payloads),
           "queue": {"service_payload": qp, "service_time_s": round(D, 5),
                     "note": "percent reductions at a fixed utilisation rho are independent of the service time D; absolute waits scale with D",
                     "operating_points": {k: queue_table(v, D, a.p_mal) for k, v in ops.items()}}}
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
