"""
benchmark_network.py — transmission-side evaluation on REAL compressed images.
No MongoDB. Reports (per network condition):
  * real compression savings (compressed latent vs original file),
  * simulated DTN store-and-forward metrics (success rate, throughput, latency),
  * a fair protocol comparison (TCP/IP, Epidemic, PRoPHET, Codec-Aware DTN) over one trace,
  * a MULTI-BUNDLE (large-file) protocol test that exposes TCP/IP's mid-transfer failure.

Single condition:   python benchmark_network.py --data DATA --n 200 --condition Delayed --speed-kbps 500
All conditions:     python benchmark_network.py --data DATA --n 200 --all-conditions --speed-kbps 500

These are SIMULATION outputs under explicitly stated parameters, not physical-network
measurements. Output: results/network_results[_<condition>].json
"""
import argparse, json, os, random
import numpy as np
from ai_engine import compress_image

CONDITIONS = {
    "Real-Time": {"reliability": 0.98, "wait": (0.1, 0.5)},
    "Delayed":   {"reliability": 0.70, "wait": (2.0, 5.0)},
    "Extreme":   {"reliability": 0.30, "wait": (5.0, 10.0)},
}
CHUNK = 64 * 1024
HEADER_OVERHEAD = 0.02


def simulate_protocol(trace, num_bundles, total_kb, copies, continuous, ctrl):
    delivered, last = 0, 0
    if continuous:
        run, started = 0, False
        for slot, up in enumerate(trace):
            if up:
                started = True; run += 1
                if run >= num_bundles:
                    delivered = num_bundles; last = slot; break
            elif started:
                delivered = 0; break
    else:
        sent = 0
        for slot, up in enumerate(trace):
            if up and sent < num_bundles:
                sent += 1; last = slot
            if sent >= num_bundles:
                break
        delivered = sent
    dr = (delivered / num_bundles) * 100 if num_bundles else 0
    overhead = total_kb * max(0, copies - 1) + total_kb * ctrl
    return {"delivery_ratio": round(dr, 1), "overhead_kb": round(overhead, 1),
            "delivered": delivered == num_bundles}


def benchmark_protocols(trace, num_bundles, total_kb):
    return {
        "TCP_IP":     simulate_protocol(trace, num_bundles, total_kb, 1, True, 0.05),
        "Epidemic":   simulate_protocol(trace, num_bundles, total_kb, 5, False, 0.10),
        "PRoPHET":    simulate_protocol(trace, num_bundles, total_kb, 2, False, 0.07),
        "Neural_DTN": simulate_protocol(trace, num_bundles, total_kb, 1, False, 0.02),
    }


def simulate_transfer(size_bytes, cond, speed_bps, rng):
    total_chunks = max(1, (size_bytes // CHUNK) + 1)
    sent, t, drops, trace, guard = 0, 0.0, 0, [], 0
    while sent < total_chunks and guard < 100000:
        guard += 1
        if rng.random() <= cond["reliability"]:
            trace.append(True); t += CHUNK / speed_bps; sent += 1
        else:
            trace.append(False); drops += 1; t += rng.uniform(*cond["wait"])
    return sent == total_chunks, t, trace, drops


def run_once(args, cond_name, comp_cache):
    cond = CONDITIONS[cond_name]
    speed_bps = args.speed_kbps * 1024
    rng = random.Random(args.seed)

    successes, latencies, throughputs = 0, [], []
    orig_total, comp_total = 0, 0
    proto_del = {k: 0 for k in ["TCP_IP", "Epidemic", "PRoPHET", "Neural_DTN"]}
    proto_oh = {k: 0.0 for k in proto_del}
    # multi-bundle (large-file) protocol test — use the ORIGINAL uncompressed size
    big_del = {k: 0 for k in proto_del}

    for p, (raw_len, comp_len) in comp_cache:
        orig_total += raw_len; comp_total += comp_len
        size = int(comp_len * (1 + HEADER_OVERHEAD))
        delivered, lat, trace, drops = simulate_transfer(size, cond, speed_bps, rng)
        successes += int(delivered)
        if delivered and lat > 0:
            latencies.append(lat); throughputs.append((comp_len / 1024) / lat)
        nb = max(1, (size // CHUNK) + 1)
        for k, v in benchmark_protocols(trace if trace else [True], nb, comp_len / 1024).items():
            proto_del[k] += int(v["delivered"]); proto_oh[k] += v["overhead_kb"]
        # large multi-bundle transfer of the ORIGINAL image (exposes TCP failure)
        _, _, big_trace, _ = simulate_transfer(raw_len, cond, speed_bps, rng)
        big_nb = max(1, (raw_len // CHUNK) + 1)
        for k, v in benchmark_protocols(big_trace if big_trace else [True], big_nb, raw_len / 1024).items():
            big_del[k] += int(v["delivered"])

    n = len(comp_cache)
    out = {
        "condition": cond_name, "base_speed_kbps": args.speed_kbps, "n_images": n,
        "transmission_success_rate_pct": round(100.0 * successes / n, 2),
        "avg_throughput_kbps": round(float(np.mean(throughputs)), 2) if throughputs else 0,
        "avg_latency_s": round(float(np.mean(latencies)), 2) if latencies else 0,
        "compression_savings_pct": round(100.0 * (1 - comp_total / orig_total), 2),
        "avg_original_kb": round(orig_total / n / 1024, 1),
        "avg_compressed_kb": round(comp_total / n / 1024, 1),
        "protocol_comparison_compressed": {
            k: {"delivery_ratio_pct": round(100.0 * proto_del[k] / n, 1),
                "avg_overhead_kb": round(proto_oh[k] / n, 2)} for k in proto_del},
        "protocol_delivery_large_multibundle_pct": {
            k: round(100.0 * big_del[k] / n, 1) for k in big_del},
    }
    suffix = f"_{cond_name}" if args.all_conditions else ""
    json.dump(out, open(f"results/network_results{suffix}.json", "w"), indent=2)
    print(f"\n== {cond_name} @ {args.speed_kbps} KB/s ==")
    print(f"  success {out['transmission_success_rate_pct']}% | throughput {out['avg_throughput_kbps']} KB/s | "
          f"latency {out['avg_latency_s']} s | savings {out['compression_savings_pct']}%")
    print(f"  large-file delivery (multi-bundle): " +
          ", ".join(f"{k} {v}%" for k, v in out["protocol_delivery_large_multibundle_pct"].items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--condition", choices=list(CONDITIONS), default="Delayed")
    ap.add_argument("--all-conditions", dest="all_conditions", action="store_true")
    ap.add_argument("--speed-kbps", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    # compress all images ONCE (shared across conditions)
    paths = []
    for root, _d, files in os.walk(args.data):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(root, f))
    random.Random(args.seed).shuffle(paths)
    paths = paths[:args.n]
    print(f"Compressing {len(paths)} images (once)...")
    comp_cache = []
    for p in paths:
        with open(p, "rb") as fh:
            raw = fh.read()
        comp, _proxy = compress_image(raw, quality=85)
        comp_cache.append((p, (len(raw), len(comp))))

    conds = list(CONDITIONS) if args.all_conditions else [args.condition]
    summ = {c: run_once(args, c, comp_cache) for c in conds}
    if args.all_conditions:
        print("\n================ CONDITION COMPARISON ================")
        print(f"{'Condition':>10} | {'Success':>8} | {'Throughput':>11} | {'Latency':>8}")
        for c in conds:
            o = summ[c]
            print(f"{c:>10} | {o['transmission_success_rate_pct']:>6}% | "
                  f"{o['avg_throughput_kbps']:>8} KB/s | {o['avg_latency_s']:>6} s")
    print("\nSaved results/network_results*.json")


if __name__ == "__main__":
    main()
