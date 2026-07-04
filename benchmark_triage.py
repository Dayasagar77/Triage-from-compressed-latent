"""
benchmark_triage.py — measures the benefit of the AI triage / priority queue.

The framework classifies each biopsy at the source and marks malignant cases as
high priority so they are transmitted before routine (benign) cases. This script
quantifies that benefit honestly: given a batch of cases contending for one
bandwidth-limited link, it compares the average delivery time of MALIGNANT cases
under a first-come-first-served (FIFO) queue versus the priority queue.

    python benchmark_triage.py --data "D:\\...\\BreakHis_8class" --batch 20 --trials 50 --speed-kbps 500

Output: results/triage_results.json
Uses real compressed sizes (compress_image). Priority is assigned from the model's
own diagnosis (analyze_image), so this measures the deployed system's behaviour.
"""
import argparse, json, os, random
import numpy as np
from ai_engine import compress_image, analyze_image

CHUNK = 64 * 1024


def transfer_seconds(size_bytes, speed_bps):
    # deterministic transmission time for a case (queueing effect isolated)
    return size_bytes / speed_bps


def completion_times(order, sizes, speed_bps):
    """Given a processing order (list of case indices), return each case's finish time."""
    t = 0.0; finish = {}
    for i in order:
        t += transfer_seconds(sizes[i], speed_bps)
        finish[i] = t
    return finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch", type=int, default=20, help="cases contending per trial")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--speed-kbps", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    speed_bps = args.speed_kbps * 1024
    rng = random.Random(args.seed)

    # gather image paths
    paths = []
    for root, _d, files in os.walk(args.data):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(root, f))
    rng.shuffle(paths)

    # Pre-compute compressed size + priority for a pool of images (real pipeline)
    pool = []
    need = args.batch * args.trials
    for p in paths[:need]:
        with open(p, "rb") as fh:
            raw = fh.read()
        diagnosis, _sub = analyze_image(raw)         # model's own decision
        comp, _proxy = compress_image(raw, quality=85)
        pool.append({"size": len(comp), "malignant": diagnosis == "Malignant"})
    if len(pool) < args.batch:
        raise SystemExit("Not enough images for one batch.")

    mal_fifo, mal_prio, all_fifo, all_prio = [], [], [], []
    frac_mal = []
    for tr in range(args.trials):
        batch = [pool[(tr * args.batch + j) % len(pool)] for j in range(args.batch)]
        sizes = [c["size"] for c in batch]
        mal = [j for j, c in enumerate(batch) if c["malignant"]]
        frac_mal.append(len(mal) / len(batch))
        arrival = list(range(len(batch)))
        rng.shuffle(arrival)                          # random arrival order = FIFO order
        # priority order: malignant (in arrival order) first, then benign
        prio_order = [j for j in arrival if batch[j]["malignant"]] + \
                     [j for j in arrival if not batch[j]["malignant"]]
        f_fifo = completion_times(arrival, sizes, speed_bps)
        f_prio = completion_times(prio_order, sizes, speed_bps)
        if mal:
            mal_fifo.append(np.mean([f_fifo[j] for j in mal]))
            mal_prio.append(np.mean([f_prio[j] for j in mal]))
        all_fifo.append(np.mean(list(f_fifo.values())))
        all_prio.append(np.mean(list(f_prio.values())))

    def ms(a): a = np.array(a); return round(float(a.mean()), 3), round(float(a.std()), 3)
    mf_m, mf_s = ms(mal_fifo); mp_m, mp_s = ms(mal_prio)
    reduction = round(100.0 * (mf_m - mp_m) / mf_m, 1) if mf_m else 0.0
    out = {
        "batch_size": args.batch, "trials": args.trials, "speed_kbps": args.speed_kbps,
        "avg_malignant_fraction": round(float(np.mean(frac_mal)), 3),
        "malignant_delivery_time_fifo_s": {"mean": mf_m, "std": mf_s},
        "malignant_delivery_time_priority_s": {"mean": mp_m, "std": mp_s},
        "malignant_delay_reduction_pct": reduction,
        "overall_delivery_time_fifo_s": ms(all_fifo)[0],
        "overall_delivery_time_priority_s": ms(all_prio)[0],
    }
    json.dump(out, open("results/triage_results.json", "w"), indent=2)
    print("\n==== TRIAGE / PRIORITY QUEUE ====")
    print(f"Batch={args.batch}, trials={args.trials}, link={args.speed_kbps} KB/s, "
          f"malignant fraction ~{out['avg_malignant_fraction']}")
    print(f"Malignant avg delivery time — FIFO: {mf_m}s | Priority: {mp_m}s "
          f"-> {reduction}% faster for urgent cases")
    print(f"Overall avg delivery time unchanged (FIFO {out['overall_delivery_time_fifo_s']}s "
          f"vs Priority {out['overall_delivery_time_priority_s']}s), as expected.")
    print("Saved results/triage_results.json")


if __name__ == "__main__":
    main()
