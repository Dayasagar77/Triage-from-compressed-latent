"""
per_case_bytes.py - Table 8 (actual bytes per field and transmission time per case); Reviewer 2,
point 4, and Reviewer 4, comment 4.

Pure arithmetic on measured sizes; no model is run. Reads the mean bytes per 224x224 field from the
existing result files and converts them into bytes and seconds per case.

  * results/actual_bitstream_results.json -> factorized-prior and scale-hyperprior bitstreams at
        lambda 0.01: ACTUAL range-coded bytes (r4_actual_bitstream.py; every round trip lossless),
        each on its codec's own 400-image evaluation set (Set F and Set H)
  * results/compression_comparison.json   -> JPEG q20 and q30, JPEG2000 ratio 40 (actual bytes, Set F)
        and the residual-autoencoder payload, neural q90 (50,184 bytes, fixed size)
  * JPEG and JPEG2000 at the payload's SSIM (0.9794) are linearly interpolated between the two
    nearest measured operating points in compression_comparison.json.
Fields per case: 10 (an illustrative static-telepathology referral) and 96 (BreakHis average,
7,909 fields / 82 patients). Link rates: 64 kb/s and 500 kb/s (1 kb/s = 1,000 bit/s). Times are
payload only (bytes * 8 / rate), with no protocol overhead. 1 KB = 1,024 bytes.

The first version of this script (13 Oct 2026) used the entropy-model estimates of the learned
codecs (4,996 and 3,643 bytes); this version uses the actual bitstreams (4,989 and 3,632 bytes).

    python per_case_bytes.py --results results
Output: results/per_case_bytes.json
"""
import argparse, json, os


def interp_bytes_at_ssim(points, target):
    """points: {key: {avg_bytes, avg_ssim}}; linear interpolation of bytes at a target SSIM."""
    pts = sorted((v["avg_ssim"], v["avg_bytes"]) for v in points.values())
    for (s0, b0), (s1, b1) in zip(pts, pts[1:]):
        if s0 <= target <= s1:
            return b0 + (target - s0) / (s1 - s0) * (b1 - b0), (s0, b0, s1, b1)
    raise ValueError("target SSIM outside the measured range")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    J = lambda f: json.load(open(os.path.join(a.results, f)))
    cc, ab = J("compression_comparison.json"), J("actual_bitstream_results.json")
    fp, hp = ab["codecs"]["factorized_l0.01"], ab["codecs"]["hyperprior_l0.01"]
    assert fp["all_round_trips_lossless"] and hp["all_round_trips_lossless"]
    assert fp["eval_set"] == "F" and hp["eval_set"] == "H"

    payload = cc["neural"]["90"]
    target = payload["avg_ssim"]
    jpeg_eq, jpeg_br = interp_bytes_at_ssim(cc["jpeg"], target)
    jp2_eq, jp2_br = interp_bytes_at_ssim(cc["jpeg2000"], target)

    rows = [
        ("Factorized-prior bitstream (triage code)", fp["actual_bytes"], fp["ssim"], "actual range-coded bytes, Set F"),
        ("Scale-hyperprior bitstream", hp["actual_bytes"], hp["ssim"], "actual range-coded bytes, Set H"),
        ("JPEG, quality 20", cc["jpeg"]["20"]["avg_bytes"], cc["jpeg"]["20"]["avg_ssim"], "actual bytes, Set F"),
        ("JPEG, quality 30", cc["jpeg"]["30"]["avg_bytes"], cc["jpeg"]["30"]["avg_ssim"], "actual bytes, Set F"),
        ("JPEG2000, ratio 40", cc["jpeg2000"]["40"]["avg_bytes"], cc["jpeg2000"]["40"]["avg_ssim"], "actual bytes, Set F"),
        ("Residual-autoencoder latent (49 KB payload)", payload["avg_bytes"], target, "actual bytes (fixed size), Set F"),
        ("JPEG at equal SSIM", jpeg_eq, target, "interpolated " + str(jpeg_br)),
        ("JPEG2000 at equal SSIM", jp2_eq, target, "interpolated " + str(jp2_br)),
    ]
    rates = {"64 kb/s": 64_000, "500 kb/s": 500_000}
    out = []
    for label, b, s, kind in rows:
        r = {"representation": label, "bytes_per_field": round(b, 2), "kib_per_field": round(b / 1024, 3),
             "ssim": s, "size_kind": kind}
        for n in (10, 96):
            for rn, bps in rates.items():
                r[f"case{n}_s_at_{rn}"] = round(b * n * 8 / bps, 3)
        out.append(r)
    lat, jpg = out[0], out[3]
    summary = {
        "case96_jpeg_s_at_64": jpg["case96_s_at_64 kb/s"],
        "case96_saving_latent_vs_jpeg_s_at_64": round(jpg["case96_s_at_64 kb/s"] - lat["case96_s_at_64 kb/s"], 3),
        "case96_saving_latent_vs_jpeg_s_at_500": round(jpg["case96_s_at_500 kb/s"] - lat["case96_s_at_500 kb/s"], 3),
        "case10_saving_latent_vs_jpeg_s_at_64": round(jpg["case10_s_at_64 kb/s"] - lat["case10_s_at_64 kb/s"], 3),
        "payload_over_jpeg_equal_ssim": round(payload["avg_bytes"] / jpeg_eq, 2),
        "payload_over_jpeg2000_equal_ssim": round(payload["avg_bytes"] / jp2_eq, 2),
        "int8_payload_over_jpeg_equal_ssim": round(cc["neural"]["50"]["avg_bytes"] / jpeg_eq, 2),
    }
    json.dump({"rows": out, "summary": summary}, open(os.path.join(a.results, "per_case_bytes.json"), "w"), indent=2)
    for r in out:
        print(f"{r['representation'][:46]:46s} {r['bytes_per_field']:>9.1f} B | 10@64k {r['case10_s_at_64 kb/s']:7.2f} s | "
              f"96@64k {r['case96_s_at_64 kb/s']:7.2f} s | 96@500k {r['case96_s_at_500 kb/s']:6.2f} s")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
