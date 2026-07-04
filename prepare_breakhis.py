"""
prepare_breakhis.py — reorganize the raw BreakHis download into the 8-subtype
ImageFolder layout expected by train.py.

BreakHis encodes the tumour subtype in every filename, e.g.:
    SOB_B_TA-14-4659-40-001.png   -> TA  = Tubular Adenoma   (benign)
    SOB_M_DC-14-2523-40-001.png   -> DC  = Ductal Carcinoma  (malignant -> "Carcinoma")
Format: SOB_<B|M>_<CODE>-<year>-<slide>-<magnification>-<seq>.png
This script reads that code from each file, so it works no matter how the raw
folders are nested (.../benign/SOB/<subtype>/<patient>/<mag>/*.png).

USAGE (Windows PowerShell):
    python prepare_breakhis.py --src "D:\path\to\BreaKHis_v1" --dst "path\to\BreakHis_8class"
    # optional: only one magnification (40, 100, 200 or 400):
    python prepare_breakhis.py --src ... --dst ... --mag 100
    # optional: move or symlink instead of copy:
    python prepare_breakhis.py --src ... --dst ... --mode move

Then train:
    python train.py --data "path\to\BreakHis_8class" --epochs 25 --k 5
"""
import argparse
import os
import shutil

# BreakHis filename code -> subtype folder name (must match SUBTYPES in ai_engine.py)
CODE_TO_SUBTYPE = {
    "A":  "Adenosis",            # benign
    "F":  "Fibroadenoma",        # benign
    "PT": "Phyllodes Tumor",     # benign
    "TA": "Tubular Adenoma",     # benign
    "DC": "Carcinoma",           # malignant (Ductal Carcinoma)
    "LC": "Lobular Carcinoma",   # malignant
    "MC": "Mucinous Carcinoma",  # malignant
    "PC": "Papillary Carcinoma", # malignant
}
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def parse_code(filename):
    """Extract the subtype code from a BreakHis filename. Returns code or None."""
    name = os.path.basename(filename)
    stem = os.path.splitext(name)[0]
    head = stem.split("-")[0]          # e.g. "SOB_M_DC"
    parts = head.split("_")
    if len(parts) < 3:
        return None
    code = parts[-1].upper()           # "DC"
    return code if code in CODE_TO_SUBTYPE else None


def parse_magnification(filename):
    """Return the magnification field (e.g. '40','100','200','400') or None."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    fields = stem.split("-")
    # SOB_<...>-<year>-<slide>-<MAG>-<seq>  -> mag is the 4th dash field (index 3)
    if len(fields) >= 5:
        return fields[3]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Root of the raw BreakHis download")
    ap.add_argument("--dst", required=True, help="Output ImageFolder root to create")
    ap.add_argument("--mag", default="all", help="Magnification filter: 40/100/200/400 or 'all'")
    ap.add_argument("--mode", choices=["copy", "move", "symlink"], default="copy")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        raise SystemExit(f"--src not found: {args.src}")

    for sub in CODE_TO_SUBTYPE.values():
        os.makedirs(os.path.join(args.dst, sub), exist_ok=True)

    counts = {sub: 0 for sub in CODE_TO_SUBTYPE.values()}
    skipped = 0
    seen = 0

    for root, _dirs, files in os.walk(args.src):
        for fn in files:
            if not fn.lower().endswith(IMG_EXTS):
                continue
            seen += 1
            code = parse_code(fn)
            if code is None:
                skipped += 1
                continue
            if args.mag != "all":
                mag = parse_magnification(fn)
                if mag != str(args.mag):
                    continue
            subtype = CODE_TO_SUBTYPE[code]
            src_path = os.path.join(root, fn)
            dst_path = os.path.join(args.dst, subtype, fn)
            if os.path.exists(dst_path):
                continue
            if args.mode == "copy":
                shutil.copy2(src_path, dst_path)
            elif args.mode == "move":
                shutil.move(src_path, dst_path)
            else:
                try:
                    os.symlink(os.path.abspath(src_path), dst_path)
                except OSError:
                    shutil.copy2(src_path, dst_path)  # symlink may need admin on Windows
            counts[subtype] += 1

    print("\n=== BreakHis prepared ===")
    total = 0
    for sub, n in counts.items():
        print(f"  {sub:22s}: {n}")
        total += n
    print(f"  {'TOTAL':22s}: {total}")
    print(f"  images seen: {seen}, skipped (unrecognized name): {skipped}")
    if total == 0:
        print("\nWARNING: 0 images sorted. Check that --src points to the folder that")
        print("contains the BreakHis .png files (filenames like SOB_M_DC-14-2523-40-001.png).")
    else:
        print(f"\nReady. Train with:\n  python train.py --data \"{args.dst}\" --epochs 25 --k 5")


if __name__ == "__main__":
    main()
