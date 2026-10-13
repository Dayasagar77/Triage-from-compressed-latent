"""
make_split_files.py - Reviewer-4 comment #5: supplementary split files from the official BreakHis Folds.csv.

For each of the five official splits, lists every slide folder (the dataset's patient unit) with its patient
identifier, tumour class, benign/malignant label, side of the split (train/test) and number of images, and writes
a per-split summary plus a few structural facts (how often each folder is tested, which folders are never tested,
and identifiers whose folders fall on both sides of a split).

    python make_split_files.py --folds "<archive/Folds.csv>" --out splits
Outputs: splits/fold{1..5}_slides.csv, splits/fold_summary.csv, splits/split_facts.json
"""
import argparse, json, os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", required=True); ap.add_argument("--out", default="splits")
    a = ap.parse_args()
    d = pd.read_csv(a.folds)
    parts = d['filename'].str.split('/')
    d['slide_folder'] = parts.str[-3]                       # .../<class>/SOB/<tumour>/<slide folder>/<mag>/<file>
    d['tumour_class'] = parts.str[-4]
    d['benign_malignant'] = parts.str[-6]
    d['patient_id'] = d['slide_folder'].str.split('-').str[-1]
    os.makedirs(a.out, exist_ok=True)
    summ, tests = [], {}
    for f in sorted(d.fold.unique()):
        g = d[d.fold == f]
        t = (g.groupby(['slide_folder', 'patient_id', 'tumour_class', 'benign_malignant', 'grp']).size()
             .rename('n_images').reset_index().sort_values('slide_folder'))
        t.insert(0, 'fold', f)
        t[['fold', 'slide_folder', 'patient_id', 'tumour_class', 'benign_malignant', 'grp', 'n_images']].to_csv(
            os.path.join(a.out, f'fold{f}_slides.csv'), index=False)
        for side in ('train', 'test'):
            s = g[g.grp == side]
            summ.append(dict(fold=f, split=side, n_images=len(s), n_slide_folders=s.slide_folder.nunique(),
                             n_patient_ids=s.patient_id.nunique(), n_benign_img=int((s.benign_malignant == 'benign').sum()),
                             n_malignant_img=int((s.benign_malignant == 'malignant').sum())))
        tests[f] = set(g[g.grp == 'test'].slide_folder)
    pd.DataFrame(summ).to_csv(os.path.join(a.out, 'fold_summary.csv'), index=False)
    folders = sorted(d.slide_folder.unique())
    appear = {s: sum(s in t for t in tests.values()) for s in folders}
    cross = []
    for f in sorted(d.fold.unique()):
        g = d[d.fold == f][['slide_folder', 'patient_id', 'grp']].drop_duplicates()
        for pid, x in g.groupby('patient_id'):
            if set(x.grp) == {'train', 'test'}:
                cross.append(dict(fold=int(f), patient_id=pid, test=sorted(x[x.grp == 'test'].slide_folder),
                                  train=sorted(x[x.grp == 'train'].slide_folder)))
    facts = dict(test_appearances={str(k): sum(v == k for v in appear.values()) for k in sorted(set(appear.values()))},
                 never_tested=[s for s in folders if appear[s] == 0], cross_split_id=cross)
    json.dump(facts, open(os.path.join(a.out, 'split_facts.json'), 'w'), indent=1)
    print('wrote', a.out)


if __name__ == "__main__":
    main()
