#!/bin/bash
# run_r13.sh -- Reviewer 4, comments 1, 9 and 10: re-train the proposed source classifier with a fixed seed (42)
# and SAVE every held-out prediction (needed for the E1 row of the queue table, the classifier's patient-level
# intervals and the seeded proposed arm of Table 4). For Daya to put in /root/tele on FIST himself, next to
# r13_source_predictions.py. It waits for any running telepathology job, so it can be started at any time.
cd /root/tele || exit 1
export PYTHONIOENCODING=utf-8
mkdir -p logs results
echo "=== waiting for any running telepathology job to finish: $(date) ==="
while pgrep -f "run_full.sh|run_seeded.sh|r10_baselines.py|r5_codec_perfold.py" >/dev/null 2>&1; do sleep 60; done
echo "=== r13 (seed 42) start: $(date) ==="
~/venv/bin/python -u r13_source_predictions.py --data ~/tele/BreakHis_8class --folds ~/tele/Folds.csv --seed 42
RC=$?
if [ "$RC" -eq 0 ] && [ -s results/source_casewise_results.json ] && [ -s results/source_heldout_predictions.csv ]; then
    touch R13_DONE.flag
    echo "=== R13 DONE: $(date) ==="
else
    echo "r13 exit $RC, $(date)" > R13_FAILED.flag
    echo "=== R13 FAILED (exit $RC): $(date) ==="
fi
