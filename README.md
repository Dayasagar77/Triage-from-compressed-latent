# Triage from the Compressed Latent

Code and machine-readable results for the paper **"Triage from the Compressed Latent:
Bandwidth-Constrained Telepathology with Delay-Tolerant Delivery."**

An integrated telepathology pipeline for rural, low-bandwidth settings: a benign/malignant
**triage decision is read directly from the ~5 KB compressed latent** of a learned image
codec (without reconstruction), used to prioritise malignant cases over a delay-tolerant
(store-and-forward) link. Everything is evaluated **patient-wise** on BreakHis with the
official folds (a leakage-controlled protocol; Section 4.2 of the paper states the remaining
patient-identity caveats), with external validation on BACH.

> **Reproducibility note.** Appendix A of the revised paper maps each result to the script
> and result file that produced it; the table below follows it. The runs added in the revision
> fix their random seeds (each script states its seeds), except the zero-shot latent run on
> BACH (`r6_latent_bach.py`). The original training runs did not fix their seeds, so re-running
> them reproduces the protocol but not every digit. These files have not yet been audited
> independently.

## Repository layout

```
*.py                     experiment, training, and figure scripts
results/                 machine-readable results (JSON, CSV, logs)
results/figures/         generated figures
splits/                  per-fold lists of training and test slide folders (supplementary files)
theone_rerun/            The ONE settings and raw reports behind Table 12
theone/                  The ONE settings from the first submission (superseded, see below)
models/                  place trained weights here (see models/README_WEIGHTS.md)
paper/                   manuscript PDF from July 2026, before peer review
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data (download separately — public datasets)
- **BreakHis** (breast histopathology): https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
  Arrange as `BreakHis_8class/<subtype>/*.png` (use `prepare_breakhis.py`), with the official
  patient-wise folds in `archive/Folds.csv`.
- **BACH / ICIAR-2018** (external validation): https://iciar2018-challenge.grand-challenge.org/Dataset/
  Class folders `{Normal, Benign, InSitu, Invasive}` under a `Photos/` directory.

### Trained weights
Not stored in git (>100 MB). See [`models/README_WEIGHTS.md`](models/README_WEIGHTS.md) to
download them, or regenerate with the training scripts.

## Reproducing the paper

Each result is produced by the script shown here (Table 16 of the paper). Result files are in
`results/`, `splits/` and `theone_rerun/`. Paths to the datasets are passed with `--data` /
`--folds` / `--bach`.

| Paper item | Script | Result file |
|---|---|---|
| Table 2 (dataset) | `prepare_breakhis.py` (folder counts) | `BreakHis_8class/` |
| Split facts and supplementary split files (§4.2) | `make_split_files.py` (from `Folds.csv`) | `fold1..5_slides.csv`, `fold_summary.csv`, `split_facts.json` |
| Table 3 (per-magnification) | `train.py --per-mag` | `kfold_results_{40,100,200,400}X.json` |
| Table 4 (baseline vs proposed) | `train.py --model baseline` | `kfold_results_baseline.json`, `kfold_results.json` |
| Table 4 ablations (no SE; direct two-class) | `r10_baselines.py` | `kfold_results_resnet50_nose.json`, `kfold_results_direct_binary.json` |
| Table 4 seeded proposed row; Table 9 full-image rows (fixed seeds) | `r13_source_predictions.py` | `source_casewise_results.json`, `source_heldout_predictions.csv` |
| Table 5 (per-subtype metrics) | `train.py` (pooled report) | `classification_report.txt` |
| Table 6 (protocol comparison) | literature summary | references [39]-[44] |
| Leakage ablation (§5.1, Table 6) | `imagewise_ablation.py` | `imagewise_results.json` |
| Hierarchical ablation (§5.1) | `train_hierarchical.py` | `hierarchical_results.json` |
| Table 7 (rate-distortion) | `compare_compression.py`, `train_compression_v2.py`, `train_hyperprior_pure.py` | `compression_comparison.json`, `learned_codec_results.json`, `hyperprior_results.json` |
| Table 8 (actual bytes and time per case) | `per_case_bytes.py` | `per_case_bytes.json` (from `actual_bitstream_results.json`, `compression_comparison.json`) |
| Actual bitstream bytes (§5.2, Table 8) | `r4_actual_bitstream.py` | `actual_bitstream_results.json` |
| Hyperprior-latent triage (§5.2) | `triage_from_hyperprior_latent.py` | `hyperprior_latent_triage.json` |
| Edge encoder latency (§5.2) | `edge_latency.py` | `edge_latency.json` |
| Image-wise latent operating point (§5.2, §5.3, Table 14) | `emit_latent_confusion.py` | `latent_confusion.json` |
| Table 9 latent rows (codec retrained within each fold) | `r5_codec_perfold.py` | `latent_leakagefree_results.json` |
| Patient-level intervals (§5.3) | `patient_level_intervals.py` | `patient_level_intervals.json` |
| Patient-identifier sensitivity analysis (§5.3) | `r9b_latent_casewise.py` | `latent_casewise_results.json`, `latent_heldout_predictions.csv` |
| Latent UMAP (Fig 8, §5.3) | `latent_umap.py` | `latent_umap.json` |
| Diagnosis-vs-bytes (Fig 9, §5.3) | `diagnose_vs_bytes.py` (data; plot by `make_figures_r4.py`) | `diagnose_vs_bytes.json` |
| Table 10 (transport at 500 kb/s, actual payloads) | `r12_network_units_fixed.py` | `network_units_fixed.json` |
| Table 11 (overhead) | `r12_network_units_fixed.py` (49 KB column also `benchmark_network.py`) | `network_units_fixed.json`, `network_results.json` |
| Table 12 (DTN simulation in The ONE, re-run) | The ONE, commit c973bfb (`theone_rerun/tele_base.txt`, `tele_<Router>.txt`); `theone_summary.py` | `theone_rerun/reports/*.txt`, `theone_rerun_results.json` |
| Table 13 (first-order estimate) | `realistic_triage_sim.py` | `realistic_triage_results.json`; r from `source_casewise_results.json` |
| Triage 31.8% (Section 5.5; times rescaled to 500 kb/s) | `benchmark_triage.py` | `triage_results.json` |
| Table 14, review queue | `queue_sim.py` (waits divided by its 0.784 s service time) | `queue_sim_results.json` |
| Table 14, transmission queue; analytic (Cobham) values (§5.5) | `queue_two_decisions.py` | `queue_two_decisions.json` |
| Truly malignant delay change (§5.5) | `queue_sim.py` (r·W_promoted + (1−r)·W_missed vs W_FIFO) | `queue_sim_results.json` |
| Figure 11, τ-sweep (§5.5) | `tau_sweep.py`; drawn by `make_fig11_service_units.py` | `tau_sweep.json` |
| External validation (§5.6) | `eval_external_bach.py` | `external_bach_results.json` |
| Domain adaptation (§5.6, Table 15) | `finetune_bach.py` | `bach_finetune_results.json` |
| Latent triage on BACH (§5.6, Table 15) | `r6_latent_bach.py`, `r6b_latent_bach_finetune.py` | `latent_bach_results.json`, `latent_bach_finetune_results.json` |
| Imbalanced fine-tune (§5.6) | `finetune_bach_imbalanced.py` | `bach_finetune_imbalanced.json` |
| BACH threshold recal (§5.6) | `recalibrate_bach_threshold.py` | `bach_threshold_recal.json` |
| Macenko stain norm (§5.6) | `bach_stain_macenko.py` | `bach_stain_macenko.json` |
| Figure 1 (architecture) | `make_architecture.py` | none (diagram) |
| Figure 2 (example fields) | none: BreakHis images `SOB_B_A-14-22549AB-100-001.png` and `SOB_M_DC-14-10926-100-001.png` | `BreakHis_8class/` |
| Figures 3–6, 9 and 10 (plots) | `make_figures_r4.py` (default mode draws the revised plots; `--check` redraws the first-submission images pixel for pixel) | `kfold_results*.json`, `confusion_matrix_40X.csv`, `compression_comparison.json`, `learned_codec_results.json`, `hyperprior_results.json`, `diagnose_vs_bytes.json`, `network_results.json` |
| Figure 7 (reconstruction, held-out images) | `make_reconstruction_figure_heldout.py` | `fig_reconstruction_heldout.png`, `fig_reconstruction_heldout.json` |
| Graphical abstract | `make_graphical_abstract_merged.py` | none (diagram) |

Example:
```bash
python emit_latent_confusion.py --data path/to/BreakHis_8class \
    --folds path/to/archive/Folds.csv --codec-lambda 0.01 --epochs 20
```

The DTN results in Table 12 are produced by [The ONE](https://akeranen.github.io/the-one/)
(commit c973bfb) with the settings in `theone_rerun/`; `theone_rerun/theone_summary.py`
summarises the raw reports in `theone_rerun/reports/`. `theone/README_THEONE.md` describes how
to build and run The ONE.

## Files from the first submission

Kept for the record. The revised paper does not use them, except where noted.

| File(s) | Status |
|---|---|
| `theone/`, `results/theone_results.json` | First-submission Table 12, whose settings did not match the reported scenario. Table 12 now comes from `theone_rerun/`. |
| `results/network_results_{Real-Time,Delayed,Extreme}.json` (from `benchmark_network.py`) | First-submission transport table at 500 × 1,024 bytes per second; its throughput and latency fields are superseded by `network_units_fixed.json`. The protocol-overhead values in `network_results.json` remain valid: Figure 10 and the 49 KB column of Table 11 use them. |
| `make_reconstruction_figure.py`, `results/figures/fig_reconstruction.png` | First-submission Figure 7, which used images from the codec's training set. Figure 7 now comes from `make_reconstruction_figure_heldout.py`. |
| `classify_from_latent.py`, `results/latent_classification_results.json` | First-submission latent-triage results. The Table 9 latent rows now come from `r5_codec_perfold.py` (codec retrained within each fold). |
| `synthetic_wsi_sim.py`, `results/synthetic_wsi_results.json` | First-submission synthetic whole-slide table; not in the revised paper. |
| `paper/paper.pdf` | Manuscript PDF from July 2026, before peer review. |

## Citation

```bibtex
@article{dayasagar_triage_compressed_latent,
  title   = {Triage from the Compressed Latent: Bandwidth-Constrained Telepathology with Delay-Tolerant Delivery},
  author  = {Dayasagar, G. and Nivethika, Deepa and Pandian, M. Senthil},
  year    = {2026},
  note    = {Preprint}
}
```

## License
Code released under the MIT License (see [LICENSE](LICENSE)). BreakHis and BACH are covered by
their own dataset licenses.
