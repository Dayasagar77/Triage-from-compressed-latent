# Response to Reviewer — Revision Round (Minor–Major Revision)

We thank the reviewer for the detailed and constructive report. All points are addressed below; numbers trace to result files in `upgraded_backend/results/`.

## Weaknesses

**W1 / Q1. WSI edge-compute scalability.** Agreed and now addressed (Section 5.6). Running the ResNet-50 on all ~10,000 tiles of a slide (58 ms each) would cost ~10 min and 2–3 kJ, impractical off-grid. The resolution, now stated, is the tiered design standard in WSI pipelines: a lightweight foreground detector or small MobileNet-class screen (a fraction of the ResNet cost) rejects background tiles and ranks the rest, so the heavy ResNet-50 and codec run only on the top-K candidate tiles (hundreds, not tens of thousands), cutting per-slide compute and energy by 1–2 orders of magnitude. Building this two-tier edge cascade is named a priority for the whole-slide extension.

**W2. Codec choice.** We agree the hyperprior should be the default. Section 5.2 now states plainly that, because the source classifier already dominates the edge memory budget, the factorized-prior codec's ~2 M-parameter saving is marginal; on any device that runs the classifier the **scale-hyperprior is the recommended default for both transmission and triage** (it beats JPEG on RD and triages at 81.1%, on par with the factorized codec's 82.3%). The factorized-prior codec is reserved for microcontroller-class nodes that run compression but not the classifier.

**W3 / Q2. Triage specificity, bursts, τ [measured].** We ran the empirical τ-sweep (`tau_sweep.py`, Figure 11). At ρ = 0.6, raising τ from 0.5 to 0.95 shrinks the promoted fraction (0.71 → 0.35) and the missed-malignant delay (1.02 → 0.75 s, ~27%) but sensitivity falls steeply (0.90 → 0.50). Since the delay is only weakly sensitive to τ at moderate load while sensitivity halves, a **low threshold (τ ≈ 0.5–0.6, sensitivity ≥ 0.87) is preferable** — false-positive promotions are not the binding constraint at realistic load. This is now in Section 5.5.

**W4 / stain. Macenko normalization [measured].** We tested it directly (`bach_stain_macenko.py`): Macenko normalization **did not recover** the gap — balanced accuracy fell (0.58 → 0.54) and AUC dropped (0.65 → 0.54). The BACH shift is therefore not primarily a global colour offset that normalization can remove; it reflects deeper morphological, class-composition, and scanner differences. Fine-tuning remains the effective remedy, and stain-invariant training is more promising than post-hoc normalization (Section 5.8). This corrected our prior expectation that normalization would help.

## Minor comments

1. **Table 1** — "Comp. pathology" → "Computational pathology."
2. **Section 3.1** — the capture-adequacy/ROI discussion is now elevated with a bold lead-in as the primary failure mode of patch-based telepathology.
3. **Figure 7** — regenerated with a 4× zoom inset (yellow box) on each panel so the chromatin/nuclear softening is directly visible in the original-vs-reconstruction comparison.
4. **Section 3.4** — ρ is now explicitly flagged at first mention as a deliberate application-layer abstraction (a real Bundle-Protocol header is a fixed size plus variable payload), not a standard networking model.
5. **Table 8** — caption now notes the scale-hyperprior's 0.9 KB / SSIM 0.625 point is a sub-clinical lower-bound anchor only.

## Questions

1. **Energy for WSI** — see W1: ~2–3 kJ for all-tile inference is indeed prohibitive, which is exactly why the two-tier cascade (cheap screen → ResNet on top-K) is required; this is now argued explicitly.
2. **τ tuning [measured]** — Figure 11 gives the empirical sweep; the delay is weakly sensitive to τ at moderate load, so a low τ preserving sensitivity (≈0.5–0.6) is best.
3. **Encryption + latent decryption** — Section 3.4 now clarifies: encryption is end-to-end between the clinic and the diagnostic centre, so the receiving server holds the key, decrypts on arrival, and runs triage and (optional) reconstruction on the plaintext; relays only carry ciphertext. Compressed-domain triage at the receiver is unaffected. Triage at an untrusted relay would need a trusted relay or privacy-preserving computation, which we do not pursue.

---
Tables (1–17), figures (1–10), and citations resolve; novelty claims remain hedged ("to the best of our knowledge"); no number is unsupported by a result file. Both additional experiments (`tau_sweep.py` → Figure 11; `bach_stain_macenko.py`) have been run and integrated with measured numbers.
