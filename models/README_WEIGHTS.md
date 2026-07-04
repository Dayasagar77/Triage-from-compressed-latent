# Model weights

The trained weights (~212 MB) are **not** stored in git (one file exceeds
GitHub's 100 MB limit). Download them and place the `.pth` files in this `models/`
folder:

| File | Size | Description |
|------|------|-------------|
| `final_cancer_model.pth` | 104 MB | ResNet-50 + SE source-side classifier (CancerNet) |
| `learned_codec_l{0.002,0.01,0.05}.pth` | 12 MB each | Factorized-prior codec, three rates |
| `hyperprior_l{0.002,0.01,0.05}.pth` | 20 MB each | Scale-hyperprior codec, three rates |
| `clinical_encoder.pth`, `clinical_decoder.pth` | ~8 MB each | Auxiliary reconstruction models |

**Download:** <ADD ZENODO DOI OR RELEASE LINK HERE>

All weights can also be regenerated from scratch with the training scripts
(`train.py`, `train_compression_v2.py`, `train_hyperprior_pure.py`).
