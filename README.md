# MBAge-IJCB2026 — 1st Place Solution

A hierarchical multimodal ensemble for tongue and face age estimation, winner of the IEEE IJCB 2026 MBAge Challenge ([CodaBench #15116](https://www.codabench.org/competitions/15116/)).

> **Karthik Sivarama Krishnan** &middot; **Koushik Sivarama Krishnan** &middot; Independent Researchers

## Result

Competition metric: `S = 0.1/(1+MSE) + 0.4·Acc@1Y + 0.3·Acc@5Y + 0.2·Acc@10Y`.
Final combined score: `S_final = 0.4·S_phase1 + 0.6·S_phase2`.

**Final Combined Leaderboard (top 3):**

| Rank | Participant | Combined |
|------|-------------|---------:|
| 🥇 **1st** | **ks7585 (this work)** | **0.4129** |
| 🥈 2nd | hqp001 | 0.4102 |
| 🥉 3rd | cuinuan | 0.4063 |

**Per-phase results (this work):**

| Phase | Rank | S | MSE | 1Y-ACC | 5Y-ACC | 10Y-ACC |
|-------|------|---|-----|--------|--------|---------|
| Phase 1 (tongue only) | 2nd | 0.3727 | 97.17 | 0.16 | 0.53 | 0.74 |
| **Phase 2 (tongue + face)** | **1st** | **0.4396** | **46.92** | **0.176** | **0.648** | **0.864** |

## Method Overview

Three complementary models are weighted-averaged. Each is trained on the competition-provided pre-extracted tongue features (four CNN backbones at 28&times;28 spatial resolution) and face Gabor features (banks of 5 scales &times; 8 orientations across four face regions).

```
                                              ┌─────────────────────┐
   Tongue features (4 CNN backbones)   ──┐    │  Final prediction   │
                                          ├──>│  ŷ = clip(           │
   Face Gabor features (4 banks)       ──┘    │     0.400·hier +    │
                                              │     0.425·s12 +     │
   Component 1: hier_s42   (40.0%) ──────────>│     0.175·film6_hier│
   Component 2: s12_all    (42.5%) ──────────>│  , 20, 69)           │
   Component 3: film6_hier (17.5%) ──────────>│                     │
                                              └─────────────────────┘
```

**hier_s42** — Hierarchical decade-then-fine DLDL. A five-way decade classifier (20s–60s) gates ten-bin within-decade DLDL heads, with predictions computed by soft hierarchical routing. Trained with `train_phase2_hier.py`.

**s12_all** — Dual-backbone (DenseNet121 + ResNet18) hierarchical model with all fourteen feature engineering techniques: pre-pool LayerNorm, std + 2×2 grid pooling, DCT and Haar projections, higher-order and spatial moments, Gabor-bank marginals/asymmetry, iSQRT-COV, Compact Bilinear Pooling (dim 1024), mutual-information channel selection (top-64), Efficient Channel Attention, CKA backbone de-duplication, learned-query attention pooling, confidence-weighted late fusion, MMTM cross-backbone fusion. Trained with `train_sprint12.py`.

**film6_hier** — Median of seven models: six FiLM-conditioned transformers trained with seeds {0, 1, 2, 3, 4, 42} (triple-head decoder combining DLDL, regression, and ordinal predictions) plus the hier_s42 model. FiLM projects face features into per-channel affine parameters that modulate tongue features. Trained with `train_phase2_v3.py`.

**Blend weights** were chosen by exhaustive grid search at 0.025 resolution over the ternary simplex (114 candidates), retaining the smallest deviation from the canonical 40/40/20 weighting that yielded a positive jittered out-of-fold score delta with no negative per-metric component. This refinement contributed +0.00022 to the final leaderboard score.

## Repository Layout

```
scripts/
  train_phase2_hier.py        # Component 1: hierarchical decade+fine DLDL
  train_sprint12.py           # Component 2: 14-flag feature-engineered model
  train_phase2_v3.py          # Component 3: FiLM-conditioned transformer
  build_selective_channels.py # Optional: channel ranking helper
  build_final_blend.py        # Assemble the final submission ZIP
server_preds/
  hier_repro_s42_{oof,test}.npy        # Out-of-fold + test predictions
  film_s{0,1,2,3,4,42}_repro_{oof,test}.npy
  s12_all_repro_test.csv               # s12_all writes CSV instead of npy
  s12_all_ages.npy                     # ground-truth training ages
  test_uuids.txt                       # canonical 477-row test order
submissions/
  v6_grid_closest_to_ref.zip           # final LB=0.4396 submission
  v6_grid_closest_to_ref.csv           # CSV form
methodology.pdf                        # one-page report (IJCB submission)
```

## Reproduction

### Requirements

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 torchvision numpy scipy scikit-learn pandas
```

Training requires a single GPU. The full pipeline was trained on an NVIDIA RTX 5070 (12 GB) in approximately 3.5 hours.

### Data

Download the competition feature files from CodaBench and place them in `data/`:

```
data/
  tongue_ResNet18_features_v2.0.pkl       (1.3 GB)
  tongue_DenseNet121_features_v2.0.pkl    (1.3 GB)
  tongue_MobileNetV2_features_v2.0.pkl    (300 MB)
  tongue_EfficientNetB0_features_v2.0.pkl (400 MB)
  face_features_gabor_v2.0.pkl            (2 MB)
```

### Train the eight component models

```bash
# Component 1: hierarchical decade+fine model (one seed)
python scripts/train_phase2_hier.py --data_dir data --exp_name repro_s42 --seed 42

# Component 2: feature-engineered model with all 14 flags
python scripts/train_sprint12.py --data_dir data --exp_name s12_all_repro --seed 42 \
  --face_banks ABCD --d_hidden 384 --dropout 0.35 --patience 35 \
  --use_layernorm_pre --use_std_pool --use_grid_pool \
  --use_dct --use_haar --use_higher_moments --use_spatial_moments \
  --use_gabor_marginals --use_gabor_asymmetry \
  --use_isqrt_cov --isqrt_reduce_dim 32 \
  --use_cbp --cbp_dim 1024 --use_mi_selection --mi_top_k 64 \
  --use_eca --use_cka_dedup --use_learned_query --use_confidence_fusion \
  --fusion_type mmtm

# Component 3: six FiLM-conditioned models
for SEED in 0 1 2 3 4 42; do
  python scripts/train_phase2_v3.py --data_dir data --exp_name repro_film_s${SEED} \
    --face_mode film --triple_head \
    --wing_weight 0.5 --metric_weight 1.0 --metric_start 20 \
    --w_kl 0.3 --w_l1 0.5 --w_var 0.2 --seed $SEED
done
```

### Build the final submission

```bash
python scripts/build_final_blend.py
# → submissions/v6_grid_closest_to_ref.zip
```

The included `server_preds/` already contains the out-of-fold and test predictions from our training run, so the blend can be reproduced without re-training.

## Reproducibility Verification

The hierarchical model is **bit-exact reproducible** across different GPUs and CUDA versions. The out-of-fold predictions hash to MD5 `9ee818c7a9557ea0852ff9634bd62227` on both the original training run and an independent retraining on different hardware (PyTorch 2.11.0+cu128, RTX 5070 vs. RTX 5090).

## Citation

If you use this code, please cite the competition summary paper (forthcoming) and this repository:

```
Karthik Sivarama Krishnan and Koushik Sivarama Krishnan.
MBAge-IJCB2026: A Hierarchical Multimodal Ensemble for Tongue and Face Age Estimation.
1st Place Solution to the IEEE IJCB 2026 MBAge Challenge, 2026.
https://github.com/ks7585/MBAge-IJCB2026
```

## Documents

- [`methodology.pdf`](methodology.pdf) — one-page methodology summary submitted to the IJCB 2026 organizers
- [`methodology.tex`](methodology.tex) — LaTeX source for the above (compile with `pdflatex` on Overleaf or locally)

## License

Released under the MIT License. See [LICENSE](LICENSE).
