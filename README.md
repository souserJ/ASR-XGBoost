[English](README.md) | [中文](README.zh-CN.md)

# ASR-XGBoost (Demo)

A **minimal end-to-end demo** of the **A**daptive **S**patial **R**egularization **XGBoost** core idea, featuring **block-wise adaptive λ selection** and a **multi-scenario simulation study**.

Building on standard pixel-wise XGBoost, spatial continuity is embedded directly into the training objective: neighborhood soft labels constrain each pixel's prediction, and a **gating function** concentrates the regularization strength where the model is most uncertain (predicted probability near 0.5). This repository further demonstrates:

- **Connectivity-weighted soft labels**: neighborhood weights defined by a resistance surface, so pixels on opposite sides of a terrain barrier are no longer treated as "neighbors";
- **Block-wise adaptive λ**: the plane is partitioned into "country" blocks (or blocks you draw yourself); each block selects its optimal regularization strength λ\* by **within-block k-fold cross-validation**, and training uses a per-pixel λ map;
- **Multi-scenario simulation study**: multiple data generators × multiple block partitions × multiple seeds, aggregated as mean±std.

> This is a public "small but real" demo: synthetic data, single file, runs in seconds. The full version (real data, production-grade implementation) is used for the academic paper and is not released.

## Data Generation

| Component | Method |
|---|---|
| Environmental fields X1~X3 | **GSTools** Matern Gaussian random fields, **nested dual-scale** (long-range trend + local variation), 3 covariates |
| Resistance surface R | Terrain ridges + noise; **sinuous coastline** (superposed sinusoids, resistance = 100 absolute barrier) |
| Suitability p_true | Covariate response functions + **interaction term** (nonlinear spatial structure); superimposed fragment patches for learnable sharp edges |
| Observed labels y | ① Bernoulli occurrence sampling + block-wise detection/reporting heterogeneity noise (**proxy for under-reporting**); ② point-process pattern: **LGCP/inhomogeneous Poisson** (pixel event count ~ Poisson(λ(s)); occurrence = at least 1 event) |

## Core Method

Training objective:

$$
\mathcal{L}_{\mathrm{ASR}} = \mathcal{L}_{\mathrm{CE}} + \lambda_i \cdot \delta(\hat p)\,(p - \tilde p)^2
$$

- $p$: model output probability; $\tilde p$: soft label (neighborhood weighted mean)
- Gating function $\delta(\hat p) = 1 - 2|\hat p - 0.5|$: the closer the prediction is to 0.5 (more uncertain), the stronger the regularization; computed from the baseline prediction $\hat p$ and **frozen** (frozen gating), guaranteeing per-sample Hessian positive definiteness
- Soft label $\tilde p_i = \dfrac{\sum_j w_{ij}\,\hat p_j}{\sum_j w_{ij}}$, connectivity weight $w_{ij} = \exp\!\big(-\beta \cdot (R_i + R_j)/2\big)$
- Block-wise adaptive: $\lambda_i = \lambda^*_{\text{block}(i)}$, each block's $\lambda^*$ selected by **within-block k-fold CV** with the **1-SE rule** (smallest λ within one standard error of the optimal λ, glmnet convention) — automatically picks the minimal sufficient regularization on flat curves, avoiding random tie-breaking and over-smoothing; candidate range 0–3.5 (λ=0 = no regularization, equivalent to CE; 3.5 = upper sensitivity bound in the paper); small regions are merged by **centroid nearest-neighbor** (corresponding to "small-country merging" of administrative-block strategy B, ensuring stable within-block CV)
- Data split: total pixels randomly sampled via `--sample` (default **40%**) as the dataset (simulating sparse observation); within the dataset, split via `--split` (default **7:2:1**, consistent with the paper); the test set does not participate in any selection

## Files

| File | Description |
|---|---|
| `asr_demo.py` | Main script: single detailed demo + `--study` multi-scenario simulation study |
| `draw_regions.py` | Interactive region-drawing script (mouse polygon), outputs `regions.npy` |
| `requirements.txt` | numpy / xgboost / matplotlib / gstools |

## Usage

```bash
pip install -r requirements.txt

python asr_demo.py                          # single detailed demo (300×300, 16 blocks, 40% sampling, spatial CV evaluation and 6-panel figure: truth/blocks/λ*/CE/ASR/difference)
python asr_demo.py --study                    # simulation study: 3 difficulty levels × 3 partitions × 2 seeds = 18 runs (defaults: 16 blocks, min-pixels 150; per-simulation cache)
python asr_demo.py --study --gens g_clean     # easy only: low noise
python asr_demo.py --study --gens g_mid       # medium only: full-block σ≈0.22 moderate noise
python asr_demo.py --study --gens g_hard      # hard only: weak signal + high noise
python asr_demo.py --study --reps 10   # formal protocol (16 blocks / min-pixels 150 are the defaults), paired significance (recommended)
python asr_demo.py --study --reps 3           # increase number of seeds
python asr_demo.py --study --no-cache         # ignore cache, force full rerun
python asr_demo.py --blocks regions.npy     # use your own drawn regions (single-run mode)
```

**Draw your own regions** (Windows GUI):

```bash
python draw_regions.py
# left-click: add point | right-click: close a block | c: undo | q: save and exit
python asr_demo.py --blocks regions.npy
```

## Simulation Study (--study)

Design: default **three difficulty levels (g_clean / g_mid / g_hard) × 3 block partitions × 2 seeds = 18 simulations**, with **16 spatial blocks and `--min-pixels 150`** as the formal defaults (`--gens` to select a single difficulty or add other generators). Each simulation reports both **test-set** metrics (AUC / Brier / Recall plus spatial metrics Moran's I / ContED / Iso ratio, matching the indicator rows of the paper's External validation table) and **final-style spatial CV** metrics (3×3 grid-block folds, blocks never split across folds, each fold retrains and recomputes soft labels/gating, no leakage; ASR uses **training-side preselected λ\*** by default (fair version, same protocol as the test set)), then aggregates mean±std and reports the average ASR improvement over CE. Additionally reports **subset gains**: the ASR−CE improvement restricted to pixels where λ\*>0 (where ASR is actually active) and to the CE∈[0.4,0.6] risk band (strongest gating). Naming matches the paper: SR(λ=1.0) (fixed strength), ASR (block-wise adaptive).

- Data generators (by difficulty): `g_clean` (**easy**: low noise) / `g_mid` (**medium**: full-block σ≈0.22 moderate noise) / `g_hard` (**hard**: weak signal logit×0.45 + σ≈0.35, CE predictions largely near 0.5, amplifying the ASR gating gain); other generators: `g_noisy` (high under-reporting heterogeneity) / `g_noisy2` (full-block high noise) / `g_barrier` (strong terrain barrier) / `g_lgcp` (point-process LGCP)
- Block partitions: `p_voronoi` (Voronoi random "country" blocks) / `p_grid` (regular grid blocks, corresponding to the 5°×5° grid convention) / `p_resist` (ecological resistance partitions)
- Output: per-simulation progress (cache/rerun markers) → test-set summary table → spatial CV summary table → mean improvement → subset gains → paired significance (per-simulation Δ = ASR−CE, paired t + Wilcoxon) → generator × partition combination table (test set + spatial CV) → λ\* selection distribution
- **Cache**: per-simulation results stored in `study_cache.pkl` (keys include all parameters; parameter changes invalidate automatically); reruns hit the cache in seconds; `--no-cache` forces a full rerun

Example output (formal protocol: **16 blocks / min-pixels 150**; 3 difficulties × 3 partitions × 10 seeds = 90 simulations, both test-set and spatial-CV protocols):

```
Simulation study summary: 3 data generators × 3 partitions × 10 seeds = 90 simulations (16 blocks, min-pixels 150)
------------------------------------------------------------------
Test-set metrics (7:2:1 split, ASR uses training-side block λ*; mean±std):
     Metric      |         CE         |     SR(λ=1.0)      |        ASR
     AUC         |   0.7229±0.0940    |   0.7259±0.0932    |   0.7276±0.0923
    Brier        |   0.2072±0.0311    |   0.2060±0.0308    |   0.2055±0.0306
    Recall       |   0.6556±0.0968    |   0.6558±0.0969    |   0.6590±0.0990
  Moran's I      |   0.8667±0.0711    |   0.8966±0.0529    |   0.9109±0.0408
  Cont. ed.      |   0.0764±0.0073    |   0.0662±0.0083    |   0.0617±0.0103
  Iso ratio      |   0.4412±0.0379    |   0.3730±0.0501    |   0.3338±0.0695
------------------------------------------------------------------
Spatial CV (3×3 grid-block folds, blocks never split across folds, retrain + recompute soft labels per fold, ASR uses training-side preselected λ* (fair version)):
     Metric      |         CE         |     SR(λ=1.0)      |        ASR
     AUC         |   0.6907±0.0800    |   0.6947±0.0793    |   0.6966±0.0777
    Brier        |   0.2121±0.0270    |   0.2105±0.0268    |   0.2099±0.0263
    Recall       |   0.6163±0.0867    |   0.6178±0.0893    |   0.6178±0.0889
------------------------------------------------------------------
Paired significance (per-simulation Δ = ASR − CE; paired t + Wilcoxon, n=90):
  SpatialCV AUC      Δ+0.0060±0.0035  t=+15.86  p<0.0001  Wilcoxon p<0.0001
  SpatialCV Brier    Δ-0.0022±0.0011  t=-18.11  p<0.0001  Wilcoxon p<0.0001
  Test Iso ratio     Δ-0.1075±0.0566  t=-17.91  p<0.0001  Wilcoxon p<0.0001
------------------------------------------------------------------
By-generator accuracy (across 3 partitions × 10 seeds; each cell CE / SR(λ=1.0) / ASR):
   Generator   |          AUC(CV)            |          Brier(CV)           |          Recall(CV)          |          Iso(test)
   g_clean     |     0.7672/0.7709/0.7711     |     0.1836/0.1820/0.1821     |     0.7004/0.7055/0.7027     |     0.4405/0.3968/0.3802
    g_mid      |     0.7180/0.7214/0.7229     |     0.2069/0.2056/0.2050     |     0.6281/0.6267/0.6287     |     0.4556/0.3894/0.3644
    g_hard     |     0.5868/0.5917/0.5958     |     0.2457/0.2439/0.2426     |     0.5202/0.5213/0.5218     |     0.4277/0.3328/0.2568
------------------------------------------------------------------
Training-side block-wise adaptive λ* selection distribution (all blocks, all simulations):
  λ=0.0: 100 blocks, λ=0.5: 66 blocks, λ=1.0: 119 blocks, λ=1.5: 188 blocks, λ=2.0: 218 blocks,
  λ=2.5: 165 blocks, λ=3.0: 58 blocks, λ=3.5: 17 blocks
------------------------------------------------------------------
Key points: all models use 7:2:1 split + validation early stopping (max 1000 rounds, consistent with the paper).
The ASR gain over CE increases monotonically with difficulty (spatial-CV ΔAUC +0.0039 / +0.0050 /
+0.0090, all per-tier paired tests significant), with no degradation in discriminative accuracy;
spatial metrics (Moran's I / ContED / Iso ratio, consistent with the paper) improve across the board
(isolated-pixel ratio drops 17.1 percentage points in the hard setting); gating-band gains
CE∈[0.4,0.6] are ~3.5–5× the global gain (test ΔAUC 0.0252 vs 0.0050; spatial CV 0.0210 vs 0.0060,
validating the gating design); λ* adapts to data difficulty (means 1.25 / 1.44 / 2.18).

## About This Repository (Demo Positioning)

| Aspect | This demo (public) |
|---|---|
| Data | GSTools synthetic fields + response functions + LGCP (no real disease data) |
| Connectivity cost | Local resistance approximation $(R_i+R_j)/2$ |
| Partitioning | Synthetic blocks / user-drawn regions; small regions merged by centroid nearest-neighbor; λ-only partitioning |
| Sampling | Random pixels + block-wise detection heterogeneity |
| Validation | final-style spatial CV (3×3 grid-block folds, blocks never split across folds, retrain + recompute soft labels per fold, no leakage; ASR uses training-side preselected λ*, fair version; AUC / Brier / Recall, mean±std, matching the indicator rows of the paper's External validation table) + test-set evaluation + three-difficulty simulation study (both test-set and spatial-CV protocols) |
| Structure | Single-file demo |

> This is a "small but real" public demo: synthetic data, single file, runs in seconds, intended for method demonstration and reproducible simulation-based validation; the full version (real data, production-grade implementation) is used for the academic paper and is not released.

## Citation

If you use or reference this code, please cite:

> (To be filled after publication: authors, title, journal, year, DOI)

## License

[MIT](LICENSE). Free to use as long as the copyright notice is retained; for academic use, please cite as requested above.
