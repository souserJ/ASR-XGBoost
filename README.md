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
python asr_demo.py --study                    # simulation study: 3 difficulty levels g_clean,g_mid,g_hard × 3 partitions × 2 seeds = 18 runs (per-simulation cache)
python asr_demo.py --study --gens g_clean     # easy only: low noise
python asr_demo.py --study --gens g_mid       # medium only: full-block σ≈0.22 moderate noise
python asr_demo.py --study --gens g_hard      # hard only: weak signal + high noise
python asr_demo.py --study --gens g_clean,g_mid,g_hard --reps 10   # all three levels, paired significance (recommended)
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

Design: default **three difficulty levels (g_clean / g_mid / g_hard) × 3 block partitions × 2 seeds = 18 simulations** (`--gens` to select a single difficulty or add other generators). Each simulation reports both **test-set** metrics (AUC / Brier / Recall plus spatial metrics Moran's I / ContED / Iso ratio, matching the indicator rows of the paper's External validation table) and **final-style spatial CV** metrics (3×3 grid-block folds, blocks never split across folds, each fold retrains and recomputes soft labels/gating, no leakage; ASR uses **training-side preselected λ\*** by default (fair version, same protocol as the test set)), then aggregates mean±std and reports the average ASR improvement over CE. Additionally reports **subset gains**: the ASR−CE improvement restricted to pixels where λ\*>0 (where ASR is actually active) and to the CE∈[0.4,0.6] risk band (strongest gating). Naming matches the paper: SR(λ=1.0) (fixed strength), ASR (block-wise adaptive).

- Data generators (by difficulty): `g_clean` (**easy**: low noise) / `g_mid` (**medium**: full-block σ≈0.22 moderate noise) / `g_hard` (**hard**: weak signal logit×0.45 + σ≈0.35, CE predictions largely near 0.5, amplifying the ASR gating gain); other generators: `g_noisy` (high under-reporting heterogeneity) / `g_noisy2` (full-block high noise) / `g_barrier` (strong terrain barrier) / `g_lgcp` (point-process LGCP)
- Block partitions: `p_voronoi` (Voronoi random "country" blocks) / `p_grid` (regular grid blocks, corresponding to the 5°×5° grid convention) / `p_resist` (ecological resistance partitions)
- Output: per-simulation progress (cache/rerun markers) → test-set summary table → spatial CV summary table → mean improvement → subset gains → paired significance (per-simulation Δ = ASR−CE, paired t + Wilcoxon) → generator × partition combination table (test set + spatial CV) → λ\* selection distribution
- **Cache**: per-simulation results stored in `study_cache.pkl` (keys include all parameters; parameter changes invalidate automatically); reruns hit the cache in seconds; `--no-cache` forces a full rerun

Example output (3 difficulties × 3 partitions × 10 seeds = 90 simulations, both test-set and spatial-CV protocols):

```
Simulation study summary: 3 data generators × 3 partitions × 10 seeds = 90 simulations
------------------------------------------------------------------
Test-set metrics (7:2:1 split, ASR uses training-side block λ*; mean±std):
     Metric      |         CE         |     SR(λ=1.0)      |        ASR
     AUC         |   0.7261±0.0903    |   0.7286±0.0897    |   0.7304±0.0883
    Brier        |   0.2065±0.0301    |   0.2053±0.0300    |   0.2048±0.0298
    Recall       |   0.6568±0.0971    |   0.6596±0.0977    |   0.6617±0.0970
  Moran's I      |   0.8669±0.0718    |   0.8970±0.0535    |   0.9117±0.0385
  Cont. ed.      |   0.0763±0.0069    |   0.0662±0.0081    |   0.0617±0.0105
  Iso ratio      |   0.4425±0.0353    |   0.3730±0.0493    |   0.3340±0.0700
------------------------------------------------------------------
Spatial CV (3×3 grid-block folds, blocks never split across folds, retrain + recompute soft labels per fold, ASR uses training-side preselected λ* (fair version)):
     Metric      |         CE         |     SR(λ=1.0)      |        ASR
     AUC         |   0.6938±0.0841    |   0.6980±0.0837    |   0.6996±0.0822
    Brier        |   0.2109±0.0283    |   0.2092±0.0283    |   0.2086±0.0279
    Recall       |   0.6194±0.0910    |   0.6217±0.0931    |   0.6214±0.0936
------------------------------------------------------------------
Paired significance (per-simulation Δ = ASR − CE; paired t + Wilcoxon, n=90):
  SpatialCV AUC      Δ+0.0058±0.0035  t=+15.52  p<0.0001  Wilcoxon p<0.0001
  SpatialCV Brier    Δ-0.0022±0.0010  t=-20.47  p<0.0001  Wilcoxon p<0.0001
  Test Recall        Δ+0.0049±0.0167  t=+2.75  p=0.0072  Wilcoxon p=0.0080
  Test Iso ratio     Δ-0.1085±0.0594  t=-17.25  p<0.0001  Wilcoxon p<0.0001
------------------------------------------------------------------
By-generator accuracy (across 3 partitions × 10 seeds; each cell CE / SR(λ=1.0) / ASR):
   Generator   |          AUC(CV)            |          Brier(CV)           |          Recall(CV)          |          Iso(test)
   g_clean     |     0.7800/0.7837/0.7837     |     0.1797/0.1779/0.1778     |     0.7121/0.7158/0.7147     |     0.4409/0.3931/0.3832
    g_mid      |     0.7157/0.7199/0.7211     |     0.2073/0.2058/0.2053     |     0.6273/0.6305/0.6311     |     0.4493/0.3854/0.3551
    g_hard     |     0.5858/0.5903/0.5939     |     0.2457/0.2439/0.2428     |     0.5189/0.5188/0.5185     |     0.4374/0.3404/0.2636
------------------------------------------------------------------
Training-side block-wise adaptive λ* selection distribution (all blocks, all simulations):
  λ=0.0: 5 blocks, λ=0.5: 16 blocks, λ=1.0: 31 blocks, λ=1.5: 43 blocks, λ=2.0: 47 blocks,
  λ=2.5: 27 blocks, λ=3.0: 8 blocks, λ=3.5: 3 blocks
------------------------------------------------------------------
Key points: all models use 7:2:1 split + validation early stopping (max 1000 rounds, consistent with the paper).
The ASR gain over CE increases monotonically with difficulty (spatial-CV ΔAUC +0.0037 / +0.0054 /
+0.0081, all paired tests significant), with no degradation in discriminative accuracy; spatial
metrics (Moran's I / ContED / Iso ratio, consistent with the paper) improve across the board
(isolated-pixel ratio drops 17.4 percentage points in the hard setting); gating-band gains
CE∈[0.4,0.6] are ~4–5× the global gain (test ΔAUC 0.0238 vs 0.0044; spatial CV 0.0245 vs 0.0058,
validating the gating design); λ* adapts to data difficulty (means 1.23 / 1.60 / 2.17).

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
