# HouseTS: A Large-Scale, Multimodal Spatiotemporal U.S. Housing Dataset + Benchmark

This repository contains the  **benchmark** for **HouseTS**, a large-scale multimodal spatiotemporal dataset for long-horizon housing-market forecasting at the U.S. ZIP-code level.

HouseTS aligns multiple modalities under a unified ZIP-month panel, including:
- **Monthly housing-market indicators**
- **Monthly POI counts**
- **Annual census / socioeconomic variables** aligned to the monthly timeline
- (Dataset also includes auxiliary modalities such as aerial imagery + derived annotations; see Kaggle for full contents.)

The benchmark supports **univariate** and **multivariate** forecasting with standardized train/val/test splitting, windowing, transforms, and evaluation.

---

## Dataset

HouseTS data (tabular signals) is available via Google Drive:

- Google Drive download: https://drive.google.com/file/d/1OC_PTXfaGuQ50-mu2LkfQRLdhjPUbyu7/view?usp=sharing

HouseTS aerial imagery data is hosted on Kaggle:

- Kaggle dataset page: https://www.kaggle.com/datasets/shengkunwang/housets-dataset

### Expected local path

By default, the benchmark expects:

- `data/raw/HouseTS.csv`

You can also point to `.csv`, `.parquet`, or `.xlsx` via config/CLI.

### Minimal schema

Your tabular file should include at least:
- `zipcode` (ZIP code; will be normalized to a 5-digit string)
- `date` (timestamp; will be parsed)
- `price` (forecast target; default `data.target_col`)

All other numeric columns are treated as continuous covariates for multivariate settings.

> Notes:
> - Non-feature columns like `city` / `city_full` (if present) are dropped by default.
> - The loader adds `year` and `month` time markers from `date`.
> - Missing values are handled with a benchmark imputation routine.


## Quick start

All examples below are run from the repository root.

### 1) Run a single experiment (config-driven)

The config runner merges:
- `configs/default.yaml`
- `configs/task/<task>.yaml`
- `configs/windows/<window>.yaml`
- `configs/models/<model>.yaml`

Example (multivariate, window `w6_h3`, model `dlinear`):

```bash
python scripts/run_one.py \
  --task multivariate \
  --window w6_h3 \
  --model dlinear \
  --data data/raw/HouseTS.csv \
  --device gpu
```
### 2) Run a univariate baseline

```bash
python scripts/run_one.py \
  --task univariate \
  --window w12_h6 \
  --model ar_univariate \
  --data data/raw/HouseTS.csv \
  --device cpu
```

---

## Window Presets

The repository currently provides the following window presets:

- `w6_h3`
- `w6_h6`
- `w6_h12`
- `w12_h3`
- `w12_h6`
- `w12_h12`

For example:

- `w6_h3`: `seq_len=6`, `label_len=3`, `pred_len=3`
- `w12_h6`: `seq_len=12`, `label_len=6`, `pred_len=6`

---

## Supported Model Configs

The current `configs/models/` directory includes the following model configs.

### Statistical baselines

- `ar_univariate`
- `ardl`
- `arima`
- `var`
- `var_ms`

### Classical machine learning

- `rf`
- `xgb`

### Deep learning

- `rnn`
- `lstm`
- `dlinear`
- `timemixer`
- `patchtst`
- `informer`
- `autoformer`
- `fedformer`

### Graph neural networks

- `gcn_tcn`
- `graph_wavenet`
- `stgcn`
- `stsgcn`

### Foundation-model variants

- `chronos2_zero`
- `chronos2_ft`
- `timesfm_xreg_zero`
- `timesfm_xreg_ft`

---

---

## GNN models: dataloader structure

GNN models (GCN-TCN, STGCN, GraphWaveNet, STSGCN) require a geographic
lat/lon file to build the k-NN graph adjacency.  The expected path is:

```
data/raw/zip_latlon.csv   # columns: zipcode, lat, lon
```

Override the path per model via `--set model.hparams.latlon_path=<path>`.

### Why GNN dataloaders are different

| | DL models | GNN models |
|---|---|---|
| **Unit of one sample** | one ZIP × one time window | one time window × **all N ZIPs** |
| **Batch shape** | `[B, L, Dx]` — B mixes ZIPs and time positions | `[B, L, N, Dx]` — B is time positions only, N always equals total ZIPs |
| **Spatial coupling** | None — each ZIP is processed independently | Full — message-passing across N geographic neighbors per step |
| **Batch size meaning** | number of (ZIP, window) pairs | number of time windows (all N nodes included in each) |

**DL dataloader** (`WindowDataset`): generates one `(zip_i, t₀)` anchor per item.
The DataLoader collects B such anchors into a tensor `[B, L, Dx]`.
Spatial information across ZIPs is entirely absent; each row is independent.

**GNN dataloader** (`GraphWindowDataset`): generates one `t₀` anchor per item —
but returns the feature matrix for **all N nodes at that time step**.
The batch tensor `[B, L, N, Dx]` lets the network perform graph message-passing
across the N-dimension, so every ZIP can receive information from its
geographic neighbors.

After the GNN forward pass the output `[B, H, N, Dy]` is reshaped to
`[B×N, H, Dy]` so the standard `StreamingEvaluator` receives the same
`(n_samples, horizon, features)` format it expects from DL models.

### STSGCN

STSGCN (Wu et al., AAAI 2020, [code](https://github.com/Davidham3/STSGCN))
differs from STGCN in that it captures spatial and temporal dependencies
**synchronously** in a single graph operation rather than in separate sequential
stages.

It constructs a spatial-temporal synchronous adjacency
`A_st ∈ R^{T_local·N × T_local·N}` by stacking `T_local` copies of the spatial
graph on the diagonal and adding identity connections between consecutive steps:

```
A_st = [ A_s  I    0  ]
       [ I    A_s  I  ]   (T_local = 3)
       [ 0    I    A_s]
```

A GCN applied to the flattened `T_local·N`-node graph then aggregates across
both the spatial and temporal axes in one pass.  Each prediction step has its
own independent STSGCM branch, extracting the representation at the centre
time step.

---

## Data Usage and Attribution

HouseTS integrates or aligns signals derived from several public data sources, including:

- housing-market time series
- OpenStreetMap-derived POI statistics
- U.S. Census / ACS socioeconomic variables
- USDA NAIP aerial imagery

Please review the paper and the upstream data-source licensing / attribution requirements before redistribution, publication of derivatives, or commercial use.

---

## Citation

If you use HouseTS or this benchmark code in your research, please cite:

```bibtex
@article{wang2025housets,
  title={HouseTS: A Large-Scale, Multimodal Spatiotemporal U.S. Housing Dataset and Benchmark},
  author={Wang, Shengkun and Sun, Yanshen and Chen, Fanglan and Wang, Linhan and Ramakrishnan, Naren and Lu, Chang-Tien and Chen, Yinlin},
  journal={arXiv preprint arXiv:2506.00765},
  year={2025}
}
```

---
