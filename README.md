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

### Any dataset via a dataset config

The benchmark is dataset-agnostic: everything specific to one dataset (file path,
id/time/target columns, which columns to model, drop list, and the graph settings for GNN
models) lives in one YAML file under `configs/dataset/`. `configs/dataset/dc_house.yaml`
is the working example (matches `data/DC_House.csv`); `configs/dataset/housets.yaml` is a
template for the full HouseTS.csv.

```yaml
dataset:
  name: dc_house
data:
  path: data/DC_House.csv
  id_col: zipcode
  time_col: date
  target_col: price
  feature_cols: [median_sale_price, homes_sold, ...]  # null => auto-infer all numeric cols
  drop_cols: [city, metro, state]
  lat_col: latitude
  lon_col: longitude
  freq: M
graph:
  mode: knn            # knn | adjacency
  k: 10
  max_km: 100.0
  adjacency_path: null
```

Minimal schema requirements for a new dataset CSV: an id column, a time column (parsed as
a timestamp), and the target column. If you're using a GNN model in `knn` mode, also
include `latitude`/`longitude` columns (constant per id) — the loader reads them directly,
no separate geocoding step is needed. If `feature_cols` is set, only those columns (plus
id/time/target/lat/lon) are read and modeled — everything else in the CSV is ignored.
Missing values are handled with a benchmark imputation routine; the loader adds `year` and
`month` time markers from the time column.

Add a new dataset by copying `configs/dataset/dc_house.yaml` and pointing it at your CSV.


## Quick start

All examples below are run from the repository root.

### 1) Run a single experiment (config-driven)

The config runner merges, in order:
- `configs/default.yaml`
- `configs/dataset/<dataset>.yaml`
- `configs/task/<task>.yaml`
- `configs/windows/<window>.yaml`
- `configs/models/<model>.yaml`

Runs are written to `runs/<dataset>/<model>__<task>__<window>/`.

Example (dc_house, multivariate, window `w6_h3`, model `dlinear`):

```bash
python scripts/run_one.py \
  --dataset dc_house \
  --task multivariate \
  --window w6_h3 \
  --model dlinear \
  --device gpu
```
### 2) Run a univariate baseline

```bash
python scripts/run_one.py \
  --dataset dc_house \
  --task univariate \
  --window w12_h6 \
  --model ar_univariate \
  --device cpu
```

### 3) Cut evaluation cost on the test set

```bash
python scripts/run_one.py --dataset dc_house --model timesfm_zero --window w12_h12 \
  --test-stride 3                    # only evaluate every 3rd test window
python scripts/run_one.py --dataset dc_house --model dlinear --window w6_h3 \
  --test-cutoff-date 2022-01-01      # test starts at this exact date instead of a ratio split
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

GNN models (GCN-TCN, STGCN, GraphWaveNet, STSGCN) need a node adjacency graph. This is a
dataset-level property (not per-model), configured in the `graph:` block of the
dataset config (see `configs/dataset/dc_house.yaml`) — either mode:

- **`mode: knn`** (default) — builds a k-NN graph from the `lat_col`/`lon_col` columns
  already present in the dataset CSV (no separate geocoding step). Tune `k` / `max_km`.
- **`mode: adjacency`** — loads a precomputed `[N, N]` adjacency matrix directly from
  `adjacency_path` (`.npy`, or `.csv` with a header/index of ids that get reindexed to
  match the dataset's ZIP ordering). `N` must equal the number of nodes actually used by
  the run (i.e. after any `--n-zip` subsampling).

```bash
python scripts/run_one.py --dataset dc_house --model gcn_tcn --window w6_h6 \
  --set graph.mode=adjacency --set graph.adjacency_path=data/dc_house_adj.npy
```

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
