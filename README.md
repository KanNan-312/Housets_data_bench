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
  drop_cols: [city, metro, state, latitude, longitude]
  freq: M
graph:
  path: null            # graph.npz for GNN models — see "GNN models" section below
```

Minimal schema requirements for a new dataset CSV: an id column, a time column (parsed as
a timestamp), and the target column. If `feature_cols` is set, only those columns (plus
id/time/target) are read and modeled — everything else in the CSV is ignored. Missing
values are handled with a benchmark imputation routine; the loader adds `year` and `month`
time markers from the time column. The benchmark itself never reads lat/lon or builds a
graph — see "GNN models" below for how the graph is supplied.

Add a new dataset by copying `configs/dataset/dc_house.yaml` and pointing it at your CSV.


## Quick start

All examples below are run from the repository root.

### 1) Run a single experiment (config-driven)

The config runner merges, in order:
- `configs/default.yaml`
- `configs/dataset/<dataset>.yaml`
- `configs/task/<task>.yaml`
- `configs/models/<model>.yaml`

`run_one.py` itself never sets defaults for window/split values — it only overrides the
merged YAML config when a flag is explicitly passed. So the window shape (lookback/horizon)
and split (train/val ratio, test cutoff) are configured either in `configs/default.yaml`
(the global default) or per-dataset in `configs/dataset/<name>.yaml` (add a `window:`/`split:`
block there to override for just that dataset), and the CLI flags below are for one-off
overrides on top of whichever config is in effect.

Runs are written to `runs/<dataset>/<model>__<task>__<window>/`, where `<window>` is derived
from the effective `seq_len`/`pred_len` (e.g. `w12_h6`).

Example (dc_house, multivariate, model `dlinear`, using whatever window/split configs say):

```bash
python scripts/run_one.py \
  --dataset dc_house \
  --task multivariate \
  --model dlinear \
  --device gpu
```
### 2) Run a univariate baseline with a specific window shape

```bash
python scripts/run_one.py \
  --dataset dc_house \
  --task univariate \
  --seq-len 12 --label-len 6 --pred-len 6 \
  --model ar_univariate \
  --device cpu
```

### 3) Configure the split and cut evaluation cost on the test set

```bash
python scripts/run_one.py --dataset dc_house --model timesfm_zero \
  --seq-len 12 --pred-len 12 \
  --test-stride 3                    # only evaluate every 3rd test window
python scripts/run_one.py --dataset dc_house --model dlinear \
  --seq-len 6 --label-len 3 --pred-len 3 \
  --test-cutoff-date 2022-01-01      # test starts at this exact date instead of a ratio split
python scripts/run_one.py --dataset dc_house --model dlinear \
  --train-ratio 0.8 --val-ratio 0.1  # override the train/val split of the remaining (pre-test) span
```

---

## Window and split configuration

Window shape (`seq_len`/`label_len`/`pred_len`, i.e. lookback/decoder-label/horizon lengths)
and the train/val/test split live directly in config — there is no fixed set of window
presets to choose from. `configs/default.yaml` sets the global defaults:

```yaml
split:
  train_ratio: 0.7        # fraction of the pre-test span used for training
  val_ratio: 0.1          # fraction of the pre-test span used for validation
  test_start_date: null   # e.g. "2022-01-01" — overrides ratio-based test-set boundary

window:
  seq_len: 12             # lookback length
  label_len: 6            # decoder label length
  pred_len: 12            # forecast horizon
  test_stride: 1          # >1 strides through non-overlapping test windows to cut eval cost
```

Override either block for a specific dataset by adding a `window:`/`split:` block to its
`configs/dataset/<name>.yaml`, or override individual values from the CLI with
`--seq-len`/`--label-len`/`--pred-len`/`--test-stride`/`--train-ratio`/`--val-ratio`/
`--test-cutoff-date` (each only takes effect if explicitly passed), or with
`--set window.seq_len=12` / `--set split.train_ratio=0.8` for anything else.

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
- `stllm_plus`
- `dcrnn` — diffusion-convolutional seq2seq (Li et al., ICLR 2018), direct port
- `stgformer` — spatiotemporal graph transformer (Dreamzz5/STGformer), direct port
- `d2stgnn` — decoupled dynamic STGNN (VLDB 2022), direct port (dynamic graph is computed
  internally each forward pass; time-of-day/day-of-week features are omitted since this
  benchmark's datasets are monthly)
- `cast` — causal spatio-temporal representation learning (yutong-xia/CaST), direct port
  (self-discovers pseudo-environments via a VQ codebook — no external environment labels needed)
- `stexplainer` — explainable STGNN (HKUDS/STExplainer), run **forecast-only**: its GIB
  structure-distillation term is kept as an internal training regularizer, but the learned
  explanation mask isn't surfaced anywhere since this benchmark has no explanation-quality
  scoring path
- `aist` — **simplified** port of the attention-based interpretable crime model
  (YeasirRayhanPrince/aist): keeps only the graph-attention mechanism over crime counts,
  batched over all nodes jointly; drops the paper's required taxi/POI/street-crime data and
  its per-region training loop
- `st_hhol` — **simplified, static-graph** port inspired by ST-HHOL's hierarchical hypergraph
  idea: a trainable hypergraph convolution over the crime-count panel only, trained with the
  benchmark's normal offline time split; drops the paper's weather/POI/socioeconomic/311 data
  sources and its online/streaming training loop

### Foundation-model variants

- `chronos2_zero`
- `chronos2_ft`
- `timesfm_xreg_zero`
- `timesfm_xreg_ft`

---

---

## GNN models: dataloader structure

GNN models (GCN-TCN, STGCN, GraphWaveNet, STSGCN, ST-LLM+) need a node adjacency graph.
The benchmark never builds this itself (no implicit lat/lon handling) — it only loads a
precomputed graph from a `.npz` file, pointed to by `graph.path` in the dataset config
(dataset-level, not per-model, since every GNN model on a dataset shares the same graph):

```python
np.savez("graph.npz", A=A, ids=np.array(ids))
```

- `A`: dense `[N, N]` adjacency matrix.
- `ids`: length-`N` array of region ids giving `A`'s row/column order (`ids[i]` is the
  region at row/col `i`) — these must match the dataset's id column values.

At load time (`src/housets_bench/graph/loader.py`), the graph is **reindexed by matching
`ids` against the dataset's actual region ids** — not by trusting row order — so it works
correctly regardless of what order the matrix was built in, and after any `--n-zip`
subsampling (only the subsampled regions are looked up; a region missing from `ids` raises
a clear error rather than silently misaligning).

```bash
python scripts/run_one.py --dataset dc_house --model gcn_tcn --seq-len 6 --label-len 3 --pred-len 6 \
  --set graph.path=data/dc_house_graph.npz
```

If you need a geographic k-NN graph from lat/lon, `scripts/build_knn_graph.py` is a
standalone offline builder (not imported by the benchmark) that produces this `graph.npz`
format from a lat/lon CSV:

```bash
python scripts/build_knn_graph.py --input data/DC_House.csv \
  --id-col zipcode --lat-col latitude --lon-col longitude \
  --k 10 --max-km 100 --out data/dc_house_graph.npz
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

## Case library: per-instance best model + neighbor explainability

Once you have trained checkpoints under `runs/`, `scripts/build_case_library.py`
scores every already-trained model on **every** (region, lookback/forecast
window) instance across the full train+val+test span — not just the
aggregate metrics in `metrics.json` — and records which model wins each
instance, in raw target units:

```bash
python scripts/build_case_library.py \
  --runs-root runs/dc_house \
  --models dlinear patchtst gcn_tcn stgformer \
  --out runs/dc_house/case_library.csv
```

Three CSVs are written, each derived from the one before it:

1. **`<out>_detail.csv`** — one row per (model, instance, horizon step):
   `model, model_category, split, region_id, lookback_start, forecast_start,
   forecast_end, step_ahead, target_date, y_true_raw, y_pred_raw,
   abs_error_raw`. The actual forecasted and true values, in raw target
   units — computed once, directly from each model's predictions, before any
   summarization. This is the source of truth everything else is aggregated
   from; skip it with `--no-detail` if you only want the summary (it's the
   largest of the three — one row per horizon step, not per instance).
2. **`--out-long`** (optional) — one row per (model, instance): `mae, rmse`
   aggregated over the horizon, no per-step values.
3. **`--out`** — the case library summary: one row per instance, the winning
   model only — `region_id, lookback_start, forecast_start, forecast_end,
   model_best, model_best_mae, model_best_rmse, model_best_category`
   (category is `spatial_temporal` / `DL` / `foundation`, or `other` for
   statistical/ML baselines).

It reuses each run's own saved checkpoint (no retraining — except models with
no train-dependent state, like `timesfm_zero`/`chronos2_zero`, which run
directly since there's nothing a missing checkpoint would lose) and evaluates
with `window.test_stride` forced to `1` regardless of what the run was
trained with, since the stride exists only to cut normal benchmark evaluation
cost, not for this exhaustive per-instance comparison — so this can be
considerably slower than a normal `run_one.py` evaluation. All runs passed
together must share the same dataset, window shape, and split boundaries
(validated up front, with a clear error listing any mismatch).

`scripts/explain_instance.py` gives an on-demand, model-agnostic breakdown of
one instance's forecast, via occlusion (zero part of the input, rerun the
forward pass, measure how much the target's forecast moves — under this
benchmark's z-score transform, zeroing is approximately "replace with the
average", not an implausible perturbation):

```bash
python scripts/explain_instance.py \
  --run-dir runs/dc_house/stgformer__multivariate__w6_h3 \
  --region 20001 --lookback-start 2021-06-01 --device cpu
```

- **Feature contribution** (any model — GNN, DL, foundation, statistical/ML):
  occludes one input variable at a time (own price history, homes_sold,
  inventory, ...) for the target region and reports each one's share of the
  forecast change.
- **Neighbor contribution** (spatiotemporal/GNN models only, skipped
  otherwise): occludes one neighbor region at a time and reports each one's
  share, including a `self` row for how much the forecast relies on the
  target region's own history vs. its neighbors.

Both work identically across every model in the registry — not just ones
with built-in attention — since they only call the public
`model.predict_batch(...)` API.

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
