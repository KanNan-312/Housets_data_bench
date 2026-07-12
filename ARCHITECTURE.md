# HouseTS Benchmark – Code Architecture

## Repository layout

```
Housets_data_bench/
├── configs/                  # YAML config fragments
│   ├── default.yaml          # base defaults (data path, split ratios, transforms, dataloader)
│   ├── task/                 # univariate.yaml / multivariate.yaml  (features_mode)
│   ├── windows/              # w6_h3.yaml … w12_h12.yaml  (seq_len, label_len, pred_len)
│   └── models/               # one file per model  (model.name, model.hparams)
├── scripts/
│   ├── run_one.py            # ← main entry point
│   ├── run_sweep.py          # grid-sweep helper
│   ├── run_dl_baselines.py   # batch runner for DL models
│   └── ...
├── src/housets_bench/
│   ├── data/                 # loading, schema, imputation, windowing, dataset
│   ├── bundles/              # RawBundle / ProcBundle dataclasses + builder
│   ├── transforms/           # log / clip / zscore / pca stages + pipeline
│   ├── models/               # all forecasters (base, registry, dl/, ml/, stats/, etc.)
│   ├── metrics/              # evaluator, loss helpers, regression metrics
│   ├── experiments/          # runner.py, sweep.py, artifacts.py
│   └── utils/                # config loading, deep_update, path resolution
└── runs/                     # output directory (auto-created)
```

---

## Entry point: `scripts/run_one.py`

```
run_one.py
  parse_args()          # --task, --window, --model, --data, --device, --set …
  load + deep_update    # merge default.yaml → task → window → model configs
  run_one_cfg(cfg)      # ← all real work happens here  (experiments/sweep.py)
  make_run_dir()        # runs/<model>__<task>__<window>/
  save_yaml / save_json # config.yaml, metrics.json, env.json
  print JSON result
```

`--set key=value` can override any nested config key at the CLI.

---

## Config system

Configs are plain YAML dicts that are **deep-merged** in this order:

```
configs/default.yaml          (baseline settings for everything)
configs/task/<task>.yaml      (sets task.features_mode: S / MS)
configs/windows/<window>.yaml (sets window.seq_len, pred_len, label_len)
configs/models/<model>.yaml   (sets model.name, model.hparams.*)
```

`deep_update(base, override)` recursively merges dicts; scalars override.  
CLI `--set` overrides are applied last via `pop_cli_overrides`.

---

## Data pipeline

### Step 1 – Load raw table

```
load_aligned(path, target_col="price")
  read_table()              # csv / parquet / xlsx
  FeatureSchema.infer(df)   # detects id_col, time_col, continuous_cols, drop non-numeric
  clean_raw_table()         # parse dates, normalize zipcodes, drop non-feature cols, add year/month
  align_to_tensor()         # pivot → np.ndarray [Z, T, D]  +  three_stage_impute()
  → AlignedData(zipcodes, dates, values[Z,T,D], time_marks[T,2], schema)
```

### Step 2 – Split

```
make_ratio_split(n_time, train_ratio=0.7, val_ratio=0.1)
  → TimeSplit(train=(0, t1), val=(t1, t2), test=(t2, T))
```
Boundaries are integer time-step indices into the `dates` axis.

### Step 3 – Build ProcBundle

`build_proc_bundle(aligned, split, spec, features_mode, pipeline, …)`

```
1. pipeline.fit_transform(values[Z,T,D], train_range=split.train)
      → values_proc[Z,T,D']   (fit stats on train slice only)

2. select x_cols / y_cols  from features_mode
      S  → x=[target], y=[target]
      MS → x=[all features], y=[target]
      M  → x=[all], y=[all]

3. generate_window_indices(values_proc, split_range, spec)
      → list of (zip_idx, time_idx) anchor tuples per split

4. WindowDataset(aligned_proc, indices, spec)
      __getitem__ returns dict:
        x       [seq_len, Dx]          encoder input
        y       [label_len+pred_len, Dy]  decoder target (label window + horizon)
        x_mark  [seq_len, 2]           year/month time marks for encoder
        y_mark  [label_len+pred_len, 2] time marks for decoder
        meta    SampleMeta(zip, t_start, t_end)

5. DataLoader with collate_fn
      batches: x[B,L,Dx], y[B,Ly,Dy], x_mark[B,L,2], y_mark[B,Ly,2], x_mask[B,L]

→ ProcBundle(raw, pipeline, aligned_proc, x_cols, y_cols, datasets, dataloaders,
             raw_target_col, raw_target_index)
```

`x_mask` is 1 for real tokens, 0 for left-padding (used when `pad_to` > `seq_len`).

---

## Transform pipeline

Stages applied in order: `log → clip → zscore → pca` (configurable).  
Each stage is fitted on the **train slice only** and applied to the full tensor.

| Transform | Config key | Fitted params |
|-----------|-----------|---------------|
| `LogTransform` | `transforms.log` | none (stateless) |
| `ClipTransform` | `transforms.clip` | quantile / sigma bounds |
| `ZScoreTransform` | `transforms.zscore` | per-feature mean, std |
| `PCATransform` | `transforms.pca` | PCA components |

`pipeline.inverse(values, keep_log=False/True)` is used at evaluation to convert predictions back to raw price space for metric computation.

---

## Model system

### `BaseForecaster` (`models/base.py`)

```python
class BaseForecaster(ABC):
    def fit(self, bundle: ProcBundle, *, device) -> None:  # optional, default no-op
    def predict_batch(self, batch, *, bundle, device) -> torch.Tensor:  # [B, H, Dy]
```

### Registry (`models/registry.py`)

```python
@register("dlinear")
class DLinearForecaster(BaseForecaster): ...
```

`get_model("dlinear")` → calls the registered factory → returns a **new instance**.  
All public class attributes (e.g. `epochs`, `lr`, `hidden_size`) become hyper-parameters that `apply_hparams(model, cfg.model.hparams)` sets via `setattr`.

### Model families

| Family | Files | Notes |
|--------|-------|-------|
| DL | `dl/{dlinear,rnn,lstm,patchtst,timemixer,informer,autoformer,fedformer}.py` | Full training loop inside `fit()` |
| ML | `ml/{rf,xgb}.py` | Flatten windows → sklearn/XGBoost fit |
| Stats | `stats/ardl.py` | statsmodels ARDL |
| Naive | `naive/ar_univariate.py` | per-ZIP AR(p) |
| Foundation | `foundation/{chronos,timesfm}.py` | zero-shot or fine-tuned |
| GNN | `gnn/{gcn_tcn_geo,graph_wavenet,stgcn}.py` | spatial graph models |

---

## DL training loop (inside `fit()`)

All eight DL models follow the same structure:

```python
def fit(self, bundle, *, device):
    train_dl = bundle.dataloaders["train"]
    val_dl   = bundle.dataloaders["val"]
    net = <ModelNet>(...).to(dev)
    opt = Adam(net.parameters(), lr=self.lr)

    best_val, best_state, bad_epochs = inf, None, 0
    _train_total = min(len(train_dl), max_train_batches) if max_train_batches else len(train_dl)

    epoch_bar = tqdm(range(self.epochs), desc="[model_name]", unit="ep")
    for ep in epoch_bar:

        # ── train ────────────────────────────────────────────
        net.train()
        train_bar = tqdm(train_dl, desc="  train", leave=False)
        for bi, batch in enumerate(train_bar):
            if max_train_batches and bi >= max_train_batches: break
            x, y_true = batch["x"], batch["y"][:, -pred_len:, :]
            # (informer/autoformer/fedformer also use x_mark, y_mark, dec_in)
            y_pred = net(x)
            loss = F.mse_loss(y_pred, y_true)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(net.parameters(), self.grad_clip)
            opt.step()
            train_bar.set_postfix({"loss": ...})

        # ── validate ──────────────────────────────────────────
        net.eval()
        with torch.no_grad():
            val_bar = tqdm(val_dl, desc="  val  ", leave=False)
            for batch in val_bar:
                ...accumulate SSE...
                val_bar.set_postfix({"mse": ...})
        val_mse = sse / n

        # ── early stopping ───────────────────────────────────
        if val_mse < best_val - 1e-12:
            best_val = val_mse
            best_state = deepcopy(net.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= self.patience: break

        epoch_bar.set_postfix({"train": ..., "val": ..., "best": ...})
        self.train_history.append({"epoch": ep+1, "train_mse": ..., "val_mse": ..., ...})

    net.load_state_dict(best_state)   # restore best checkpoint
```

`self.train_history` is a list of per-epoch dicts that ends up in `metrics.json` under `"train_history"`.

**Decoder-based models** (Informer, Autoformer, FEDformer) additionally:
- Build `dec_in = self._make_decoder_input(y_full, label_len, pred_len)` — the label window concatenated with a zero-padded forecast horizon.
- Pass `(x, x_mark, dec_in, y_mark)` to the network.

---

## Evaluation loop

Called twice after training: once for `"val"`, once for `"test"`.

```python
evaluate_forecaster(model, bundle, split="test", device, max_batches, metric_space="log")
  → EvalResult(rmse, mape, n_points, metric_space)
```

Internally uses `StreamingEvaluator` which:

1. Calls `model.predict_batch(batch, bundle, device)` → `y_pred [B, H, Dy]` (processed space)
2. Embeds predictions into the full feature dimension
3. Always calls `pipeline.inverse(…, keep_log=False)` → fully inverted **raw price** values (pipeline-agnostic, fair for any model regardless of internal transforms)
4. For RMSE, the space depends on `metric_space`:
   - `"log"` (default): `RMSE(log1p(p_raw), log1p(t_raw))` — **log-RMSE on raw prices** (scale-normalised, comparable across models including LLMs predicting on original scale)
   - `"original"`: `RMSE(p_raw, t_raw)` — **RMSE in raw dollar space**
5. MAPE always uses `p_raw` and `t_raw` directly

Set via config: `run.metric_space: original`, or CLI: `--set run.metric_space=original`.

A second pass via `evaluate_mse_loss` computes **processed-space MSE** (no inverse transform) for all three splits; this is what the DL training loop minimises.

---

## Experiment runner: `run_one_cfg` (`experiments/sweep.py`)

```
run_one_cfg(cfg, device)
  load_aligned(data.path)             # Step 1: load + align
  [optional: subsample n_zip ZIPs]
  build_bundle_from_cfg(aligned, cfg) # Steps 2–5 above
  _log_dataset_summary(aligned, bundle)  # prints ZIPs × T, split sizes, window, pipeline
  get_model(model.name)               # instantiate from registry
  apply_hparams(model, cfg.model.hparams)
  model.fit(bundle, device)           # training loop (tqdm bars visible here)
  evaluate_forecaster(model, bundle, split="val")
  evaluate_forecaster(model, bundle, split="test")
  evaluate_mse_loss(model, bundle, split="train/val/test")
  extract_train_history(model)
  → dict with "model", "task", "window", "val", "test", "n_train/val/test",
              "timing", "loss", "pipeline", ["train_history"]
```

---

## Output artefacts

`make_run_dir(root, name)` creates `runs/<name>/` containing:

| File | Contents |
|------|----------|
| `config.yaml` | Fully merged config used for the run |
| `metrics.json` | Return dict from `run_one_cfg` (val/test metrics, timing, loss, train history) |
| `env.json` | Python / package versions at run time |

---

## Key dataclasses at a glance

| Class | Location | Purpose |
|-------|----------|---------|
| `AlignedData` | `data/io.py` | Raw tensor `[Z, T, D]` + metadata before transforms |
| `FeatureSchema` | `data/schema.py` | Column roles: id, time, target, continuous, drop |
| `TimeSplit` | `data/split.py` | Integer index ranges for train/val/test |
| `WindowSpec` | `data/windowing.py` | `seq_len`, `label_len`, `pred_len` |
| `RawBundle` | `bundles/datatypes.py` | `AlignedData + TimeSplit + WindowSpec + features_mode` |
| `ProcBundle` | `bundles/datatypes.py` | Everything a model needs: processed data, dataloaders, pipeline |
| `EvalResult` | `metrics/evaluator.py` | `rmse`, `mape`, `n_points`, `metric_space` |
