# External Baseline Protocol

This directory records the fairness protocol for LLM/event-aware external
baselines. The baseline implementations are not vendored in this repository by
default. Official repositories should be run with the canonical artifacts
exported by `scripts/export_canonical_setup.py`.

## Canonical Artifacts

Export PeMS04/PeMS08 split and scaler files before running any external
baseline:

```bash
python scripts/export_canonical_setup.py \
  --config configs/pems04.yaml \
  --config configs/pems08.yaml \
  --out_dir outputs/canonical_setup \
  --tag h12_t12
```

Each export creates:

- `<dataset>_<tag>.npz`
- `<dataset>_<tag>.json`

The `.npz` file contains:

- `train_windows`, `val_windows`, `test_windows`: `[M, 3]` global window
  indices with columns `[his_start, his_end, fut_end]`.
- `scaler_mean`, `scaler_std`: training-only `StandardScaler` statistics.
- `split_ranges`: train/val/test global time ranges.

The `.json` file contains dataset/config metadata and the exact window slicing
semantics:

```text
x_his = data[his_start:his_end]
x_fut = data[his_end:fut_end]
```

External baselines must not rebuild their own split, scaler, or sliding-window
indices.

## Baseline Implementation Status

| Baseline | Category | Required status in logs | Source status | Notes |
|---|---|---|---|---|
| Time-LLM | LLM time-series baseline | `official` if official repo is used | Official repo: `https://github.com/KimMeen/Time-LLM` | Adapt loader to consume canonical windows/scaler. |
| UrbanGPT | Urban LLM baseline | `official` if official repo is used | Official repo: `https://github.com/HKUDS/UrbanGPT` | Adapt PeMS04/08 loader to canonical windows/scaler. |
| ST-LLM | Spatio-temporal LLM baseline | `official` if official repo is used | Official repo: `https://github.com/ChenxiLiu-HNU/ST-LLM` | Adapt loader and report MAE/RMSE. |
| Strada-LLM | Graph-aware probabilistic LLM | `official` or `reimplemented` | Paper: `https://arxiv.org/abs/2410.20856`; official code not confirmed in this repo | If official code is unavailable, reimplementation must be logged explicitly. |
| FUSE-Traffic | Event-aware fusion competitor | `official` or `reimplemented` | Paper/summary found; official code not confirmed in this repo | Must consume the same Prompt-STDiff semantic/event files. |
| Structured control | Non-LLM exogenous control | `control` | Uses raw weather/incident/holiday/POI features, not RoBERTa embeddings. |

Every run must be recorded with `utils.result_writer.ExperimentResult`, including
the `implementation` field and a serialized `settings_json` field.

## Fairness Requirements

1. Use the exported canonical windows and scaler.
2. Use the same PeMS raw data file as the Prompt-STDiff config.
3. Use the same horizons `[3, 6, 12]`.
4. For semantic/event-aware baselines, use the same semantic cache files as
   Prompt-STDiff:
   - `semantic_embeddings.npy`
   - `prompts.csv`
   - `dynamic_semantic_bank.npz` when enabled
5. For non-LLM structured controls, use the same raw weather/incident/holiday/POI
   fields that produced Prompt-STDiff prompts, but encode them as numeric/one-hot
   features instead of frozen RoBERTa embeddings.

## Semantic-Augmented Baseline Adapter

The repository does not vendor official GWNet/AGCRN/PDFormer/DiffSTG/PriSTI/
SpecSTG implementations. To keep those baselines auditable, semantic injection
is provided as a reusable adapter rather than silently modifying third-party
model definitions.

Use:

```python
from semantic_injection import (
    BatchSemanticComposer,
    DeterministicSemanticInputAdapter,
    GWNetSemanticWrapper,
    AGCRNSemanticWrapper,
    PDFormerSemanticWrapper,
    DiffusionSemanticConditionAdapter,
    adjusted_input_dim,
)
```

Deterministic graph baselines should compose a per-batch semantic tensor from
the same cache as Prompt-STDiff:

```python
z_batch = composer.compose(batch, device=device, num_nodes=x_his.shape[2])
x_his_aug = det_adapter(x_his, z_batch=z_batch)
```

When `use_semantic=False`, `det_adapter(x_his)` returns the original `x_his`
object and creates no projection parameters. This preserves the original
baseline construction path.

For official deterministic baselines, prefer the named wrappers:

```python
in_dim = adjusted_input_dim(config)  # original F, or F + semantic_proj_dim
base_model = build_gwnet(in_dim=in_dim, ...)
model = GWNetSemanticWrapper(
    base_model,
    sem_dim=composer.sem_dim,
    d_proj=config["baseline"]["semantic_proj_dim"],
    use_semantic=config["baseline"]["use_semantic"],
)

z_batch = composer.compose(batch, device=device, num_nodes=x_his.shape[2])
pred = model(x_his, z_batch=z_batch)
```

Default layouts:

- `GWNetSemanticWrapper`: feeds `[B, F, N, T]` to the wrapped model.
- `AGCRNSemanticWrapper`: feeds `[B, T, N, F]` to the wrapped model.
- `PDFormerSemanticWrapper`: feeds `[B, T, N, F]` to the wrapped model.

Override `baseline_layout` if an official implementation uses a different
layout.

Diffusion baselines should attach projected semantics to their existing
condition mapping:

```python
cond = diffusion_adapter(cond, z_batch=z_batch)
```

When `use_semantic=False`, the original condition mapping object is returned.
Run the adapter contract check with:

```bash
python scripts/check_semantic_injection.py --config configs/pems04.yaml --device cpu
python scripts/check_deterministic_semantic_wrappers.py --config configs/pems04.yaml --device cpu
```

## Loader Integration Sketch

External repos should implement the following logic:

```python
bundle = np.load("outputs/canonical_setup/pems08_h12_t12.npz")
windows = bundle["train_windows"]  # or val_windows/test_windows
mean = bundle["scaler_mean"]
std = bundle["scaler_std"]

raw = load_pems_data(...)
raw = raw[..., :input_dim]
scaled = (raw - mean) / std

for his_start, his_end, fut_end in windows:
    x_his = scaled[his_start:his_end]
    x_fut = scaled[his_end:fut_end]
```

This mirrors `dataio.traffic_dataset.build_dataloaders` while making the split
and scaler explicit for external codebases.

## AGCRN runner

`baselines/runners/run_agcrn.py` trains/evaluates AGCRN under this repo's canonical
split, scaler, metrics, and result logging. It does not modify the official AGCRN
repo. Point `--agcrn_repo` or `--model_file` at the official checkout.

Original AGCRN, no semantic features:

```bash
python -m baselines.runners.run_agcrn \
  --config configs/pems03.yaml \
  --device cuda:0 \
  --agcrn_repo baselines/external_repos/AGCRN \
  --epochs 50 \
  --eval_interval 5 \
  --save_tag agcrn_original
```

AGCRN + Prompt-STDiff semantic adapter:

```bash
python -m baselines.runners.run_agcrn \
  --config configs/pems03.yaml \
  --device cuda:0 \
  --agcrn_repo baselines/external_repos/AGCRN \
  --use_semantic \
  --semantic_proj_dim 128 \
  --epochs 50 \
  --eval_interval 5 \
  --save_tag agcrn_semantic
```

For quick interface checks only, use `--fallback`; fallback results are smoke-test
only and must not be reported as AGCRN paper baselines.

```bash
python -m baselines.runners.run_agcrn \
  --config configs/pems03.yaml \
  --device cpu \
  --fallback \
  --epochs 1 \
  --max_train_batches 1 \
  --max_eval_batches 1 \
  --batch_size 2 \
  --num_workers 0
```

If the official AGCRN constructor in your checkout uses non-standard required
arguments, pass `--model_file path/to/AGCRN.py` first. If auto-instantiation still
fails, add the missing constructor mapping in `_constructor_value(...)` in the
runner rather than changing the official baseline code.
