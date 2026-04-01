# Prompt-STDiff (PyTorch Skeleton)

可运行、可训练的 Prompt-STDiff 工程骨架，覆盖：
- 数据加载与切窗
- 物理图/语义图构建
- Prompt 与离线语义编码接口
- 扩散训练与反向采样
- Semantic-Guided Dynamic Noise Prior
- Step-Aware Cross-Modal Router
- MAE/RMSE/MAPE/CRPS 评估

## 1. 目录

```text
prompt-stdiff/
  configs/
  dataio/
  diffusion/
  graph/
  models/
  semantic/
  trainers/
  utils/
  train.py
  evaluate.py
  infer.py
```

## 2. 数据格式

默认按以下路径组织（支持 `pems03 / pems04 / pems08`）:

```text
data/
  pems03/
    data.npz
    adjacency.csv
    semantic_embeddings.npy
    A_sem.npy
    A_sem_norm.npy
    dynamic_semantic_bank.npz
```

`data.npz` 读取优先键：`data` -> `x` -> `arr_0`，要求形状 `[T, N, F]`。

`adjacency.csv` 至少两列：`src,dst`；可选第三列 `distance`。

## 3. 语义嵌入离线编码

推荐先准备更丰富的节点语义元数据（而不是直接用 `node_mapping.csv`）：

```bash
python scripts/build_semantic_metadata.py \
  --node_mapping_csv data/pems03/node_mapping.csv \
  --adjacency_csv data/pems03/adjacency.csv \
  --out_csv data/pems03/node_metadata.csv \
  --mode weak_labels
```

说明：
- `template`：只生成可手工补全的模板列。
- `weak_labels`：按图结构自动给出粗粒度标签（`ASSUMPTION`），可作为冷启动版本。

然后基于 `node_metadata.csv` 生成语义向量（可同时导出 prompt 文本便于检查）：

```bash
python -m semantic.offline_encoder \
  --metadata_csv data/pems03/node_metadata.csv \
  --out_file data/pems03/semantic_embeddings.npy \
  --prompts_out_file data/pems03/prompts.csv
```

基于新的 `Z_sem` 构建语义图（原始图+归一化图）：

```bash
python scripts/build_semantic_graph.py \
  --z_sem data/pems03/semantic_embeddings.npy \
  --a_sem_out data/pems03/A_sem.npy \
  --a_sem_norm_out data/pems03/A_sem_norm.npy \
  --top_k 20 \
  --norm_mode sym
```

## 3.1 动态语义与严格时间截断（阶段二）

若你有天气/事故/节假日等时间事件表，可先构建动态语义银行：

```bash
python scripts/build_dynamic_semantic_bank.py \
  --events_csv data/pems03/dynamic_events.csv \
  --out_npz data/pems03/dynamic_semantic_bank.npz \
  --start_time "2018-01-01 00:00:00" \
  --freq_minutes 5 \
  --model_name sentence-transformers/all-roberta-large-v1
```

如果你还没有 `dynamic_events.csv`，可先自动初始化（基于时间上下文）：

```bash
python scripts/init_dynamic_events.py \
  --data_npz data/pems03/data.npz \
  --out_csv data/pems03/dynamic_events.csv \
  --mode time_context \
  --start_time "2018-01-01 00:00:00" \
  --freq_minutes 5
```

如果你需要“真实外部事件抓取”（天气+节假日+可选事故 CSV 合并）：

```bash
python scripts/fetch_real_dynamic_events.py \
  --out_csv data/pems03/dynamic_events.csv \
  --start_date 2018-01-01 \
  --end_date 2018-03-31 \
  --latitude 34.0522 \
  --longitude -118.2437 \
  --country_code US \
  --timezone America/Los_Angeles
```

若要同时加入 POI 事件上下文（基于 OSM/Overpass）：

```bash
python scripts/fetch_real_dynamic_events.py \
  --out_csv data/pems03/dynamic_events.csv \
  --start_date 2018-01-01 \
  --end_date 2018-03-31 \
  --latitude 34.0522 \
  --longitude -118.2437 \
  --country_code US \
  --timezone America/Los_Angeles \
  --include_poi_context \
  --poi_radius_m 20000 \
  --poi_top_k 4 \
  --poi_catalog_csv data/pems03/poi_catalog.csv
```

然后再构建动态语义银行：

```bash
python scripts/build_dynamic_semantic_bank.py \
  --events_csv data/pems03/dynamic_events.csv \
  --out_npz data/pems03/dynamic_semantic_bank.npz \
  --start_time "2018-01-01 00:00:00" \
  --freq_minutes 5 \
  --model_name sentence-transformers/all-roberta-large-v1
```

启用配置（`configs/*.yaml`）：

```yaml
dataset:
  dynamic_semantic:
    enabled: true
    bank_file: dynamic_semantic_bank.npz
    fusion_alpha: 0.35
    recency_tau_steps: 288
    strict_truncation: true
```

说明：
- `strict_truncation=true` 时，每个样本只使用 `cutoff_step`（预测起点）之前的事件语义。
- 动态语义会与静态 `Z_sem` 融合后再进入 prior / router / denoiser。

## 4. Kaggle 原始数据预处理（你给的链接）

数据源：[elmahy/pems-dataset](https://www.kaggle.com/datasets/elmahy/pems-dataset)

先下载并解压到本地（示例）：

```bash
kaggle datasets download -d elmahy/pems-dataset -p data/raw --unzip
```

将原始长表转换为训练用 `data.npz`：

```bash
python scripts/preprocess_kaggle_pems.py \
  --raw_root data/raw \
  --out_root data \
  --splits pems03,pems04,pems08
```

说明：该 Kaggle 包中 `PEMSxx.npz` 已是时序张量，`PEMSxx.csv` 是图边文件，脚本会自动转换为训练需要的
`data.npz + adjacency.csv + node_mapping.csv`。

如果列名无法自动识别，可显式指定：

```bash
python scripts/preprocess_kaggle_pems.py \
  --raw_root data/raw \
  --out_root data \
  --splits pems03 \
  --time_col timestamp \
  --node_col station_id \
  --feature_cols flow,speed,occupancy
```

如需把 `PEMS04/PEMS08` 的 3 通道降为单通道（例如只用第 0 通道）：

```bash
python scripts/preprocess_kaggle_pems.py \
  --raw_root data/raw \
  --out_root data \
  --splits pems04,pems08 \
  --feature_indices 0
```

## 5. 训练

```bash
python train.py --config configs/default.yaml
```

可选使用数据集专用配置：

```bash
python train.py --config configs/pems03.yaml
python train.py --config configs/pems04.yaml
python train.py --config configs/pems08.yaml
```

指定设备（例如 CUDA）：

```bash
python train.py --config configs/pems04.yaml --device cuda:0
python train.py --config configs/pems08.yaml --device cuda:0
```

也可以直接指定 GPU ID（支持 `0-9`）：

```bash
python train.py --config configs/pems04.yaml --gpu_id 0
python evaluate.py --config configs/pems04.yaml --ckpt outputs/checkpoints/pems04/best.pt --gpu_id 0
python infer.py --config configs/pems04.yaml --ckpt outputs/checkpoints/pems04/best.pt --split test --out outputs/pems04_preds.npy --gpu_id 0
```

## 6. 评估

```bash
python evaluate.py --config configs/default.yaml --ckpt outputs/checkpoints/best.pt
```

默认会额外报告 `horizon=3/6/12` 的 MAE/RMSE/MAPE/CRPS（可在 `configs/train.yaml` 的 `eval_horizons` 修改）。

MAPE 默认采用稳健设置：
- `mape_mask_threshold: 1.0`（只统计 `|y|>1.0` 的位置，避免分母接近 0 导致爆炸）
- `metric_feature_index` 可指定只评估某个特征通道（`pems04/pems08` 默认设为 `0`）。

## 7. 推理

```bash
python infer.py --config configs/default.yaml --ckpt outputs/checkpoints/best.pt --split test --out outputs/predictions.npy
```

## 8. 与论文对齐说明

- 默认噪声先验：`learn_sigma_prior=false`，使用 `x_K = gamma * mu_sem + sqrt(1-gamma^2) * eps`。
- 前向扩散训练默认使用标准高斯噪声（Eq.(9)）；语义先验用于反向初始化（Eq.(12)）。
- `learn_mu_prior=false` 时使用无参数语义投影（稳定默认）；可切到 `true` 作为扩展。
- Router 为两路：`h_sem` 与 `h_traffic` 的 step-aware 融合。
- 物理图默认采用论文式对称归一化（`physical_norm_mode: sym`），并支持 `physical_sigma: auto`。
- 训练默认 `K=50`，评估默认 `num_eval_samples=100` 用于 CRPS 与分布均值指标。
- 对论文未明确的实现细节已在代码中用 `ASSUMPTION` 标注。
