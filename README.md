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
- `external_match`：从外部静态元数据表按 `node_index`/`node_id` 确定性合并（推荐用于论文主实验）。

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

# 若事故 CSV 提供经纬度，可做 “2km + 拓扑” 映射：
python scripts/fetch_real_dynamic_events.py \
  --out_csv data/pems03/dynamic_events.csv \
  --start_date 2018-01-01 \
  --end_date 2018-03-31 \
  --latitude 34.0522 \
  --longitude -118.2437 \
  --country_code US \
  --timezone America/Los_Angeles \
  --incidents_csv data/pems03/incidents_raw.csv \
  --sensor_metadata_csv data/pems03/sensor_geo.csv \
  --adjacency_csv data/pems03/adjacency.csv \
  --incident_radius_m 2000 \
  --topology_hops 3
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

也可先单独把事故经纬度映射到传感器节点：

```bash
python scripts/map_incidents_to_sensors.py \
  --incidents_csv data/pems03/incidents_raw.csv \
  --sensor_csv data/pems03/sensor_geo.csv \
  --adjacency_csv data/pems03/adjacency.csv \
  --out_csv data/pems03/incidents_mapped.csv \
  --radius_m 2000 \
  --topology_hops 3 \
  --assign_global_if_unmapped
```
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
  semantic_required: true
  allow_random_semantic_fallback: false
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
- `semantic_required=true` 时若缺失 `semantic_embeddings.npy` 会直接报错（论文主实验推荐）。

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
python train.py --config configs/pems03_full.yaml
python train.py --config configs/pems04_full.yaml
python train.py --config configs/pems08_full.yaml
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

## 9. Incident-gated 残差均值修正（diagnose → repair）

`ρ` 分解（`scripts/eval_event_root_cause.py`）证明 incident/drop 上的覆盖失效是
**均值级**（`ρ ≫ 1`）而非方差级，因此 widening / conformal / tail-scale 这类
**只改方差**的方法理论上修不了。为此新增一个**门控的残差均值修正**机制，把 `ρ`
从“诊断指标”升级为“修复机制”：

- `RegimeShiftHead`：从泄漏安全的历史/语义条件预测每个 `(node, horizon, feature)`
  的 regime-shift 门 `g ∈ [0, 1]`（检测器），默认偏置使其初始接近 0。
- `MeanCorrectionHead`：预测**有符号**的均值平移 `δ`（不是正的乘性 scale），可双向
  移动预测中心。
- `GraphMeanPropagator`：沿物理图 `A_phy` 把 `g·δ` 传播到下游邻居
  （`Σ_k w_k A^k (g·δ)`，`w_k` 用 softmax 且 identity 初始化），让单个检测到的
  drop 修正它在空间上引发的相关 drop。

修正后的预测为 `μ̂ + propagate(g · δ)`，对所有 ensemble 样本**同样平移**，因此只移动
**中心**、不改变**离散度**（残差谱）。门稀疏 + 传播器 identity 初始化 ⇒ 非 incident
位置保持均值保持（mean-preserving）默认，full-test 点精度按构造不变。

### 两种评估模式（保留受控公平性 C1）

- **严格均值保持模式**：`use_incident_mean_correction: false`（或对已训练模型调用
  `model.set_incident_correction_enabled(False)`），门强制关闭，等价于原始受控
  scaffold，用于 diffusion-vs-Gaussian 的同 backbone 对比。
- **Incident-corrected 模式**：`use_incident_mean_correction: true`，门生效，作为新方法。

### 训练（复用 freeze-tail 协议）

加载已训练的残差扩散栈，仅训练修正头：

```bash
python train.py --config configs/pems08_incident_mean_correction.yaml --gpu_id 0
```

关键配置项：

```yaml
model:
  use_incident_mean_correction: true
  incident_correction_graph_hops: 2
  incident_correction_max_shift: 4.0
  incident_correction_gate_bias: -4.0
train:
  loss_regime_detect_weight: 1.0        # 检测损失 (BCE)
  loss_mean_correction_weight: 1.0      # 在检测位置回归 g·δ → 真实均值残差
  loss_correction_sparsity_weight: 0.1  # 非 incident 位置抑制修正，保住 full-test MAE
  incident_regime_threshold: 2.0        # 标准化残差幅度阈值定义 regime 标签
  freeze_except_prefixes:
  - regime_shift_head
  - mean_correction_head
  - mean_correction_propagator
```

### 评估闭环（收口 C3）

修正在 `trainers/evaluator.py`、`scripts/eval_event_subset.py`、`infer.py` 的采样
路径中通过 `model.apply_mean_correction(...)` 应用。修正后重跑
`scripts/eval_event_root_cause.py`，预期 headline 结果为
**“drop 上 `ρ` 由 7–62 降向 ≈1、drop-PICP 由 ≈0 升向 nominal，而 off-incident 指标不变”**，
即同时 *diagnose*（`ρ`）并 *repair* 了方差类方法修不了的均值级失效。

> 完整性要求：门阈值的选择必须仅用 validation（避免 test 泄漏）；并应报告
> off-incident `ΔMAE ≈ 0` 以证明没有用全局精度换取 incident 覆盖。

