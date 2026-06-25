# Teleop: Perception Neuron → Inspire Hand → Isaac

用诺亦腾（Noitom）**Perception Neuron** 动捕设备，实时驱动 **Isaac** 仿真环境中的
**Inspire（因时）灵巧手**。

```
┌──────────────────────┐   ┌───────────────────────┐   ┌────────────────────┐
│  mocap               │   │  retarget             │   │  sim               │
│  Perception Neuron   │──▶│  human hand → Inspire │──▶│  Inspire Hand in   │
│  (BVH file / UDP)    │   │  6-DOF actuators      │   │  Isaac Gym / Lab   │
└──────────────────────┘   └───────────────────────┘   └────────────────────┘
```

## 设计要点

- **三段解耦**：`mocap`（采集）/ `retarget`（重定向）/ `sim`（仿真）通过清晰接口连接。
- **无硬件可运行**：BVH 解析与重定向只依赖 `numpy`，CI/离线测试无需 GPU 或动捕设备。
- **可优雅降级**：Isaac Gym 与 Noitom 实时流为可选；缺失时 `DummyInspireHand`
  仍可记录指令，用于验证整条数据流。

## 目录

```text
teleop_inspire_isaac/
  mocap/
    bvh.py                 # 通用 BVH 解析器（Axis Studio/Axis Neuron 导出格式）
    perception_neuron.py   # BVHFileSource（回放）/ AxisNeuronUDPSource（实时流）
  retarget/
    inspire_retargeter.py  # 人手关节角 → Inspire 6 自由度执行器指令
  sim/
    isaac_inspire_env.py   # IsaacInspireHand（Isaac Gym）/ DummyInspireHand（无 GPU）
  pipeline.py              # 端到端 teleop 循环（含 EMA 平滑）
  config/default.yaml      # 配置示例
  scripts/run_teleop.py    # 运行入口
  assets/sample_hand.bvh   # 离线示例数据
  tests/                   # BVH 解析 + 重定向单元测试
```

## Inspire Hand 执行器约定

Inspire Hand（RH56 系列）共 **6** 个执行器，规范顺序：

| 序号 | 执行器       | 含义                |
|------|--------------|---------------------|
| 0    | `little`     | 小指弯曲            |
| 1    | `ring`       | 无名指弯曲          |
| 2    | `middle`     | 中指弯曲            |
| 3    | `index`      | 食指弯曲            |
| 4    | `thumb_bend` | 拇指弯曲            |
| 5    | `thumb_rot`  | 拇指旋转（对掌）    |

指令范围默认 `0..1000`（Inspire SDK 约定），**数值越大越张开**，因此默认对
人手弯曲做反向映射（`invert_flexion: true`）。

## 快速开始（离线，无需 GPU）

```bash
pip install numpy PyYAML
python -m teleop_inspire_isaac.scripts.run_teleop \
    --config teleop_inspire_isaac/config/default.yaml --max-frames 100
```

## 实时流程

1. **采集**：在 Axis Studio / Axis Neuron 中开启
   `Settings → Output → BVH → UDP`，选择 **字符串（ASCII）** 格式输出，
   记下端口（默认 `7002`）。同时导出一份参考骨架 `.bvh` 供解析列顺序。
2. **配置**：将 `config/default.yaml` 的 `mocap.source` 改为 `udp`，
   填好 `ref_bvh` / `port`；将 `sim.backend` 改为 `isaac` 并配置 Inspire Hand
   的 URDF/MJCF 资源路径与 `device`。
3. **运行**：

   ```bash
   python -m teleop_inspire_isaac.scripts.run_teleop --config my_isaac_config.yaml
   ```

## 标定说明

`flexion_axis` / `thumb_rot_axis` 与各自的 `*_min_deg` / `*_max_deg`
取决于具体骨架的关节坐标系，需按实际动捕数据标定：先用 `bvh` 离线源观察
张开/握拳两个极端姿态的关节欧拉角，再设置对应轴与角度区间。

## 测试

```bash
python -m pytest teleop_inspire_isaac/tests -q
```
