# PPI-Sifter

> **轻量可解释蛋白质相互作用预测模型（对比学习版）**
>
> Lightweight Interpretable PPI Prediction with Contrastive Learning

---

## 项目简介

PPI-Sifter 是一个面向高通量蛋白质相互作用（PPI）筛选与机制分析的轻量序列模型，具备两项核心能力：

- **预测**：对候选蛋白对输出 pair-level interaction probability，支持高通量筛选；
- **解释**：无需 residue-level 界面标注，通过对比学习与层级表征分析，量化回答模型学到了什么样的 partner-aware interaction pattern。

模型以冻结 ESM-C 为编码器，复用 B-PPI 风格的双向 cross-attention 判别主干，在训练目标侧增加 **LayerwiseContrastHead**，在推理侧增加 **对比机制分析链路**（layer separability · partner shift · attention entropy gap）。

---

## 模型结构

<!-- 模型结构图占位，后续替换 -->
> 📌 *Model architecture diagram — to be added.*

---

## 目录结构

```
PPI-Sifter/
├── configs/                        # 所有 YAML 配置文件
│   ├── default.yaml                # 默认训练配置
│   ├── contrast.yaml               # 消融实验配置
│   └── interpret.yaml              # 推理 + 解释导出配置
│
├── ppisifter/                      # 核心模型包
│   ├── model.py                    # PPISifter 主模型
│   ├── attention.py                # MultiHeadCrossAttention
│   ├── modules.py                  # GatedFFN, AttentionPooling
│   ├── contrast.py                 # LayerwiseContrastHead
│   ├── losses.py                   # PPILoss（返回 loss dict）
│   ├── analysis.py                 # 解释分析函数
│   ├── data.py                     # PPIDataset, DataLoader 构建
│   ├── metrics.py                  # AUROC, AUPRC, F1, MCC
│   ├── interpret.py                # 推理解释工具（兼容保留）
│   ├── io.py                       # HDF5 读写工具
│   ├── config.py                   # 配置加载
│   ├── constants.py                # 全局常量
│   ├── train_utils.py              # 训练辅助工具
│   └── utils.py                    # 通用工具
│
├── scripts/
│   ├── emb/                        # 离线特征提取
│   │   └── extract_esmc.py         # ESM-C → HDF5
│   ├── biogrid/                    # 数据预处理
│   │   └── build_pairs.py          # BioGRID → pairs CSV
│   ├── train/
│   │   ├── train.py                # 训练主入口 ✅
│   │   ├── eval.py                 # 标准评估 ✅
│   │   └── infer.py                # 推理 + 中间层导出 ✅
│   ├── interpretability/
│   │   ├── run_contrast_analysis.py   # 对比机制分析主入口 ✅ 新增
│   │   ├── plot_interpret.py          # 可视化（热图/UMAP/折线图）
│   │   └── quantify_interpret.py      # 精细化 attention 量化（备用）
│   └── tests/                      # 单元测试
│
├── docs/                           # 设计文档
│   ├── 项目设计书（对比学习）.md
│   └── 对比学习解释性分析流水线.md
│
├── data/                           # 数据目录（不入 git）
│   └── BIOGRID/
│       ├── embeddings/embeddings.h5
│       └── pairs_*.csv
│
├── checkpoints/                    # 模型权重（不入 git）
├── outputs/                        # 推理 / 分析结果（不入 git）
├── references/                     # 参考论文与项目说明
├── requirements.txt
└── README.md
```

---

## 环境安装

### 推荐环境

| 项目 | 版本                          |
|---|-----------------------------|
| Python | 3.10                        |
| PyTorch | 2.1.0                       |
| CUDA | 12.1                        |
| 显卡 | RTX 4070S（本地调试）/ A100（云端训练） |
| OS | Win11（本地）/ Linux（云端）        |

### 安装依赖

```bash
pip install -r requirements.txt
```

> ESM-C 模型权重需从 [EvolutionaryScale](https://github.com/evolutionaryScale/esm) 官方获取，不随本仓库分发。

---

## 快速开始

### 步骤 0：离线提取 ESM-C Embeddings

```bash
python scripts/emb/extract_esmc.py --config configs/default.yaml
```

输出：`data/BIOGRID/embeddings/embeddings.h5`

### 步骤 1：训练

```bash
python scripts/train/train.py --config configs/default.yaml
```

输出：`checkpoints/best_auprc.pt`

### 步骤 2：标准评估

```bash
python scripts/train/eval.py \
    --config configs/default.yaml \
    --ckpt checkpoints/best_auprc.pt
```

输出：`outputs/preds/eval_results.json` + PR 曲线图

### 步骤 3：推理 + 中间层表示导出

```bash
python scripts/train/infer.py \
    --config configs/interpret.yaml \
    --ckpt checkpoints/best_auprc.pt
```

输出：`outputs/preds/predictions.csv` + `outputs/layer_reprs/*.pt`

### 步骤 4：对比机制分析

```bash
python scripts/interpretability/run_contrast_analysis.py \
    --config configs/interpret.yaml
```

输出：`outputs/analysis/*.json`

### 步骤 5：可视化（论文插图）

```bash
python scripts/interpretability/plot_interpret.py \
    --config configs/interpret.yaml
```

输出：`outputs/interpret/*.png`

---

## 核心创新

| 创新点 | 说明 |
|---|---|
| **对比学习解释增强** | 以 Supervised Contrastive Loss 强化各层 pair representation 正负可分性 |
| **Layer-wise separability** | 对每层独立训练线性 probe，量化 AUROC，定位 partner-aware pattern 深度 |
| **Partner shift analysis** | 固定 anchor 蛋白，比较面对不同 partner 时中间表示的定向偏移 |
| **Attention entropy profiling** | 跨层统计正负样本 attention entropy 差异，形成可量化机制论证 |
| **对比学习可关闭** | `contrast.enabled=false` 退化为原版主干，方便消融对比 |

---

## 评测指标

### 主任务

- **AUPRC**（主指标，适合类别不平衡）
- AUROC
- F1 @ threshold=0.5
- MCC

### 解释性分析

- Layer probe AUROC/AUPRC（per layer）
- Silhouette score / Davies-Bouldin index
- Partner shift gap（真实 vs 负 partner 表示偏移差）
- Attention entropy gap（正负样本各层 entropy 差）
- Seed stability（跨 3 个 seed 结论一致性）

---

## 配置说明

所有超参统一由 YAML 管理，核心配置项：

```yaml
contrast:
  enabled: true          # 是否启用对比学习分支
  active_layers: [1, 2]  # 对哪些层施加对比损失
  method: supcon         # supcon | triplet | infonce
  lambda_contrast: 0.3   # 对比损失权重

model:
  d_model: 256
  n_layers: 2
  n_heads: 8
```

完整配置见 `configs/default.yaml`。

---

## 消融实验

详见 `configs/contrast.yaml`，覆盖以下消融维度：

1. **对比层选择**：layer 1 only / layer 2 only / all layers / no contrast
2. **对比方案**：SupCon / Triplet / InfoNCE / disabled
3. **主干结构**：uni-directional / bi-directional / no gated FFN
4. **正则项**：no entropy reg / no symmetry reg / full

---

## 硬件适配

| 环境 | 推荐配置 |
|---|---|
| 本地调试（Win11 RTX4070S）| `batch_size=8`, `fp16=false`, `num_workers=2` |
| 云端训练（Linux A100）| `batch_size=64`, `fp16=true`, `num_workers=8` |

路径统一使用 `pathlib.Path`，兼容 Win11 + Linux。

---

## 已知限制

| 项目 | 说明 |
|---|---|
| ESM-C 权重 | 需自行获取，不随仓库分发 |
| 最大序列长度 | 默认截断至 1024 aa |
| 多 GPU | 当前仅支持单卡【待扩展】 |
| Triplet 方案 | 需配合 `sampler.hard_neg=true` 使用 |

---

## License

MIT License
