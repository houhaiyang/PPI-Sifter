# PPI-Sifter 项目设计书（对比学习版）

> **文档版本：** v1.0  
> **创建时间：** 2026-06  
> **适用阶段：** 新建项目，全量实现  
> **技术栈：** Python 3.10 · PyTorch 2.1.0 · CUDA 12.1 · ESM-C · HDF5

---

## 一、项目概述

### 1.1 项目定位

PPI-Sifter 是一个面向高通量蛋白质相互作用（PPI）筛选与机制分析的轻量序列模型。项目核心目标有两个：

1. **预测**：对任意候选蛋白对给出可靠的 pair-level interaction probability，支持高通量筛选场景；
2. **解释**：在不依赖 residue-level 界面标注的前提下，通过对比学习与层级表征分析，回答模型内部是否学到了 partner-aware interaction pattern，以及这种模式在哪一层、以何种方式形成。

### 1.2 技术路线概览

```
冻结 ESM-C 离线 Embedding
        ↓
共享输入投影层
        ↓
N 层双向 Cross-Attention（B-PPI 风格主干）+ Gated FFN
        ↓
Attention Pooling → 对称共享表示 → MLP 分类头
        ↓（训练时并联）
LayerwiseContrastHead（投影头 + SupCon / Triplet / InfoNCE）
        ↓（推理 / 评估时并联）
解释分析链路（layer separability · partner shift · entropy gap · UMAP）
```

### 1.3 核心创新点

| 创新维度 | 描述 |
|---|---|
| **对比学习增强解释** | 以 supervised contrastive loss 强化各层 pair representation 的正负可分性，形成可量化的机制解释 |
| **Layer-wise separability** | 对每层中间表示独立训练线性 probe，量化各层 AUROC，定位 partner-aware pattern 的深度位置 |
| **Partner shift analysis** | 固定 anchor 蛋白，比较真实 partner 与负样本 partner 导致的中间表示偏移，验证 partner-conditioned 编码 |
| **Attention entropy profiling** | 跨层统计正负样本的 attention entropy 差异，用于论文的 mechanistic 论证 |
| **轻量工程路线** | ESM-C 冻结，主干参数量低，适配单卡 RTX4070S 本地调试 + 云端 A100 训练 |

---

## 二、研究背景

蛋白质相互作用预测的困难不只在于分类精度，更在于**可解释性**。仅输出 pair-level 概率的模型难以支撑机制分析与论文论证。现有方法通常直接输出 attention map 热图作为解释结果，但在只有 pair-level 0/1 标签的训练设定下，attention 往往退化为可视化附属品，并不稳定对应真实生物界面。

本项目选择一条更适合当前资源约束与 BIBM 2026 发表目标的解释路线：**以对比学习目标结构化正负样本的 cross-attention 中间空间，再通过层级量化分析回答模型的判别机制**，而不是依赖 residue-level 真值监督或复杂多任务框架。

---

## 三、问题定义

本项目需要回答三个相互关联的核心问题：

1. **预测问题**：如何在高通量候选蛋白对集合上，在未见蛋白和远同源蛋白上，输出可靠的 pair-level interaction probability？

2. **机制问题**：cross-attention 模块是否学到了 partner-aware interaction pattern？即同一蛋白在面对不同 partner 时，其中间表征和注意力分布是否发生系统性的、正负样本可区分的变化？

3. **可分析问题**：如何把"机制解释"从主观可视化提升为可量化、可复现的实验对象？

---

## 四、设计原则

### 4.1 主干对齐 B-PPI 风格

判别主干严格遵循 B-PPI 范式：共享投影 → 双向 multi-head cross-attention → gated FFN → attention pooling → 对称共享表示 → MLP 分类头。不对主干结构做任何不必要的改动。

### 4.2 解释目标定义为机制可分性

解释性不等于"输出真实界面热图"，而定义为：**模型内部是否形成了正负样本可区分的 partner-aware 交互模式，以及这种模式能否通过层级分析和对比学习被稳定捕获。**

### 4.3 轻量优先，快速迭代

不引入结构标注、复杂多任务监督。解释增强仅通过现有 pair-level 标签与中间表征分析完成，保证在 RTX4070S 上可完成 debug，在 A100 上完成完整训练。

### 4.4 严格工程分层

遵循 AI 模型设计规范：模型层、数据层、训练层、推理层、评估层、解释层严格解耦。超参统一由 YAML 配置驱动，禁止硬编码。

### 4.5 可复现

架构、超参、数据划分、随机种子全部固化，保证实验结论可复现。消融配置单独一个 YAML 文件管理。

---

## 五、模型结构

### 5.1 输入与离线编码

**输入**：两条蛋白质序列 A、B，长度分别为 \(L_A\)、\(L_B\)。

**编码器**：冻结的 ESM-C，离线提取 residue-level embeddings，存入 HDF5：

\[
X_A \in \mathbb{R}^{L_A \times d_{in}}, \quad X_B \in \mathbb{R}^{L_B \times d_{in}}
\]

其中 \(d_{in}\) 为 ESM-C 输出维度（默认 1152）。ESM-C 不参与端到端微调，仅作为特征提取器。

### 5.2 共享输入投影

\[
H_A^{(0)} = X_A W_p + b_p, \quad H_B^{(0)} = X_B W_p + b_p
\]

其中 \(W_p \in \mathbb{R}^{d_{in} \times d}\)，\(d\) 为任务空间维度（默认 256）。两条蛋白共享同一投影矩阵。

### 5.3 双向 Cross-Attention 编码器

模型包含 \(N\) 层（默认 \(N=2\)）双向 cross-attention block。第 \(l\) 层：

**A→B 方向**（A 为 query，B 为 key/value）：
\[
\tilde{H}_A^{(l)} = \mathrm{MultiHeadCrossAttn}\!\left(Q=H_A^{(l-1)},\, K=H_B^{(l-1)},\, V=H_B^{(l-1)}\right)
\]

**B→A 方向**（B 为 query，A 为 key/value）：
\[
\tilde{H}_B^{(l)} = \mathrm{MultiHeadCrossAttn}\!\left(Q=H_B^{(l-1)},\, K=H_A^{(l-1)},\, V=H_A^{(l-1)}\right)
\]

两个方向分别经过残差连接与 LayerNorm，输出 \(H_A^{(l)}\) 与 \(H_B^{(l)}\)。

**每一层额外保存：**

- `attn_map_ab_l`：A→B 方向 attention weights，形状 \([B, \text{heads}, L_A, L_B]\)
- `attn_map_ba_l`：B→A 方向 attention weights，形状 \([B, \text{heads}, L_B, L_A]\)
- `hidden_a_l`、`hidden_b_l`：各层输出 hidden states

这些保存数据**不改变前向主链**，仅在 `return_layer_reprs=True` 时追加到返回 dict。

### 5.4 Gated FFN

每层 cross-attention 后接 Gated FFN：

\[
\mathrm{GatedFFN}(H) = (W_1 H \odot \sigma(W_g H)) W_2
\]

作用：以 partner-conditioned 表示为输入，门控筛选保留判别性成分，精炼后输出。

### 5.5 Attention Pooling

最后一层 \(H_A^{(N)}\)、\(H_B^{(N)}\) 各自做 attention pooling，得到固定长度 pair-level global representation：

\[
s_A = \mathrm{AttnPool}(H_A^{(N)}), \quad s_B = \mathrm{AttnPool}(H_B^{(N)})
\]

其中 attention pooling 使用一层 linear 计算权重，softmax 后加权求和。

### 5.6 对称共享表示

\[
V = \bigl[(s_A + s_B) \;\|\; |s_A - s_B|\bigr]
\]

满足输入顺序置换不变性，维度 \(2d\)。

### 5.7 MLP 分类头

\[
\hat{y} = \sigma\!\left(\mathrm{MLP}(V)\right)
\]

轻量 MLP（2 层，中间 ReLU + Dropout），输出 pair-level interaction probability。

---

## 六、对比学习模块

### 6.1 核心思想

在主分类任务之外，对若干层的中间 pair representation 施加监督式对比损失，目的是：

- 强化正样本在中间空间中的聚类紧致性；
- 扩大负样本与正样本的表示距离；
- 使 partner-aware 模式在中间层中更早出现、更可分析。

这一模块**完全独立于主干**，训练时并联，推理时可关闭（`contrast.enabled=false` 退化为原版）。

### 6.2 可解释中间表征定义

对每一层 \(l \in \{1, \ldots, N\}$，提取以下表征作为对比对象：

| 名称 | 定义 | 形状 |
|---|---|---|
| `pooled_a_l` | 第 \(l\) 层 \(H_A^{(l)}\) 的 mean pooling | \([B, d]\) |
| `pooled_b_l` | 第 \(l\) 层 \(H_B^{(l)}\) 的 mean pooling | \([B, d]\) |
| `pair_repr_l` | \([(s_A^l + s_B^l) \| \|s_A^l - s_B^l\|]\)，同主干对称构造 | \([B, 2d]\) |
| `attn_stats_l` | entropy, head variance, symmetry error | dict |

核心对比对象为 `pair_repr_l`，因为它直接反映了该层 partner-aware interaction 的压缩表示。

### 6.3 投影头

对比学习不直接在 `pair_repr_l` 原始空间做，而是先经过轻量投影头降维至对比空间：

\[
z_l = \mathrm{Proj}_l(\text{pair\_repr\_l}), \quad z_l \in \mathbb{R}^{d_c}
\]

\(d_c\) 默认 128，投影头为 2 层 MLP + L2 normalize。

### 6.4 对比损失

**方案 A — Supervised Contrastive Loss（推荐默认）：**

\[
L_{\mathrm{supcon}}(z_l, y) = \sum_{i} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{k \neq i} \exp(z_i \cdot z_k / \tau)}
\]

其中 \(P(i)\) 为 batch 内与样本 \(i\) 同标签的正样本集合，\(\tau\) 为温度（默认 0.07）。

**方案 B — Anchor-Partner Triplet Loss（可选）：**

固定 anchor A，真实 partner \(B^+\) 与 hard negative \(B^-\) 形成 triplet：

\[
L_{\mathrm{triplet}} = \max\!\left(0,\; \|z_A - z_{B^+}\|_2 - \|z_A - z_{B^-}\|_2 + m\right)
\]

margin \(m\) 默认 0.5。

**多层加权求和：**

\[
L_{\mathrm{con}} = \sum_{l \in \mathcal{L}_{\mathrm{active}}} \alpha_l \cdot L_l
\]

默认 `active_layers: [1, 2]`，\(\alpha_l\) 统一，可配置。

---

## 七、损失函数

### 7.1 主分类损失

\[
L_{\mathrm{cls}} = L_{\mathrm{wbce}} + \lambda_1 L_{\mathrm{focal}}
\]

weighted BCE 处理类别不平衡；focal loss 加大难样本权重。

### 7.2 轻量 Attention 正则

虽然 attention map 不是主要解释对象，仍保留三项轻量约束：

\[
L_{\mathrm{reg}} = \lambda_2 L_{\mathrm{sparse}} + \lambda_3 L_{\mathrm{sym}} + \lambda_4 L_{\mathrm{entropy}}
\]

| 项 | 作用 |
|---|---|
| \(L_{\mathrm{sparse}}\) | 抑制全图均匀扩散，使注意力聚焦 |
| \(L_{\mathrm{sym}}\) | 约束 A→B 与 B→A 的对称一致性 |
| \(L_{\mathrm{entropy}}\) | 鼓励正样本注意力更聚焦，负样本更平坦 |

### 7.3 总损失

\[
L = L_{\mathrm{cls}} + \beta_1 L_{\mathrm{con}} + \beta_2 L_{\mathrm{reg}}
\]

训练初期建议 \(\beta_1 = 0.1\) 热身，待 \(L_{\mathrm{cls}}\) 趋于稳定后提升至 0.5。

### 7.4 关键超参汇总

| 参数 | 默认值 | 说明 |
|---|---|---|
| `lambda_1` | 0.5 | focal loss 权重 |
| `lambda_2` | 0.1 | sparse reg 权重 |
| `lambda_3` | 0.1 | sym reg 权重 |
| `lambda_4` | 0.05 | entropy reg 权重 |
| `beta_1` | 0.1~0.5 | contrast loss 权重 |
| `beta_2` | 0.1 | attention reg 权重 |
| `contrast_tau` | 0.07 | 对比温度 |
| `d_model` | 256 | 任务空间维度 |
| `n_layers` | 2 | cross-attention 层数 |
| `n_heads` | 8 | attention 头数 |

---

## 八、解释分析框架

> **原则：解释模块与训练主干完全解耦，所有分析在推理后的中间表示上离线执行。**

### 8.1 Layer-wise Separability Analysis

**问题**：哪一层的 pair representation 对正负样本区分力最强？

**做法**：
1. 推理阶段导出全部测试集各层 `pair_repr_l`；
2. 对每层独立训练 logistic probe（线性分类器，无正则）；
3. 报告各层 probe AUROC / AUPRC；
4. 辅以 silhouette score 与 Davies-Bouldin index；
5. t-SNE / UMAP 可视化各层聚类结构。

**期望结论**：若某中间层 AUROC 接近最终分类头，说明 partner-aware pattern 在该层已充分形成。

### 8.2 Anchor-Partner Shift Analysis

**问题**：同一蛋白面对不同 partner，中间表示是否发生 partner-conditioned 的可识别偏移？

**做法**：固定 anchor protein A，分别与真实 partner \(B^+\)、随机负 partner \(B^-\)、hard negative partner \(B^{-}_{\mathrm{hard}}\) 配对，计算各层表示偏移：

\[
\Delta_l(A;\, B_1, B_2) = \bigl\|\mathrm{Pool}(H_A^{(l)}(A, B_1)) - \mathrm{Pool}(H_A^{(l)}(A, B_2))\bigr\|_2
\]

若 \(\Delta_l(A; B^+, B^-) \gg \Delta_l(A; B^-, B^-_2)\)，则验证了 partner-conditioned 编码。

### 8.3 Attention Entropy Gap Profile

对每一层的 attention map 计算正负样本的 entropy 差异：

\[
\mathrm{EntropyGap}_l = \mathbb{E}_{y=1}[H_l] - \mathbb{E}_{y=0}[H_l]
\]

其中 \(H_l\) 为第 \(l\) 层 attention row entropy 均值。负值（正样本 entropy 更低）表明正样本注意力更聚焦，具备生物学合理性。

辅助统计量：
- head-wise variance
- symmetry consistency error
- top-k token stability across seeds

### 8.4 Cross-Partner Perturbation（可选）

推理阶段对固定蛋白 A 替换 partner，或对 partner 做局部 masking / shuffle，观察：
- 各层表示变化幅度（early layers 应变化小，deep layers 应变化大）；
- 最终预测分数变化趋势。

此分析可形成论文中的消融案例，验证 cross-attention 而非输入 embedding 本身决定了判别结果。

---

## 九、数据流程

### 9.1 输入格式

训练数据为蛋白对 CSV 表，核心字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `protein_a` | str | UniProt ID 或序列 |
| `protein_b` | str | UniProt ID 或序列 |
| `label` | int | 1=互作，0=非互作 |
| `split` | str | train / val / test |
| `negative_type` | str（可选）| random / hard / curated |

### 9.2 Embedding 缓存（HDF5）

所有蛋白离线提取 ESM-C embeddings，存入 HDF5：

```
embeddings.h5
├── {protein_id}/
│   ├── embedding   # L × 1152 float16
│   └── length      # int
```

支持断点续写，训练时按 protein_id 随机访问，避免全量加载。

### 9.3 数据划分策略

| 划分策略 | 用途 |
|---|---|
| Protein-disjoint | 主实验，防止训练蛋白泄露到测试集 |
| Cluster-disjoint | 严格泛化测试，序列同源性控制 |
| Random split | 仅用于 smoke test |

类别比例建议：训练集正负 1:4 负采样，验证集与测试集使用自然比例。

### 9.4 中间表示导出格式

推理阶段额外导出至 `outputs/layer_reprs/`：

```
outputs/
├── layer_reprs/
│   ├── pair_repr_l{0-N}.pt      # [n_samples, 2d] per layer
│   ├── attn_stats_l{0-N}.json   # entropy, variance, sym_error
│   └── predictions.csv          # protein_a, protein_b, prob, label
└── analysis/
    ├── layer_separability.json
    ├── partner_shift.json
    └── attention_entropy_profile.json
```

---

## 十、工程目录结构

遵循 AI 模型设计规范标准目录，完整结构如下：

```
PPI-Sifter/
│
├── docs/
│   ├── PPI-Sifter-项目设计书（对比学习）.md        # 本文档
│   └── 对比学习解释性分析流水线.md
│
├── configs/
│   ├── default.yaml            # 默认训练配置（含 contrast 块）
│   ├── contrast.yaml           # 消融实验配置（各对比方案）
│   └── interpret.yaml          # 推理 + 解释导出配置
│
├── data/
│   ├── dataset.py              # PPIDataset，HDF5 按需读取
│   ├── sampler.py              # 负采样、batch 构建
│   └── split.py                # protein-disjoint / cluster-disjoint 划分
│
├── model/
│   ├── __init__.py
│   ├── model.py                # PPISifter 主模型（含 return_layer_reprs 接口）
│   ├── attention.py            # MultiHeadCrossAttention
│   ├── ffn.py                  # GatedFFN
│   ├── pooling.py              # AttentionPooling
│   ├── contrast.py             # LayerwiseContrastHead（投影头 + 前向）
│   └── losses.py               # PPILoss（返回 loss dict）
│
├── train/
│   ├── train.py                # 训练主入口
│   ├── trainer.py              # Trainer class，封装训练循环
│   └── scheduler.py            # 学习率调度
│
├── inference/
│   ├── infer.py                # 批量推理 + layer_reprs 导出
│   └── predict.py              # 单对推理接口
│
├── evaluate/
│   ├── eval.py                 # 标准评估（AUROC, AUPRC, F1, MCC）
│   └── metrics.py              # 指标计算工具
│
├── scripts/
│   ├── emb/
│   │   └── extract_esmc.py     # 离线提取 ESM-C embeddings → HDF5
│   ├── data/
│   │   └── build_pairs.py      # 构建训练对 CSV（含 BioGRID 预处理）
│   └── interpretability/
│       ├── run_contrast_analysis.py   # 核心分析入口（separability+shift+entropy）
│       ├── plot_interpret.py          # 绘图：热图 / UMAP / entropy 折线图
│       └── quantify_interpret.py      # 精细化 attention 量化（备用）
│
├── utils/
│   ├── logger.py               # 结构化日志
│   ├── seed.py                 # 随机种子固化
│   ├── io.py                   # HDF5 读写工具
│   └── misc.py                 # 其他公共工具
│
├── checkpoints/                # 模型权重保存（gitignore）
├── outputs/                    # 推理结果、分析结果（gitignore）
├── references/                 # B-PPI 论文、ESM-C 文档等参考资料
├── requirements.txt
└── README.md
```

---

## 十一、配置文件规范

### `configs/default.yaml` 核心结构

```yaml
project:
  name: ppi-sifter
  version: "1.0"
  seed: 42

data:
  hdf5_path: data/BIOGRID/embeddings/embeddings.h5
  pairs_csv: data/BIOGRID/pairs_train.csv
  split_strategy: protein_disjoint
  neg_ratio: 4
  max_len_a: 1024
  max_len_b: 1024

model:
  d_input: 1152          # ESM-C 输出维度
  d_model: 256
  n_layers: 2
  n_heads: 8
  dropout: 0.1
  ffn_multiplier: 4

contrast:
  enabled: true
  active_layers: [1, 2]   # 从第 1 层起标记（1-indexed）
  method: supcon           # supcon | triplet | infonce
  d_proj: 128
  temperature: 0.07
  margin: 0.5              # triplet 专用
  lambda_contrast: 0.3

loss:
  lambda_focal: 0.5
  lambda_sparse: 0.1
  lambda_sym: 0.1
  lambda_entropy: 0.05
  beta_contrast: 0.3
  beta_reg: 0.1

train:
  epochs: 50
  batch_size: 32
  lr: 1e-4
  weight_decay: 1e-4
  grad_clip: 1.0
  warmup_steps: 500
  fp16: false
  early_stopping_patience: 10
  save_metric: auprc

hardware:
  device: cuda
  num_workers: 4
  pin_memory: true
```

---

## 十二、训练流程

```
初始化（模型 + 对比头 + 优化器 + 调度器）
        ↓
加载 HDF5 + pairs CSV → DataLoader
        ↓
for each epoch:
    for each batch:
        前向 → loss_dict = PPILoss(…)
        loss_dict["total"].backward()
        梯度裁剪 → optimizer.step() → scheduler.step()
        记录 loss_dict 各分项到 TensorBoard
        ↓
    每 epoch 末：val 评估 → 若 AUPRC 最优 → 保存 checkpoint
    early stopping 检查
        ↓
训练结束 → 保存 best_auprc.pt（model + contrast_head state_dict）
```

**Checkpoint 命名规范：**

```
checkpoints/
├── best_auprc.pt           # 最优 AUPRC checkpoint（主要）
├── best_auroc.pt           # 最优 AUROC checkpoint（备用）
├── epoch_{N}.pt            # 每 10 epoch 定期保存
└── last.pt                 # 最新 epoch
```

每个 `.pt` 包含：`model_state_dict`, `contrast_head_state_dict`, `optimizer_state_dict`, `epoch`, `metrics`, `config`。

---

## 十三、评测方案

### 13.1 主任务指标

| 指标 | 说明 |
|---|---|
| **AUPRC** | 主指标，类别不平衡下最具代表性 |
| AUROC | 辅助指标 |
| F1 @ threshold=0.5 | 点估计参考 |
| MCC | Matthews 相关系数，平衡精度度量 |
| Precision / Recall curve | 用于论文图表 |

### 13.2 解释性分析指标

| 指标 | 回答的问题 |
|---|---|
| Layer probe AUROC/AUPRC（per layer） | 哪一层形成了可分离的 pair-level 表示？ |
| Silhouette score（per layer） | 正负样本的表示聚类质量如何？ |
| Davies-Bouldin index | 类内紧密度 vs 类间分离度 |
| Partner shift gap \(\Delta_l\) | 模型是否以 partner-conditioned 方式编码信息？ |
| Attention entropy gap（per layer） | 正样本的注意力是否比负样本更聚焦？ |
| Symmetry consistency error | 双向 attention 是否一致？ |
| Seed stability（跨 3 个 seed） | 层级结论是否稳定，而非偶然？ |

### 13.3 Benchmark 模型

| 模型 | 说明 |
|---|---|
| Mean-pooling + MLP | 最简 baseline |
| Siamese dual encoder | 无交互，序列级表示 |
| B-PPI style（无对比分支） | 验证主干贡献 |
| PPI-Sifter（无对比分支） | 消融对比学习贡献 |
| **PPI-Sifter（full）** | 本项目最终模型 |

---

## 十四、消融实验设计

### 14.1 对比层选择消融

| 配置 | active_layers |
|---|---|
| Layer 1 only | `[1]` |
| Layer 2 only | `[2]` |
| All layers | `[1, 2]` |
| No contrast | `enabled: false` |

### 14.2 对比方案消融

| 配置 | method |
|---|---|
| Supervised Contrastive | `supcon` |
| Anchor-Partner Triplet | `triplet` |
| InfoNCE | `infonce` |
| 无对比损失 | disabled |

### 14.3 主干结构消融

| 配置 | 改动 |
|---|---|
| No cross-attention | 替换为 self-attention |
| Uni-directional attention | 只保留 A→B |
| Bi-directional attention | 双向（默认） |
| Remove gated FFN | FFN 替换为标准 FFN |

### 14.4 正则项消融

| 配置 | 改动 |
|---|---|
| No entropy reg | `lambda_entropy: 0` |
| No symmetry reg | `lambda_sym: 0` |
| Full reg | 默认 |

---

## 十五、硬件适配说明

| 环境 | 配置 |
|---|---|
| **本地调试（Win11 RTX4070S）** | `batch_size: 8`, `n_layers: 2`, `fp16: false`, `num_workers: 2` |
| **云端训练（Linux A100）** | `batch_size: 64`, `n_layers: 2~4`, `fp16: true`, `num_workers: 8` |

本地专用于 bug 定位，完整训练在云端执行。路径使用相对路径，兼容两端。

---

## 十六、依赖与运行

### 环境依赖

```
Python       3.11
PyTorch      2.1.0 + CUDA 12.1
esm          >=2.0.0   (ESM-C 推理)
h5py         >=3.9.0
numpy        >=1.24.0
pandas       >=2.0.0
scikit-learn >=1.3.0
umap-learn   >=0.5.0
matplotlib   >=3.7.0
seaborn      >=0.12.0
pyyaml       >=6.0
tensorboard  >=2.14.0
tqdm         >=4.65.0
```

### 关键运行命令

```bash
# 步骤 0：离线提取 embedding
python scripts/emb/extract_esmc.py --config configs/default.yaml

# 步骤 1：训练
python train/train.py --config configs/default.yaml

# 步骤 2：评估
python evaluate/eval.py --config configs/default.yaml --ckpt checkpoints/best_auprc.pt

# 步骤 3：推理 + 中间表示导出
python inference/infer.py --config configs/interpret.yaml --ckpt checkpoints/best_auprc.pt

# 步骤 4：对比机制分析
python scripts/interpretability/run_contrast_analysis.py --config configs/interpret.yaml

# 步骤 5：可视化（论文图，可选）
python scripts/interpretability/plot_interpret.py
```

---

## 十七、已知限制与兼容说明

| 项目 | 说明 |
|---|---|
| ESM-C 权重 | 需自行从 EvolutionaryScale 官方获取，不随本仓库分发 |
| 最大序列长度 | 默认截断至 1024，超长蛋白需手动调整 `max_len` 配置 |
| 多 GPU | 当前仅支持单卡，DDP 多卡支持标注【待验证扩展】 |
| Windows 路径 | 路径统一使用 `pathlib.Path`，兼容 Win11 + Linux |
| `contrast.method=triplet` | 要求 batch 内存在 hard negative pair，需配合 `sampler.hard_neg=true` 使用 |

---

*文档结束*
