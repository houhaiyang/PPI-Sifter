# PPI-Sifter benchmark + ablation 实验设置及数据集划分指导

## 文档目标

本文档用于指导 PPI-Sifter 的实验设计、数据集划分、benchmark 对比、ablation 拆解与工程落地，目标是让实验在学术上可解释、在实现上可复现、在写作上可直接进入论文方法与实验部分。

实验设计的核心原则不是“把更多模型跑一遍”，而是确保所有模型在**同一无泄漏协议**下训练、验证和测试。已有研究表明，蛋白互作 benchmark 中如果只按元数据、PDB 编号或普通序列相似性切分，仍可能产生严重的数据泄漏，从而导致过于乐观的评估结果；在蛋白相互作用 benchmark 中，不同常见切分方式的泄漏比例可高达 86%、65%、61% 和 30%。

对 PPI-Sifter 而言，建议将实验目标分成三层：

- 第一层：pair-level PPI 分类性能是否优于 baseline。
- 第二层：在更严格的 protein-disjoint / cluster-disjoint / species-disjoint 条件下是否仍具优势。
- 第三层：residue-level attention map 与 top-k residue pairs 是否提供稳定、可分析的解释信号。

## 总体实验原则

### 1. 所有 baseline 尽量在统一 split 上重新训练

不要直接拿文献里的最终分数拼表。B4PPI 这类 benchmark pipeline 之所以有价值，就在于它提供了固定的训练集与多个测试集分区，目的是让模型在统一协议下可比较，而不是复用异构设置下的外部分数。

推荐采用以下规范：

- 所有可复现 baseline，都在你的 split 上重新训练。
- 统一输入来源，或者至少统一为“各模型最标准、最公平”的输入版本。
- 统一训练预算，如 epoch 上限、early stopping、batch size 搜索范围、学习率搜索范围。
- 统一阈值选择规则，阈值只能在 validation set 上选，不能看 test set。
- 统一负样本比例或在表中明确列出差异。

如果某个模型没有代码或无法稳定复现，则单独放在“文献参考结果”附表中，并明确注明“不同数据集 / 不同 split / 不可直接横向比较”。

### 2. 先定 split，再做一切特征处理

数据泄漏不只出现在监督训练阶段，也会出现在特征工程、归一化、采样与阈值调优阶段。通用机器学习规范要求任何依赖数据分布的处理步骤都只能在训练集上拟合，再应用到验证集与测试集。

因此，对 PPI-Sifter 的所有实验，建议遵守以下顺序：

1. 先完成蛋白级或簇级划分。
2. 再确定 train / valid / test 文件。
3. 再基于 train 构建负样本采样器、归一化统计量、类别权重。
4. 再训练模型。
5. 只在 valid 上做模型选择与阈值选择。
6. 最后一次性在 test 上汇报结果。

### 3. 随机划分只能做附录，不应做主结论

随机 pair split 可以作为“乐观上限”，但它无法反映模型对未见蛋白、未见同源簇、未见物种或未见界面模式的真实泛化能力。已有分析指出，蛋白互作预测在低同源或更严格泛化条件下仍然远未解决，因此主文应优先强调严格切分下的表现。

## 推荐 benchmark 框架

建议把主实验分成四个 benchmark setting，每个 setting 都用相同的 baseline 列表与相同的评估指标。

| Setting | 目标 | 划分单位 | 难度 | 是否作为主表 |
|---|---|---|---|---|
| Random pair split | 测试插值能力 | 蛋白对 | 低 | 否，建议附录 |
| Protein-disjoint | 测试未见蛋白泛化 | 单个蛋白 | 中 | 是 |
| Cluster-disjoint | 测试远同源泛化 | 序列簇 | 高 | 是 |
| Species-disjoint | 测试跨物种迁移 | 物种 / taxa | 高 | 是 |

如果后续有结构或界面标注数据，还可以加：

| Setting | 目标 | 划分单位 | 备注 |
|---|---|---|---|
| Interface-disjoint | 测试未见结合模式 | 复合体界面 | 对真实泛化最严格，但实现成本最高 |
| Structure-disjoint | 测试未见结构模板 | 结构簇 / complex family | 适合 docked complex 或已知复合体数据 |

## 数据集构建建议

### 正样本来源

如果继续沿用 bacterial PPI 场景，可参考 B-PPI 的思路：使用细菌专属 PPI 数据，并围绕特定生物学域构建训练语料。B-PPI 论文摘要显示，该方法在名为 B-PPI-DB 的细菌数据集上训练，数据来自 STRING，并包含 202,829 个正负样本、2646 个 taxa、正负比例 1:10。

对 PPI-Sifter，正样本表建议至少包含以下字段：

- protein_a_id
- protein_b_id
- label
- species / taxa
- source_db
- evidence_score
- sequence_a
- sequence_b
- optional: structure_id / interface_id / operon / localization

### 负样本构造

负样本比正样本更容易引入偏差，因此建议同时维护两种负样本协议：

- Easy negatives：随机不成对蛋白，适合训练前期和快速调试。
- Hard negatives：同物种、相近长度、相近局部化或同功能背景下但未记录互作的蛋白对，更适合作为主 benchmark。

论文中应明确负样本是如何采样的，以及采样是否跨 split 进行。最安全的做法是：先完成 split，再在各 split 内独立生成或筛选负样本，避免测试蛋白的信息反向影响训练采样策略。

### 去冗余与清洗

对于序列级 benchmark，建议至少做两层处理：

- 原始蛋白去重：完全相同序列或完全相同 accession 去重。
- 序列聚类：使用 MMseqs2 或 CD-HIT 对蛋白做聚类，再基于簇进行更严格划分。

一些近期工作会在无重叠蛋白协议下进一步限制序列相似性，例如只保留 sequence identity 不高于 40% 的蛋白对，以减轻同源泄漏问题。

## 数据集划分协议

### 1. Random pair split

定义：直接按蛋白对随机分配 train / valid / test。

优点：

- 实现简单。
- 便于快速调试模型是否能收敛。
- 适合早期工程验证。

缺点：

- train 和 test 往往共享大量蛋白。
- 容易高估模型性能。
- 不适合作为主结论。

建议：只用于 smoke test、debug、附录或与旧工作形式对齐。

### 2. Protein-disjoint split

定义：测试集中的任意蛋白，在训练集和验证集中都不能出现。

实现规则：

1. 先把所有唯一蛋白整理成节点集合。
2. 在蛋白级别切分为 train proteins、valid proteins、test proteins。
3. 仅保留两端蛋白都属于同一 split 的 pair；跨 split pair 直接丢弃，或单独记为不可用样本。

这是最基本的“未见蛋白”评测协议，也是 sequence-based PPI 任务中最应优先采用的主 benchmark 之一。

### 3. Cluster-disjoint split

定义：先按序列相似性把蛋白聚成簇，再以簇为单位做切分；测试簇中的任何蛋白都不能与训练簇存在近同源重叠。

实现规则：

1. 对全部蛋白跑 MMseqs2 easy-cluster。
2. 设定 identity threshold，例如 0.3、0.4 或 0.5。
3. 以 cluster id 为最小单位分配到 train / valid / test。
4. 生成 pair 时，要求 pair 两端蛋白对应的 cluster 都来自当前 split。

相比 protein-disjoint，cluster-disjoint 更难，因为它禁止模型通过“训练时见过近同源蛋白”间接泛化到测试集。

### 4. Species-disjoint split

定义：按物种、菌株、属或更高层级 taxa 划分，使测试物种在训练中完全未出现。

适用场景：

- 细菌跨物种泛化。
- 希望强调模型的 domain transfer 能力。
- 希望更贴近“新菌种数据很少”的实际应用场景。

实现规则：

1. 为每个蛋白挂接 species / taxa 元信息。
2. 先按物种集合分 train / valid / test。
3. 只保留同 split 内物种对应的蛋白对。

如果数据规模允许，species-disjoint 很适合作为论文主表之一，因为它能直接体现模型是否只是记住了物种背景。

### 5. Interface-disjoint / structure-disjoint split（可选增强）

如果未来引入结构复合体数据，需要格外警惕“界面级近重复”。已有研究指出，仅按 PDB code、提交时间或普通序列相似性划分，仍可能让训练集和测试集共享高度相似的相互作用界面，从而造成严重泄漏。

因此，结构任务的更严格做法是：

- 优先以 complex family 或 interface similarity 为单位切分。
- 不要把同一复合体的不同 pose、不同轻微变体分到不同 split。
- 若条件允许，对 train-test 间界面做相似性审计并报告 leakage ratio。

## 泄漏防控清单

在实验开始前，建议把下面这份清单做成自动检查脚本：

- 训练集与测试集是否共享 accession。
- 训练集与测试集是否共享完全相同序列。
- 训练集与测试集是否存在高于阈值的 sequence identity。
- 训练集与测试集是否共享同一 cluster id。
- 训练集与测试集是否共享同一 species / taxa。
- 若有结构，训练集与测试集是否共享近重复 interface。
- 负样本生成是否使用了 test 集信息。
- 类别权重、标准化参数、阈值是否只基于 train / valid 拟合。
- 模型选择是否只看 valid，不看 test。

建议把这些检查写入数据构建流水线，每次重新生成 split 时自动输出一个 leakage report。

## Baseline 选择建议

建议把 baseline 分为四类，保证比较具有解释性而不是只堆模型名。

### A. 简单序列或 embedding 拼接模型

作用：证明 cross-attention 的必要性。

候选形式：

- Mean pooling + MLP
- Max pooling + MLP
- Concatenate pooled embeddings + MLP
- Bilinear head / cosine-style interaction head

### B. 单塔或双塔编码模型

作用：证明“先独立编码，再简单融合”是否足够。

候选形式：

- Siamese encoder + concat head
- Dual encoder + cross feature head
- Frozen PLM + shallow interaction head

### C. 交互建模模型

作用：与 PPI-Sifter 的核心骨干直接对比。

可包含：

- 单向 cross-attention
- 双向 cross-attention
- Cross-attention without gated FFN
- Cross-attention + mean pooling

B-PPI 的核心思路就是使用 ProstT5 embedding 加 cross-attention 建模蛋白间关系，并在细菌 PPI 上显著优于其对比模型 TT3D；摘要报告其 AUPRC 为 0.926，而 TT3D 为 0.230，F1 为 0.866，而 TT3D 为 0.299。

### D. 文献中的代表性外部模型

作用：回答“和已有工作相比处于什么水平”。

如果对方代码可运行，建议重训；如果不可运行，则放附录做参考。已有工作中，GTE-PPIS 这类方法会专门对结构模块、图模块、残差连接和中间表示做 ablation，并在独立测试集上报告 AUROC / AUPRC 变化，这种写法很适合借鉴。

## PPI-Sifter 的 ablation 设计

建议按“输入表征、交互主干、池化与输出、解释性正则”四组做系统 ablation，而不是无序删模块。

### 1. 输入表征 ablation

目的：回答性能提升到底来自 ESM-C，还是来自后端交互架构。

| 组别 | 变体 | 要回答的问题 |
|---|---|---|
| Emb-1 | ProstT5 residue embedding | 与 B-PPI 输入一致时表现如何 |
| Emb-2 | Frozen ESM-C residue embedding | 当前主设定是否更优 |
| Emb-3 | ESM-C + linear projection | 投影层是否必要 |
| Emb-4 | Mean pooled protein embedding only | residue-level 表征是否关键 |

### 2. 交互骨干 ablation

目的：回答 cross-attention 与双向交互是否真正必要。

| 组别 | 变体 | 要回答的问题 |
|---|---|---|
| Int-1 | No cross-attention | 不建模链间显式交互会怎样 |
| Int-2 | Unidirectional attention | 双向是否比单向更好 |
| Int-3 | Bidirectional cross-attention | 主模型 |
| Int-4 | Remove gated FFN | gated FFN 是否有贡献 |
| Int-5 | Replace with plain FFN | gate 的价值是否来自非线性还是门控 |

### 3. 池化与分类头 ablation

目的：回答最终 pair-level 表征如何形成最合理。

| 组别 | 变体 | 要回答的问题 |
|---|---|---|
| Head-1 | Mean pooling | 简单平均是否足够 |
| Head-2 | Attention pooling | 当前主设定 |
| Head-3 | Max pooling | 是否更偏向局部强信号 |
| Head-4 | Concat only | 是否必须引入差分或乘性特征 |
| Head-5 | Symmetry fusion off | 对称融合是否稳定预测 |

### 4. 解释性输出与正则 ablation

目的：回答“可解释性分支是否真的有效，以及是否损害主任务性能”。

| 组别 | 变体 | 要回答的问题 |
|---|---|---|
| Exp-1 | No residue-level branch | 解释分支是否影响主任务 |
| Exp-2 | Attention map only | 仅输出热图是否足够 |
| Exp-3 | Top-k residue pairs only | 离散解释输出是否更稳定 |
| Exp-4 | Sparse regularization off | 稀疏正则是否提高可解释性 |
| Exp-5 | Symmetry / consistency loss off | 一致性约束是否必要 |

### 5. 训练目标 ablation

如果主损失包含 weighted BCE、focal loss、稀疏正则和对称一致性项，建议增加一张损失消融表：

| 变体 | 损失组成 | 观察重点 |
|---|---|---|
| Loss-1 | BCE | 最基础分类能力 |
| Loss-2 | Weighted BCE | 类别不平衡修正 |
| Loss-3 | Weighted BCE + focal | 困难样本学习 |
| Loss-4 | 上述 + sparse reg | 解释热图稀疏性 |
| Loss-5 | 上述 + symmetry / consistency | 双向一致性与 hotspot 稳定性 |

## 评估指标建议

### Pair-level 指标

主表建议至少包含：

- AUPRC：类别不平衡下最重要。
- AUROC：便于和文献对齐。
- F1：便于表述阈值化性能。
- MCC：在正负样本不平衡时比 accuracy 更稳健。
- Precision / Recall：便于分析保守性与覆盖率。

B-PPI 在其摘要中也重点报告了 AUPRC 与 F1，这说明在严重不平衡的细菌 PPI 场景中，这两个指标具有很强的可解释性。

### 解释性指标

如果有 residue-level 标注、binding site 或结构参考，可增加：

- Top-k residue pair precision / recall
- Hotspot overlap
- Residue attention AUPRC
- Attention entropy
- Symmetry consistency score
- Case study 中与已知界面残基的重合率

若暂时没有大规模 residue-level gold label，也可以先做以下两类分析：

- 定量：attention entropy、top-k 对不同 seed 的稳定性、不同同源簇间的一致性。
- 定性：选若干案例，把 attention map 与已知结构或对接结果做可视化对照。

## 统计与复现实验规范

建议所有主结果至少运行 3 个随机种子；如果算力允许，优先 5 个种子，并报告 mean ± std。

训练与汇报时建议固定：

- 数据版本号
- split 版本号
- embedding 版本号
- 负样本版本号
- 代码 commit id
- 训练随机种子
- 推理阈值来源（固定阈值或 validation-optimal threshold）

如果资源允许，主表给出均值与标准差，附录补充每个 seed 的原始结果。这样做可以显著减少“单次跑高了”的偶然性争议。

## 推荐论文主表结构

### 表 1：主 benchmark

| Model | Random | Protein-disjoint | Cluster-disjoint | Species-disjoint |
|---|---|---|---|---|
| Baseline 1 | AUPRC / MCC | AUPRC / MCC | AUPRC / MCC | AUPRC / MCC |
| Baseline 2 | AUPRC / MCC | AUPRC / MCC | AUPRC / MCC | AUPRC / MCC |
| B-PPI style backbone | AUPRC / MCC | AUPRC / MCC | AUPRC / MCC | AUPRC / MCC |
| PPI-Sifter | AUPRC / MCC | AUPRC / MCC | AUPRC / MCC | AUPRC / MCC |

### 表 2：输入与主干 ablation

| Variant | Protein-disjoint | Cluster-disjoint | 结论 |
|---|---|---|---|
| ProstT5 + Bi-XAttn |  |  |  |
| ESM-C + No XAttn |  |  |  |
| ESM-C + Uni-XAttn |  |  |  |
| ESM-C + Bi-XAttn |  |  |  |
| ESM-C + Bi-XAttn + Gated FFN |  |  |  |

### 表 3：解释性分支 ablation

| Variant | Pair AUPRC | Pair MCC | Attention entropy | Top-k overlap |
|---|---|---|---|---|
| No explanation branch |  |  |  |  |
| Attention map only |  |  |  |  |
| Top-k only |  |  |  |  |
| Full PPI-Sifter |  |  |  |  |

## 推荐工程目录与 vibe coding 拆解

建议把实验代码分成“数据、切分、训练、评估、审计”五层，而不是把逻辑写进一个超长 notebook。

```text
ppi_sifter/
├── configs/
│   ├── data/
│   ├── model/
│   ├── train/
│   └── experiment/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── splits/
├── scripts/
│   ├── build_pairs.py
│   ├── cluster_proteins.py
│   ├── make_splits.py
│   ├── audit_leakage.py
│   ├── train.py
│   ├── evaluate.py
│   └── summarize_results.py
├── src/
│   ├── datasets/
│   ├── models/
│   ├── losses/
│   ├── metrics/
│   └── utils/
└── outputs/
```

### 推荐脚本职责

- `build_pairs.py`：从原始数据库构建正负样本对。
- `cluster_proteins.py`：运行 MMseqs2 / CD-HIT，生成 sequence cluster。
- `make_splits.py`：生成 random、protein-disjoint、cluster-disjoint、species-disjoint 各类 split。
- `audit_leakage.py`：检查 accession overlap、sequence overlap、cluster overlap、species overlap。
- `train.py`：读取指定 split 和模型配置并训练。
- `evaluate.py`：输出 AUROC、AUPRC、MCC、F1、PR 曲线与 attention 相关统计。
- `summarize_results.py`：将多 seed 的结果汇总成论文表格。

### 配置驱动建议

推荐使用 YAML 或 JSON 配置驱动实验，而不是在代码里硬编码超参数。最少应支持以下配置字段：

```yaml
experiment_name: ppi_sifter_cluster_disjoint
split_name: cluster_disjoint_v1
embedding_type: esm_c_frozen
negative_sampling: hard_negatives_v2
model_variant: bi_xattn_gated_full
loss_variant: wbce_focal_sparse_sym
seed: 42
```

这样做的好处是：

- 更容易批量扫 ablation。
- 更容易复现过去结果。
- 更容易把结果自动汇总成论文表格。

## 推荐执行顺序

### 阶段 1：打通最小可运行链路

目标：先证明数据、模型、评估全流程能跑通。

建议：

1. 用小样本 random split 跑通训练。
2. 确认 loss 能下降、metrics 能计算、attention 输出维度正确。
3. 确认日志、checkpoint、预测文件、解释输出都能保存。

### 阶段 2：固定数据协议

目标：生成论文级 split。

建议：

1. 完成蛋白去重。
2. 生成 protein-disjoint / cluster-disjoint / species-disjoint split。
3. 对每个 split 输出统计文件与 leakage report。
4. 锁定 split 文件，不再随意更改。

### 阶段 3：主模型与强 baseline

目标：先形成主表骨架。

建议：

1. 先跑最简单 baseline。
2. 再跑 B-PPI-style backbone。
3. 再跑 PPI-Sifter full model。
4. 主表先只保留 1 个 seed，待验证方向正确后再补 3–5 seeds。

### 阶段 4：系统 ablation

目标：回答审稿人最可能问的因果问题。

建议顺序：

1. 先做输入表征 ablation。
2. 再做交互主干 ablation。
3. 再做解释性分支 ablation。
4. 最后做损失函数与正则 ablation。

### 阶段 5：案例分析与写作

目标：从“跑分”升级为“可发表故事”。

建议：

1. 挑选真阳性、假阳性、假阴性案例。
2. 绘制 attention map 与 top-k residue pairs。
3. 对照已知结构、对接结果或功能位点做解释。
4. 在论文中分别给出主结果、泛化结果、解释性案例。

## 建议在论文中提前回答的审稿问题

建议在实验设置部分主动回答以下问题：

- 为什么不能只用 random split。
- 为什么 baseline 需要重训而不是直接引用分数。
- 如何确保训练集与测试集没有蛋白或簇级重叠。
- 负样本是否引入偏差，如何控制。
- PPI-Sifter 的提升来自 ESM-C 还是来自 cross-attention。
- 解释性分支是否牺牲了主任务性能。
- 在更严格 split 下模型是否仍保持优势。

这些问题之所以重要，是因为近年的 benchmark 研究已经反复强调，PPI 或相互作用相关任务中的高分往往会受到数据划分方式的显著影响。

## 最终建议

PPI-Sifter 的实验设计应坚持一个主线：在统一、严格、可审计的数据划分协议下，证明模型在 pair-level 预测、跨域泛化和 residue-level 解释三方面都优于合理 baseline。

从工程角度，最优先完成的不是“把所有 fancy 模块都加上”，而是先把 split、leakage audit、训练配置和结果汇总这四个基础设施搭好。只要这四件事做扎实，后续 benchmark 扩展、ablation 扫描和论文写作都会顺很多。
