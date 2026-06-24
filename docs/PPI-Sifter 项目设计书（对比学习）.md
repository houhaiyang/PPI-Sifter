# PPI-Sifter 项目设计书（对比学习）

- **项目名称：** PPI-Sifter
- **项目定位：** 面向高通量蛋白互作筛选与机制分析的轻量可解释序列模型
- **核心主线：** 冻结 ESM-C + B-PPI 风格双向 cross-attention 主干 + 对比学习式机制解释 + layer-wise partner-aware pattern analysis

***

> 整体定位会从“attention map 对齐”转成“**通过对比学习与层级分析解释模型为什么能区分正负样本，以及哪一层 cross-attention 学到了 partner-aware pattern**”。这一转向仍然保留 B-PPI 风格主干与轻量工程路线，且更适合你当前资源约束与 BIBM 2026 的投稿目标。

***

## 一、项目概述

PPI-Sifter 旨在构建一个面向高通量蛋白互作筛选的轻量序列模型，在保持 pair-level 预测能力的同时，不再把解释性仅仅限定为 attention map 热图导出，而是进一步回答一个更核心的问题：**模型究竟为什么能够区分正负样本，以及这种区分能力是在什么层级、通过什么 partner-aware interaction pattern 学出来的。**

与原有“基于 attention map 的显式热点解释”版本相比，本版本将研究重点从“输出哪个残基对最重要”转向“分析 cross-attention backbone 是否学到了可分离、可稳定、可归纳的正负样本交互模式”。因此，本项目的主要创新点不再是引入额外的 residue-level 监督，而是将双向 cross-attention 的中间表示系统化地组织为**可对比、可量化、可层级分析**的解释信号，并通过对比学习目标强化这种可分性。

本版本仍然保持与 B-PPI 一致的前半段判别主干，即输入投影、双向 cross-attention、gated FFN、attention pooling、对称共享表示和 MLP 分类头，仅在训练目标与解释分析链路上增加 contrastive interpretability 分支，从而更低成本地回答“模型内部学到了什么”。

***

## 二、研究背景

蛋白质相互作用预测的核心任务，表面上是判断两条蛋白是否结合，但真正困难的问题往往是：模型到底依据了什么证据做出判断。仅输出一个 pair-level probability 虽然适合高通量筛选，但难以支撑后续机制分析、错误诊断与论文中的可解释性论证。

现有基于序列的 PPI 模型通常把注意力热图直接当作解释结果，但当训练标签只有 pair-level 0/1 时，attention map 很容易退化为“可视化附属品”，并不一定真正对应稳定的交互模式。已有项目设计与注意力正则分析也说明，在缺少 residue-level 真值时，attention 的解释价值主要依赖间接约束，而其可靠性仍需额外验证。

因此，本项目提出一个更适合当前资源条件的解释路线：不强求在现阶段准确恢复真实界面，而是先通过**对比学习 + 层级表征分析**回答两个更基础但更重要的问题：

1. 正样本与负样本在 cross-attention 表征空间中是否形成可分离结构；
2. 这种可分离性主要出现在第几层、哪一方向、哪一种交互模式中。
这一路线更适合轻量工程实现，也更容易形成 BIBM 风格的机制分析论文。

***

## 三、问题定义

本项目聚焦三个相互关联的问题。

第一，如何在高通量候选蛋白对集合上输出可靠的 pair-level interaction probability，以保证模型在未见蛋白或远同源蛋白上仍具备较好的筛选能力。

第二，如何不依赖昂贵的 residue-pair 真值标注，而是直接利用正负样本间的表征差异，分析 cross-attention 模块是否学到了 partner-aware interaction pattern。这里所谓 partner-aware，指的是同一条蛋白在面对不同 partner 时，其中间表征和注意力分布会发生系统性变化，而这种变化对正样本和负样本具有可区分性。

第三，如何把这种“机制解释”从主观可视化提升为可量化实验对象，即通过 contrastive objectives、layer-wise separability、attention entropy、symmetry consistency、cross-partner perturbation 等指标，系统回答模型的判别依据是否稳定、是否层级可定位、是否具备一定生物学合理性。

因此，本版本 PPI-Sifter 的目标不再是优先恢复精确的界面热点，而是构建一个**可分析、可对比、可复现**的 PPI 表征解释框架。

***

## 四、设计原则

### 4.1 前半段保持与 B-PPI 一致

PPI-Sifter 的判别主干仍严格遵循 B-PPI 风格：对两条蛋白的 residue embeddings 做共享投影，使用双向 multi-head cross-attention 显式建模 inter-protein dependency，之后使用 gated FFN 精炼特征，再通过 attention pooling 和对称共享表示完成 pair-level 分类。

### 4.2 解释目标从“热点恢复”转向“机制分离”

本版本不把解释性优先定义为“预测出真实界面残基对”，而是定义为：**模型内部是否形成了可区分的正负样本交互模式，以及这些模式是否能通过层级分析和对比学习被稳定捕获。** 也就是说，本项目更关心“为什么能分”，而不是“具体哪一对残基一定是界面”。

### 4.3 轻量优先

考虑到当前实验目标是短时间内完成 BIBM 2026 可投稿版本，项目避免引入大规模结构标注、复杂多任务 supervision 和高成本复合体建模。解释增强优先通过现有 pair-level 标签与中间表征分析完成，而不是额外依赖大规模 residue-level gold label。

### 4.4 可复现与工程分层

遵循模型设计规范，模型层、数据层、训练层、推理层、评估层严格解耦；超参集中在 YAML 配置；训练、评估、可视化与案例分析均提供独立脚本，避免把解释逻辑混入模型前向代码。

***

## 五、模型总体结构

### 5.1 输入与编码

输入为两条蛋白质序列。与 B-PPI 使用 ProstT5 residue embeddings 不同，PPI-Sifter 使用冻结 ESM-C 编码得到 residue-level embeddings。对于长度分别为 $L_A$ 和 $L_B$ 的两条蛋白，其输入表示分别为：

$$
X_A \in \mathbb{R}^{L_A \times d_{in}}, \quad
X_B \in \mathbb{R}^{L_B \times d_{in}}
$$

其中 $d_{in}$ 为 ESM-C 的 residue embedding 维度。ESM-C 在本项目中仅作为离线特征提取器使用，不参与端到端微调，以降低训练资源消耗并提升实验迭代效率。

### 5.2 输入投影层

为了把高维预训练特征映射到适合交互建模的潜在空间，模型使用共享线性投影：

$$
H_A^{(0)} = X_A W_p, \quad
H_B^{(0)} = X_B W_p
$$

其中 $W_p \in \mathbb{R}^{d_{in} \times d}$，$d$ 为任务空间维度。投影层的作用与 B-PPI 一致，即在不改变 residue-level granularity 的前提下压缩输入维度并建立统一交互空间。

### 5.3 双向 cross-attention 编码器

模型使用 $N$ 层双向 cross-attention block。对第 $l$ 层而言，A→B 方向以 $H_A^{(l-1)}$ 为 query、$H_B^{(l-1)}$ 为 key/value；B→A 方向反之。得到：

$$
\tilde{H}_A^{(l)} = \mathrm{CrossAttn}(H_A^{(l-1)}, H_B^{(l-1)}, H_B^{(l-1)})
$$

$$
\tilde{H}_B^{(l)} = \mathrm{CrossAttn}(H_B^{(l-1)}, H_A^{(l-1)}, H_A^{(l-1)})
$$

之后各自经过残差、LayerNorm 与 gated FFN，输出 $H_A^{(l)}$ 与 $H_B^{(l)}$。每一层同时保留 attention weights 与中间 hidden states，用于后续 layer-wise interpretability analysis。

### 5.4 Gated FFN

为了对齐 B-PPI 风格并提升非线性建模能力，每层 cross-attention 后接 gated FFN，对 partner-conditioned residue 表示做进一步筛选与重构。这一模块的作用不是增加额外复杂度，而是让交互后特征能够保留更有判别力的 partner-aware 成分。

### 5.5 Attention pooling 与对称共享表示

经过最后一层编码后，两条蛋白分别做 attention pooling，得到固定长度全局表示 $s_A$ 和 $s_B$。随后构造对称共享表示：

$$
V = [(s_A + s_B) \; || \; |s_A - s_B|]
$$

该表示满足输入顺序对称性，并作为 pair-level 分类头的输入。

### 5.6 分类头

共享表示 $V$ 输入轻量 MLP，输出 pair-level interaction logit，经 sigmoid 后得到互作概率。分类头保持轻量，以确保主建模能力仍主要来自 cross-attention interaction backbone。

***

## 六、对比学习式解释增强

### 6.1 核心思想

本版本的解释增强不依赖额外界面标签，而是把正负样本的中间交互表征当作一个可对比的表示学习问题。直观地说，如果模型真正学到了 partner-aware interaction pattern，那么：

- 正样本应在某些层的中间表示空间中形成更紧密的聚类；
- 负样本应与正样本在这些层的表征上保持可分离；
- 同一 anchor 蛋白面对真实 partner 与假 partner 时，其 cross-attention 派生表示应出现可识别的方向性偏移。

因此，本项目把解释性增强定义为：**提升并分析正负样本在 cross-attention 中间空间里的可分性，而不是只看最终分类分数。**

### 6.2 可解释表征的定义

对每一层 $l$，从模型中提取以下中间表征作为解释对象：

1. `pooled_a_l`：第 $l$ 层 A 的 pooled representation
2. `pooled_b_l`：第 $l$ 层 B 的 pooled representation
3. `pair_repr_l`：由两者构造的对称 pair representation
4. `attn_map_l`：该层双向 cross-attention 融合后的 attention map
5. `attn_stats_l`：attention entropy、head variance、symmetry error 等统计量

其中最关键的是 `pair_repr_l`，因为它是该层 partner-aware interaction information 的压缩表示，可直接用于对比学习与可分性分析。

### 6.3 正负样本对比目标

训练时，在主分类损失之外，对某一层或多层的 `pair_repr_l` 加入监督式对比损失。正样本之间、同一蛋白与真实 partner 的不同增强视图之间视为 positive pairs；正样本与负样本、或真实 partner 与 hard negative partner 之间视为 negative pairs。其目的不是替代分类任务，而是强化中间层的结构化分离。

可选的对比对象包括：

- **Batch 内 supervised contrastive**：同标签样本聚集，异标签分离。
- **Anchor-partner 对比**：固定 protein A，真实 partner B$^+$ 与 hard negative B$^-$ 做对比。
- **Layer-wise contrastive**：分别对第 1 层、第 2 层、最后一层做对比，比较哪一层最具分离力。


### 6.4 Partner-aware pattern 的 operational definition

为了避免解释概念过于抽象，本项目把 partner-aware pattern 操作化为以下可量化现象：

- 同一条蛋白 A 与不同 partner 配对时，其 cross-attended residue states $H_A^{(l)}$ 会发生显著可测的偏移；
- 对真实 partner，这种偏移更有组织性，表现为更低熵、更高 head 一致性、更稳定的层间聚类结构；
- 对负样本或 hard negative partner，偏移更随机、更分散、层间稳定性更差。

如果这些现象在实验中成立，则说明 cross-attention 学到的不只是“全局相似性”，而是确实在编码 partner-conditioned interaction pattern。

***

## 七、损失函数设计

### 7.1 主分类损失

由于 PPI 数据集往往严重类别不平衡，主分类损失仍采用 weighted BCE 与 focal loss 的组合：

$$
L_{cls} = L_{wbce} + \lambda_1 L_{focal}
$$

该部分只负责最终 pair-level interaction prediction，与原始设计保持一致。

### 7.2 对比学习损失

对某一层或多层的 `pair_repr_l` 使用 supervised contrastive loss：

$$
L_{con} = \sum_{l \in \mathcal{L}} \alpha_l \, L_{supcon}(z_l, y)
$$

其中 $z_l$ 为第 $l$ 层 pair representation 经过投影头后的低维向量，$y$ 为 pair-level 标签，$\alpha_l$ 为层权重。若使用 anchor-partner 方案，也可替换为 triplet loss 或 InfoNCE。[待实验验证]

### 7.3 轻量 attention 正则

虽然本版本不再把 attention map 作为主监督对象，但仍保留轻量 attention regularization，以保证中间交互模式更稳定、更可分析：

$$
L_{reg} = \lambda_2 L_{sparse} + \lambda_3 L_{sym} + \lambda_4 L_{entropy}
$$

其中：

- $L_{sparse}$：抑制全图均匀扩散；
- $L_{sym}$：约束 A→B 与 B→A 融合图的一致性；
- $L_{entropy}$：鼓励正样本注意力更聚焦、负样本更平坦。
这些设计与原 attention 正则思路兼容，但其角色从“界面恢复”转为“稳定可分析的机制表征”。


### 7.4 总损失

最终总损失写为：

$$
L = L_{cls} + \beta_1 L_{con} + \beta_2 L_{reg}
$$

其中 $\beta_1$ 控制对比学习对中间空间结构化的影响，$\beta_2$ 控制 attention 统计稳定性。训练初期建议先用较小 $\beta_1$ 以免破坏主任务收敛。

***

## 八、解释分析框架

### 8.1 Layer-wise separability analysis

为了回答“哪一层学到了 partner-aware pattern”，项目对每一层的 `pair_repr_l` 分别进行离线分析。核心做法包括：

- 训练后抽取各层表示；
- 用线性 probe 或 kNN 测试其对正负样本的可分性；
- 计算 silhouette score、intra-/inter-class distance；
- 对各层做 t-SNE / UMAP 可视化。

若某一中间层已表现出比最后分类头更明显的正负分离，则说明该层本身已经编码了关键交互模式。

### 8.2 Anchor-partner shift analysis

固定同一个 anchor protein A，分别与真实 partner、随机 negative、hard negative 配对，比较其各层表示 $H_A^{(l)}$ 的偏移量：

$$
\Delta_l(A; B_1, B_2) = \| \mathrm{Pool}(H_A^{(l)}(A,B_1)) - \mathrm{Pool}(H_A^{(l)}(A,B_2)) \|_2
$$

若真实 partner 导致的偏移与负 partner 显著不同，说明模型确实在以 partner-conditioned 方式编码信息，而不是只看单蛋白本身。

### 8.3 Attention statistics profiling

对每一层的 attention map 计算以下统计量：

- row / column entropy
- head-wise variance
- symmetry consistency
- top-k stability across seeds
- positive-negative distribution gap

这些指标并不要求 attention 对应真实界面，但能回答注意力模式是否具有结构化差异与统计稳定性。

### 8.4 Cross-partner perturbation

在推理阶段，对固定蛋白 A 替换不同 partner，或对 partner 做局部 masking / shuffle，观察中间层表示与最终分数如何变化。如果早层几乎不变、深层显著变化，则说明 partner-aware interaction pattern 是逐层建立的，而不是输入即固定存在。

***

## 九、数据流程与工程实现

### 9.1 数据输入

训练与推理均以蛋白对表格为核心输入，至少包含：

- `protein_a`
- `protein_b`
- `label`

可选字段包括 species、taxa、cluster_id、negative_type、evidence_score 等，用于后续 split、负样本构造与案例分析。

### 9.2 Embedding 缓存

所有蛋白的 ESM-C residue embeddings 建议离线提取并存为支持随机访问的 HDF5 仓库，以避免 NPZ 全量解压导致的内存问题。每个蛋白保存一个 `L × D` 张量及长度元数据，训练时按需读取。

### 9.3 中间表示缓存

为支持层级解释分析，模型在验证与测试阶段额外导出以下内容：

- 每层 `pair_repr_l`
- 每层 pooled residue representation
- 每层 attention statistics
- 最终预测分数与标签

这些结果建议保存为分层命名的 `.pt` 或 `.h5` 文件，供 `evaluate/` 和 `scripts/analysis/` 独立读取。

### 9.4 推理与解释导出

推理阶段除输出 pair-level interaction probability 外，还支持导出：

- layer-wise pair embeddings
- attention entropy 曲线
- partner shift 分析结果
- hard negative 对照案例
- top-k attention pairs（可选）

解释模块应作为独立分析工具，而不应改变训练主干结构。

***

## 十、评测方案

### 10.1 Pair-level 指标

主任务仍使用以下标准指标：

- AUPRC
- AUROC
- F1
- MCC
- Precision / Recall

其中 AUPRC 仍是类别不平衡设定下最重要的主指标。

### 10.2 对比学习解释指标

本版本的解释性不再只依赖 top-k overlap，而是引入以下机制分析指标：

- **Layer separability**：各层线性 probe AUROC / AUPRC
- **Contrastive clustering quality**：silhouette score、Davies-Bouldin index
- **Partner shift gap**：真实 partner 与负 partner 导致的表示偏移差
- **Attention entropy gap**：正负样本在各层 attention entropy 的差异
- **Symmetry consistency**：双向 attention map 的一致性
- **Seed stability**：不同随机种子下层级结论是否一致

这些指标共同回答：模型的可解释性是否来自稳定的内部模式，而不是偶然的热图形状。

### 10.3 划分策略

为避免数据泄漏，主实验优先采用：

- Protein-disjoint
- Cluster-disjoint

Random split 只用于 smoke test 或附录。若后续数据规模足够，可补充 species-disjoint 作为更严格泛化测试。

***

## 十一、Benchmark 与消融设计

### 11.1 主比较模型

主表建议保留以下模型：

- Mean pooling + MLP
- Siamese / dual encoder + shallow head
- B-PPI-style backbone
- PPI-Sifter（无对比分支）
- PPI-Sifter（对比分支 full）

这样可以清晰回答：性能提升来自 ESM-C、来自 cross-attention，还是来自 contrastive interpretability branch。

### 11.2 关键消融

建议重点做四组消融：

1. **层选择消融**
    - 第 1 层 contrastive
    - 第 2 层 contrastive
    - 最后一层 contrastive
    - 多层联合 contrastive
2. **对比对象消融**
    - batch supervised contrastive
    - anchor-partner triplet
    - hard negative only
    - no contrastive
3. **解释统计消融**
    - no entropy regularization
    - no symmetry regularization
    - no attention statistics export
4. **交互主干消融**
    - no cross-attention
    - uni-directional attention
    - bi-directional attention
    - remove gated FFN

这些消融直接对应审稿人最可能问的因果问题：

- 可解释性增强是否真的帮助主任务？
- 哪一层最关键？
- partner-aware pattern 是否真的来自 cross-attention，而不是输入 embedding 本身？

***

## 十二、与 B-PPI 的关系定位

PPI-Sifter 仍然站在 B-PPI 的主干之上，而不是否定 B-PPI。B-PPI 证明了双向 cross-attention 作为 PPI 判别主干是有效的；PPI-Sifter 在此基础上进一步追问：**这种主干内部究竟学到了什么样的正负样本交互模式，以及这些模式能否以对比学习和层级分析的方式被显式揭示。**

因此，本版本方法学定位应写成：

**B-PPI-style interaction backbone with contrastive interpretability over layer-wise partner-aware representations.**

***

## 十三、代码实现原则

### 13.1 主干结构不随意改动

`model.py` 应仍保留以下核心顺序：

1. 输入投影
2. 多层双向 cross-attention
3. gated FFN
4. attention pooling
5. 对称共享表示
6. MLP 分类头
7. 中间层表示导出接口

避免为了分析方便而破坏主干结构的一致性。

### 13.2 对比分支模块化

新增模块建议独立为：

- `contrast_head.py`：层级投影头与 contrastive embeddings
- `contrast_loss.py`：supervised contrastive / triplet / InfoNCE
- `analysis.py`：层可分性、partner shift、entropy gap、seed stability
- `export_repr.py`：中间表示导出与缓存

这样可以保证模型训练、解释分析和论文绘图相互独立，符合工程规范。

### 13.3 配置驱动

所有实验超参使用 YAML 管理，至少包含：

- `embedding_type`
- `model_variant`
- `contrast_layers`
- `contrast_type`
- `contrast_weight`
- `negative_sampling`
- `split_name`
- `seed`

这有助于快速扫消融并保证可复现。

***

## 十四、预期创新点

- **保持 B-PPI 风格判别主干不变。** 方法来源清晰，结构叙事稳定。
- **将解释问题从热点恢复转向机制分析。** 不再局限于“画 attention map”，而是分析模型为何能区分正负样本。
- **提出 layer-wise partner-aware pattern analysis。** 直接回答哪一层 cross-attention 学到了更强的判别交互模式。
- **用对比学习强化中间表示结构。** 在不依赖昂贵 residue-level 金标准的前提下，让解释性进入可训练、可量化、可复现的范式。
- **更适合低资源快速迭代。** 相比大规模界面对齐或结构 supervision，本路线更容易在当前资源条件下完成并形成 BIBM 论文闭环。

***

## 十五、预期成果

项目完成后，预期形成以下成果：

1. 一份从“attention 热图解释”转向“对比学习机制解释”的新项目设计书。
2. 一套可运行的 B-PPI 风格轻量 PPI 主干代码。
3. 一套支持 layer-wise representation export、contrastive training 和 mechanism analysis 的实验脚本。
4. 一套围绕 protein-disjoint / cluster-disjoint 的 benchmark 与解释性消融体系。
5. 一篇可面向 BIBM 2026 投稿的论文主线：**轻量序列 PPI 预测 + 对比学习式内部机制解释**。

最终，PPI-Sifter 将不再只是一个“输出概率 + 热图”的模型，而是一个能够系统回答“模型为什么这样预测”的轻量级 PPI 机制分析框架。

***
