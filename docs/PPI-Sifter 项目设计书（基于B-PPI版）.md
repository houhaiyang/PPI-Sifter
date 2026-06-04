# PPI-Sifter 项目设计书（基于B-PPI版）

- **项目名称：** PPI-Sifter
- **项目定位：** 面向高通量蛋白互作筛选与残基热点解释的一体化序列模型
- **核心主线：** 冻结 ESM-C + B-PPI 风格双向 cross-attention 主干 + residue-level attention map + top-k residue pairs

***

## 一、项目概述

PPI-Sifter 旨在构建一个面向高通量蛋白互作筛选的可解释序列模型，在同一个模型框架内同时完成蛋白对是否互作的判别、残基级注意力图导出以及 top-k 残基对热点输出。

本版本对原有设计进行了彻底更正：删除 protein-level fast filter，不再采用“先粗筛、再细排”的双阶段前端；模型前半段严格对齐 B-PPI 的主干设计，即输入投影、双向 cross-attention、gated FFN、attention pooling、对称共享表示和 MLP 分类头，仅在输出端扩展 residue-level 可解释性结果。

因此，PPI-Sifter 的创新点不再表述为“更换 B-PPI 的前端判别框架”，而是表述为“在保持 B-PPI 前半段判别机制一致的前提下，将 cross-attention 中间表示显式转化为可导出的 residue-level attention map，并进一步输出 top-k residue pairs 作为候选界面热点”。

***

## 二、研究背景

蛋白质相互作用是解析细胞功能、信号通路、病原机制与蛋白功能注释的关键基础。实验手段虽然可靠，但覆盖率有限、成本较高，尤其在非模式物种、跨物种互作与蛋白组级别扫描任务中，难以满足高通量筛选需求。

结构预测方法如 docking、AlphaFold-Multimer 与 AlphaFold 3 在复合物层面具有很强能力，但其计算成本不适合用于海量候选蛋白对的第一轮筛查。因此，面向高通量的序列模型仍然是现实工作流中的核心过滤器。

近年来，基于蛋白语言模型的 PPI 方法已经证明，仅用序列就能够学习具有结构意义的表征。其中，B-PPI 进一步说明：使用预训练 residue embeddings，并通过双向 cross-attention 显式建模蛋白间残基依赖关系，可以在不均衡数据设定下取得很强的判别能力。这为 PPI-Sifter 提供了直接的方法学基础。

***

## 三、问题定义

本项目关注两个彼此耦合的问题。

第一，如何在高通量候选蛋白对集合上输出尽可能可靠的 pair-level interaction probability，从而把真正值得后续实验或结构验证的候选对优先筛选出来。

第二，如何在给出 pair-level 预测的同时，进一步指出模型认为最重要的 residue-residue interaction region，使模型输出不再停留于单一概率值，而是能够为后续点突变设计、界面验证与机制分析提供更直接的依据。

因此，PPI-Sifter 的目标不是另起炉灶地重新设计一套与 B-PPI 完全不同的判别主干，而是在 B-PPI 风格的判别框架上增加标准化的解释输出链路。

***

## 四、设计原则

### 4.1 前半段与 B-PPI 一致

PPI-Sifter 的前半段网络严格遵循 B-PPI 的总体思想：对两条蛋白的 residue embeddings 做线性投影，使用双向 multi-head cross-attention 建模 inter-protein residue dependency，之后使用 gated FFN 进一步精炼特征，再通过 attention pooling 将变长残基表示压缩为固定长度向量，最后通过对称表示和 MLP 完成 pair-level 分类。

### 4.2 删除 protein-level fast filter

本次更正后，项目中不再保留 protein-level fast filter。删除原因很明确：该模块并不属于 B-PPI 主干，若继续保留，会使“前面保持与 B-PPI 一致”的表述不成立，也会让模型设计变成另一套两阶段框架。

### 4.3 可解释性增强集中在输出端

PPI-Sifter 的主要新增内容是显式导出 residue-level attention map，并基于该图计算 top-k residue pairs。也就是说，PPI-Sifter 的创新应定位为“保留 B-PPI 风格判别主干，增强其解释输出能力”，而不是“用新的粗筛机制替代 B-PPI 前端”。

***

## 五、模型总体结构

### 5.1 输入与编码

输入为两条蛋白质序列。与 B-PPI 使用 ProstT5 residue embeddings 不同，PPI-Sifter 使用冻结 ESM-C 编码得到 residue-level embeddings。对于一条长度为 \(L\) 的蛋白质序列，其表示为一个形状为 \(L 	imes d_{in}\) 的矩阵。

ESM-C 在本项目中仅作为通用特征提取器，不参与端到端微调。这样做的目的有两点：第一，减少训练开销；第二，使训练过程集中在轻量交互头上，便于在大规模 PPI 数据上快速迭代。

### 5.2 输入投影层

由于原始 residue embeddings 维度较高，模型首先通过一个可学习线性层将输入投影到较低维的潜在交互空间。设蛋白 A 和蛋白 B 的原始 embedding 分别为 \(X_A \in \mathbb{R}^{L_A 	imes d_{in}}\) 和 \(X_B \in \mathbb{R}^{L_B 	imes d_{in}}\)，投影后得到：

\[
H_A = X_A W_p, \quad H_B = X_B W_p
\]

其中 \(W_p\) 为共享线性投影参数。

这一层的作用与 B-PPI 一致，即把高维预训练特征映射到适合交互建模的任务空间。

### 5.3 双向 cross-attention

为了显式建模 inter-protein residue dependency，模型使用双向 multi-head cross-attention。

第一方向中，蛋白 A 作为 query，蛋白 B 作为 key/value，得到 A 在 B 条件下更新后的表示；第二方向中，蛋白 B 作为 query，蛋白 A 作为 key/value，得到 B 在 A 条件下更新后的表示。两个方向都包含残差连接与 dropout，以保持优化稳定性并降低过拟合风险。

该步骤的本质是让每个残基根据其 partner protein 的上下文信息重新加权自身表示，从而把跨蛋白依赖关系直接编码到 residue-level hidden states 中。

### 5.4 Gated FFN

在 cross-attention 之后，模型为两条蛋白各自引入 gated feed-forward network。该模块先对特征做维度扩展，再通过门控机制抑制无关信息，最后投影回原维度，并与输入残差相加。

这一设计同样对齐 B-PPI：它不是简单使用普通 FFN，而是通过 gate 控制信息流，使模型能够更灵活地保留真正与互作相关的 residue-level 交互特征。

### 5.5 Attention pooling

由于 pair-level 分类需要固定长度表示，而残基序列长度是可变的，因此模型对更新后的 residue hidden states 做 attention pooling。

对于每条蛋白，模型学习一个 residue importance 分布，用其对残基表示做加权求和，从而得到全局向量 \(s_A\) 和 \(s_B\)。这一聚合方式优于简单 mean pooling，因为它允许模型把更多权重分配给可能更接近互作界面的残基区域。

### 5.6 对称共享表示

蛋白互作在物理上是无向关系，因此模型必须满足输入顺序对称性。参考 B-PPI，共享表示构造为 pooled vectors 的逐元素和与绝对差的拼接：

\[
V = [(s_A + s_B) \; || \; |s_A - s_B|]
\]

该表示天然对称，能够避免模型学习到无意义的输入顺序偏置。

### 5.7 分类头

最后，将共享表示 \(V\) 输入一个两层隐藏层的 MLP，输出 pair-level interaction logit，经 sigmoid 后得到互作概率。

这一部分应尽量保持轻量，以确保主要建模能力仍来自前面的 residue-level interaction backbone，而不是过度依赖末端全连接层去记忆数据分布。

***

## 六、PPI-Sifter 的解释性增强

### 6.1 Residue-level attention map

虽然前半段主干与 B-PPI 一致，但 PPI-Sifter 不满足于只输出 pair-level probability。项目的核心增强在于：将双向 cross-attention 过程中形成的 residue-residue 相关性显式整理为 attention map，并作为标准结果输出。

该 attention map 的行表示蛋白 A 的残基，列表示蛋白 B 的残基，矩阵中的元素代表模型认为某一对残基在当前互作判断中的相对贡献程度。为了提升解释稳定性，可以对两个方向的 cross-attention 权重或派生 score map 做对称化融合。

### 6.2 Top-k residue pairs

在得到 residue-level attention map 后，模型进一步从中提取 top-k scoring residue pairs。输出形式为按分数排序的残基对列表，例如 \((i_1,j_1), (i_2,j_2), \dots, (i_k,j_k)\)。

这些 top-k residue pairs 被视为候选互作界面热点，可用于后续的点突变实验设计、结构对接优先位点分析或与已知界面标注进行对照验证。

### 6.3 与主任务的关系

需要强调的是，attention map 与 top-k residue pairs 不是独立于主任务之外的后处理装饰，而是来自主干内部真实使用的中间表示。因此，PPI-Sifter 的解释输出与 pair-level prediction 是耦合的，而不是割裂的。

***

## 七、损失函数设计

### 7.1 主分类损失

由于真实 PPI 任务往往存在明显类别不平衡，因此主分类损失建议采用 weighted BCE 与 focal loss 的组合。其中，weighted BCE 用于提升正样本权重，focal loss 用于聚焦难分类样本。

总的分类项可写为：

\[
L_{cls} = L_{wbce} + \lambda_1 L_{focal}
\]

### 7.2 Attention 正则

为了让 attention map 更适合作为界面解释，建议再加入两个轻量正则项。

第一是稀疏正则，用于抑制整张图均匀扩散，使模型更倾向于形成局部高响应区域。第二是对称一致性正则，用于约束 A→B 与 B→A 导出的得分图在转置意义上保持一致。

因此，总损失可写为：

\[
L = L_{wbce} + \lambda_1 L_{focal} + \lambda_2 L_{sparse} + \lambda_3 L_{sym}
\]

若未来引入真实界面标注，还可以继续加入 hotspot consistency loss，但该项不是当前版本的必要组成。

***

## 八、数据流程与工程实现

### 8.1 数据输入

训练与推理都以蛋白对表格作为核心输入。CSV 至少包含 `protein_a`、`protein_b` 和 `label` 三列，其中推理时 `label` 可以省略。

### 8.2 Embedding 缓存

为避免重复编码，所有蛋白的 ESM-C residue embeddings 建议离线提取并合并为单个 `.pt` 仓库。每个蛋白对应一个形状为 `L × D` 的张量，并附带长度与维度元数据。

### 8.3 训练流程

训练阶段从 embedding 仓库中读取两条蛋白的 residue embeddings，经过 padding 与 mask 组装 batch，送入主模型前向传播，计算 pair-level 概率和 attention map，再根据标签计算损失并更新参数。

### 8.4 推理与解释导出

推理阶段除输出 pair-level interaction probability 外，还支持把 residue-level attention map 导出为矩阵、热图和 top-k residue pairs CSV。这样模型就能够同时服务“高通量筛选”和“候选界面解释”两类需求。

***

## 九、评测方案

### 9.1 Pair-level 指标

主要使用 AUPRC、AUROC、F1、MCC、Precision 与 Recall 评估 pair-level 判别性能。对于高通量场景，AUPRC 特别重要，因为它比 AUROC 更能反映正负极度不平衡下的筛选质量。

### 9.2 Interpretability 指标

解释性部分建议至少报告以下指标：top-k residue pairs 与已知界面的 overlap、attention entropy、对称一致性误差以及不同随机种子的稳定性。

### 9.3 划分策略

为避免信息泄漏，数据划分不应只做随机切分，而应尽量采用 protein-disjoint、cluster-disjoint 或 species-disjoint 的更严格协议。该点与 B-PPI 使用 cluster-aware redundancy control 的原则是一致的。

***

## 十、与 B-PPI 的关系定位

PPI-Sifter 不是要否定 B-PPI，而是站在 B-PPI 的主干之上，补足其在 residue-level 可解释输出方面的不足。

更准确地说，B-PPI 证明了 bidirectional cross-attention 作为 PPI 判别主干是有效的；PPI-Sifter 则进一步把这一主干中隐含的 residue interaction information 显式导出，使输出从“一个概率值”扩展为“概率值 + attention map + top-k residue pairs”。

因此，PPI-Sifter 的方法学定位应写成：**B-PPI-style interaction backbone with explicit residue-level interpretability outputs**。

***

## 十一、代码实现原则

### 11.1 删除 fast filter 相关代码

模型代码中不再保留任何 gate、threshold 或 coarse-to-fine 分支；也不再存在单独的 fast filter 损失项。

### 11.2 主模型结构

`model.py` 应只保留以下顺序：

1. 输入投影。
2. 双向 cross-attention。
3. gated FFN。
4. attention pooling。
5. 对称共享表示。
6. MLP 分类头。
7. attention map 与 top-k residue pairs 导出接口。

### 11.3 可解释性模块

`interpret.py` 负责从模型输出中抽取 residue-level attention map、生成热图并导出 top-k residue pairs；该模块不应反向改变主干结构，而只负责标准化解释结果的组织与保存。

***

## 十二、预期创新点

- **与 B-PPI 一致的前半段主干。** 保证模型设计叙述清晰、方法来源明确。
- **冻结 ESM-C 的替代表征。** 在保持 B-PPI 架构思想的同时，用 ESM-C 作为 residue-level encoder。
- **显式可解释输出。** 将 residue-level attention map 与 top-k residue pairs 变为标准输出，而非附属图示。
- **更适合实验闭环。** 输出结果能够直接支持界面残基假设生成与后续突变验证。

***

## 十三、预期成果

项目完成后，预期形成以下成果：

1. 一份逻辑自洽、与 B-PPI 关系表述准确的项目设计书。
2. 一套删除 fast filter 后的可运行代码框架。
3. 一套支持训练、推理、解释导出与可视化的脚本。
4. 一套适用于后续 benchmark 与消融的统一配置系统。

最终，PPI-Sifter 将作为一个“B-PPI 风格判别主干 + residue-level hotspot 解释输出”的高通量 PPI 序列模型，为蛋白互作筛选与界面假设生成提供统一工具链。
