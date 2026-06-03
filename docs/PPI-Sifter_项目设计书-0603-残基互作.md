# PPI-Sifter 项目设计书（正式提交版）

- **项目名称：** PPI-Sifter
- **项目定位：** 面向高通量蛋白互作筛选与残基热点解释的一体化序列模型
- **核心主线：** 冻结 ESM-C + protein-level fast filter + residue-level attentive reranker

***

## 一、项目概述

- PPI-Sifter 旨在构建一个可用于大规模 PPI 筛选的高通量预测系统，在模型框架内同时完成蛋白对互作判别与残基级互作热点定位，并输出可解释的 attention map 。
- 项目强调“快筛、精判、可解释”三位一体，重点服务于蛋白组级候选筛查、跨物种泛化预测和后续实验优先级排序 。

***

## 二、研究背景

- PPI 实验数据库覆盖仍不完整，尤其在非模式物种、跨物种互作以及低丰度、弱互作场景中，真实互作数据稀缺且噪声较高 。
- 传统结构法如 docking 和 AlphaFold 类方法虽然精度高，但在全蛋白组高通量筛查中计算成本过高，难以承担百万级、千万级 pair 枚举任务 。
- 近年的序列模型如 D-SCRIPT、SENSE-PPI、Topsy-Turvy、B-PPI 证明了基于 PLM 的方法具备 genome-scale PPI 推断潜力，但在高通量、残基解释和误报控制之间仍存在明显权衡 。

***

## 三、研究现状

### . 传统序列方法

- 早期方法如 PIPR 等主要依赖手工或浅层序列表征，在同种内验证中可取得一定性能，但跨物种泛化有限，且难以输出结构化解释 。
- 这些方法往往在随机划分数据上表现较好，但在 protein-disjoint、cluster-disjoint 和 species split 条件下性能下降更明显 。

### . 结构意识方法

- D-SCRIPT 引入结构感知的 contact map 中间层，证明了仅用序列输入也能学习到与 docking 接触相一致的 residue-level 表征，并在跨物种场景显著优于 PIPR 。
- 但 D-SCRIPT 的结构中间层更偏“接触图复原”，并不是专门围绕高通量筛选中的热点残基对优先级排序而设计 。

### . PLM 高通量方法

- SENSE-PPI 证明冻结 ESM/ESM 类 PLM 后可在大规模 PPI 预测中取得很强的速度与泛化能力，支持 , 蛋白自对接级别的快速筛查 。
- B-PPI 则进一步说明，基于结构感知 embedding 与 cross-attention 的方法可显著提升细菌 PPI 的 AUPRC 和 F，同时维持较快推理速度 。
- 不过，现有快速模型多数仍偏向 pair-level 输出，residue-level hotspot 解释和可导出的注意力机制尚未成为统一主线 。

### . 现有不足

- 当前方法的主要不足集中在三点：一是高通量筛选下 precision 仍不足，误报会放大实验验证成本；二是跨物种泛化仍受训练分布影响；三是残基级解释常停留在“可视化”层面，缺少可直接用于实验设计的热点排序机制 。
- 因此，PPI-Sifter 的切入点是将高通量筛选、准确分类与 residue hotspot mining 统一到一个单阶段架构中 。

***

## 四、研究问题

本项目聚焦以下核心问题：

- 如何在候选蛋白对规模极大时保持足够高的筛选速度，同时尽量抑制 false positive，从而降低实验验证成本 。
- 如何在跨物种、非模式物种、低同源条件下维持稳定的预测精度与排序能力 。
- 如何把 pair-level 互作预测与 residue-level 热点定位统一在同一个模型中，避免解释与预测割裂 。
- 如何让注意力 map 具备可验证性，即其 top-k residue pairs 与已知界面、binding motif、接触图之间存在统计一致性 。
- 如何将模型设计为工程可运行、可复现、可部署的高通量筛选工具，而不是仅停留在论文原型 。

***

## 五、研究意义

### . 理论意义

- 本项目把蛋白互作预测从“二分类输出”推进到“二分类 + 热点解释”的统一建模，能够为蛋白语言模型在交互机制学习上的能力提供更直接的证据 。
- 同时，residue-level attention map 的引入有助于分析模型是否真的学习到物理互作信号，而不是仅仅记忆蛋白频次、物种偏差或同源特征 。

### . 应用意义

- 在实际应用中，PPI-Sifter 可用于蛋白组级高通量候选筛选，帮助实验人员在有限资源下优先选择最值得验证的互作对 。
- 其 attention map 还能直接导出候选界面残基对，为点突变设计、结构验证、功能注释和靶点优先级排序提供依据 。
- 对于非模式物种、病原体-宿主互作、菌群蛋白互作等高噪声场景，该模型具有更高的可部署价值 。

***

## 六、研究基础

### . ESM-C 冻结表征基础

- ESM 系列 PLM 已经证明了蛋白序列中可隐式编码结构、功能与进化信息，SENSE-PPI 也验证了冻结 PLM + 轻量头在高通量 PPI 预测中的可行性 。
- PPI-Sifter 采用冻结 ESM-C 作为 residue embedding 基座，可避免大模型全量微调带来的训练成本与显存开销 。

### . 残基级注意力基础

- D-SCRIPT 已证明 residue-contact map 可以在仅用序列的情况下逼近真实结构接触，并具有生物学解释价值 。
- ICAN、CmhAttCPI 等也表明 attention 机制可捕获 binding site 相关信号，说明 residue-level attention 在 PPI 解释中有明确技术基础 。

### . 数据与评测基础

- STRING、BioGRID、DIP、IntAct 等数据库已支持大规模训练与 benchmark，且已有成熟的 random / protein-disjoint / cluster-disjoint / species split 评测协议 。
- SENSE-PPI 与 D-SCRIPT 的实验结果为高通量 PPI 筛选设定了可参照的性能与速度基线 。
- B-PPI 则进一步说明 residue-aware cross-attention 结构可在  正负比的真实筛选设定中保持高 AUPRC 与高 F 。

***

## 七、总体目标

本项目拟实现一个面向高通量 PPI 筛选的单阶段模型，满足以下目标：

- 在大规模候选蛋白对筛选中保持高吞吐与低延迟 。
- 在严格数据划分下维持稳定的 AUPRC、AUROC、F 和 MCC 表现 。
- 输出 residue-level attention map，并可自动提取热点残基对 。
- 在 benchmark、消融和解释性分析中形成完整、可复现、可提交的实验闭环 。

***

## 八、模型整体设计

### . 输入与编码

- 输入为两条蛋白氨基酸序列，首先通过冻结 ESM-C 编码为 residue-level embeddings 。
- ESM-C 只承担通用序列表征提取，不参与任务微调，从而把可训练参数集中在交互头与解释模块上 。

### . Protein-level fast filter

- fast filter 负责基于 protein-level pooling 特征快速形成 pair-level 粗判分数，以低成本排除明显负样本 。
- 这一模块应尽量轻量化，可采用 masked attentive pooling、sum/abs-diff fusion、Hadamard product 或其组合，以保证吞吐率 。

### . Residue-level attentive reranker

- 对通过 fast filter 的候选 pair，residue-level attentive reranker 建模双向 residue-residue 交互，通过 cross-attention 生成注意力 map 。
- 该模块输出不仅用于最终分类，也用于热点提取与界面解释，因此需要保留较高分辨率和可聚合性 。

### . 输出层

- 模型最终输出三类结果：PPI 概率、attention map、top-k residue pair list 。
- 这样既能完成高通量筛查，也能为实验验证提供具体候选位点 。

### . 核心创新点

- 核心创新在于：冻结大模型保持高吞吐，protein-level 模块负责快速筛选，residue-level attention 负责细粒度解释，三者统一在一个单阶段训练框架内 。
- 该设计兼顾效率、精度与可解释性，尤其适合真实高通量筛选任务 。

***

## 九、损失函数设计

### . 分类损失

- pair-level 分类建议采用 Weighted BCE 与 Focal Loss 的组合，以适配 PPI 中正负样本极不平衡的现实分布 。
- Weighted BCE 强化正类识别，Focal Loss 聚焦难例与边界样本，从而提升 precision、AUPRC 和收敛效率 。

### . 注意力正则

- 为使 attention map 更像真实界面，应加入稀疏正则、对称一致性约束与热点一致性约束 。
- 稀疏正则鼓励少量高峰值注意力，避免大片扩散；对称约束保证 A→B 与 B→A 的一致性；热点一致性可在有结构或界面标注时对齐真实 residue contact 。

### . 总损失

- 总损失可写为：

$$
L = L_{cls} + \lambda_ L_{focal} + \lambda_ L_{sparse} + \lambda_ L_{sym} + \lambda_ L_{hotspot}
$$

- 其中 $L_{cls}$ 为主任务分类项，后四项分别约束样本不均衡、注意力稀疏性、双向一致性和热点可解释性 。
- 该设计的目标是让模型在保持高通量预测速度的同时，尽可能输出更可信的 residue-level 解释 。

***

## 十、技术路线图

### . 数据准备

- 整理 STRING、BioGRID、DIP、IntAct 等互作数据，并按物种、同源去冗余、cluster-disjoint 规则构建训练和测试集 。
- 负样本采用邻接排除或随机配对策略，并保持 : 左右的正负比例，以更接近真实筛查分布 。

### . 表征提取

- 使用冻结 ESM-C 提取每条蛋白的 residue embeddings，并离线缓存，以减少重复推理开销 。
- 该步骤是高通量运行的关键，因为 embedding 一次计算、多次复用，可显著降低 all-vs-all 筛查成本 。

### . 快速筛选

- 对所有候选 pair 先通过 protein-level fast filter 进行快速打分，筛掉大部分低概率负例 。
- 此步骤重点优化吞吐量和 precision-first 排序能力。

### . 残基重排

- 对 fast filter 保留下来的候选 pair，使用 residue-level attentive reranker 进行双向交互建模与 attention map 生成 。
- 该模块负责提升 final ranking 的精细度，并输出热点残基对 。

### . 解释与导出

- 将 attention map 聚合为 top-k residue pairs，并计算 interface enrichment、attention entropy、symmetry consistency 等指标 。
- 最终形成 pair-level 结果表、热点列表和可视化 heatmap，供实验优先级排序使用 。

***

## 十一、基准实验

### . 对比基线

- 建议对比 PIPR、D-SCRIPT、Topsy-Turvy、SENSE-PPI、B-PPI，以及简单的 ESM-C mean pooling baseline 。
- 这些方法分别代表传统 sequence model、结构感知模型、图融合模型、冻结 PLM 快筛模型和 residue-aware cross-attention 模型 。

### . 评价指标

- pair-level 指标包括 AUPRC、AUROC、F、MCC、Precision 和 Recall 。
- 解释性指标包括 top-k Jaccard、interface overlap、attention entropy、symmetry consistency 与 seed stability 。
- 高通量指标包括每秒预测 pair 数、每 GPU 吞吐量、全蛋白组扫描耗时与显存占用 。

### . 参考结果

- 已有公开结果显示，SENSE-PPI 在 human STRING. 上达到 AUPRC .、AUROC .、F .，并可在 , 蛋白自对接中于数小时内完成筛查 。
- D-SCRIPT 在跨物种设置下显著优于 PIPR，且可输出 contact map；B-PPI 在 bacterial PPI 上实现 AUPRC .、F .，并显著优于 TTD 。
- PPI-Sifter 的 benchmark 目标是：在不牺牲速度的前提下，进一步提升 precision 并强化 residue-level hotspot 输出 。

### . Benchmark 表

| 模型 | 代表特点 | 公开性能 | 适配高通量 | 残基解释 |
| :-- | :-- | --: | :-- | :-- |
| PIPR | 早期 Siamese 序列模型 | human AUPR .  | 中 | 弱 |
| D-SCRIPT | 结构感知 contact map | human AUPR .，AUROC .  | 高 | 强 |
| Topsy-Turvy | 图融合序列模型 | human AUPR .  | 高 | 中 |
| SENSE-PPI | 冻结 PLM 快筛 | human AUPRC .，AUROC .  | 很高 | 中 |
| B-PPI | cross-attention residue model | AUPRC .，F .  | 很高 | 中高 |
| PPI-Sifter | 冻结 ESM-C + fast filter + reranker | 目标：在高吞吐下进一步提高 precision 与 hotspot 质量 | 很高 | 很强 |


***

## 十二、消融实验

### . 冻结策略消融

- 比较冻结 ESM-C、部分微调 ESM-C、完全微调 ESM-C 三种方案，记录 AUPRC、F、MCC 与 wall-clock time 。
- 预期冻结方案在高通量任务中更优，因为训练和推理成本更低，且更稳定 。

### . 模块消融

- 比较“仅 fast filter”“仅 reranker”“fast filter + reranker”“去掉 attention 正则”“完整模型”等设置 。
- 重点验证 residue-level attentive reranker 是否显著提升 precision、热点定位和跨物种泛化 。

### . Loss 消融

- 比较 BCE、Weighted BCE、Focal、Weighted BCE + Focal、完整联合 loss，观察对精度和收敛速度的影响 。
- 预期 Focal 与 attention 正则可显著降低难例误报，并提升 top-k hotspot 质量 。

### . 速度消融

- 在不同 batch size、不同序列长度截断、不同 attention head 数、不同 top-k 筛选阈值下测算吞吐量与显存占用 。
- 该部分是项目高通量主线的工程验证，必须以 pair/s、proteome scan time、GPU memory 作为核心指标 。

***

## 十三、可运行实现方案

### . 环境配置

- 推荐 Python .+、PyTorch .+、CUDA .x，配套 `esm`、`transformers`、`pandas`、`numpy`、`scikit-learn`、`hydra`、`matplotlib`、`seaborn` 。
- 在大规模推理时建议使用 AMP、gradient accumulation、padding mask、LMDB/mmap 缓存和 block-sparse/top-k attention 。

### . 训练流程

- 训练流程包括：FASTA 解析、ESM-C embedding 缓存、pair 数据集构建、batch 组装、分类训练、验证集调参、early stopping 与最佳模型导出 。
- 建议以 AUPRC 为主验证指标，并同步保存 precision-recall 曲线、attention map 和候选热点文件 。

### . 推理流程

- 推理阶段支持单对预测、批量候选筛选和 proteome-scale all-vs-all 扫描 。
- 先离线计算 residue embeddings，再进行 pair 级交互推理，可以把最昂贵的编码开销一次性摊销到多个 pair 上 。

### . 部署适配

- 可封装为 Python CLI 或简单服务接口，输出包括 prediction table、heatmap、top-k residues、metrics report 四类结果 。
- 若用于实验室部署，应允许用户自定义阈值、top-k、长度截断、物种列表和负采样比例 。

***

## 十四、注意力解释分析

### . 分析目标

- 注意力 map 不只用于可视化，而应作为 residue hotspot 的统计证据 。
- 重点验证 attention peak 是否富集于真实界面、binding site 或保守 motif 区域 。

### . 评价方法

- 建议计算 top-k Jaccard、interface enrichment、attention entropy、symmetry consistency、seed stability 与 motif overlap 。
- 如果模型在正样本中呈现低熵、局部尖峰、双向一致且与已知接触图高度重合，则说明 attention 具有生物学可解释性 。

### . 应用方式

- 对于高置信 pair，可直接输出候选残基对供突变验证或结构对接优先级排序 。
- 对于家族内不同成员，可比较 attention pattern 的差异，从而分析互作特异性来源 。

***

## 十五、进度安排

### 第一阶段：数据与原型

- 完成数据收集、去冗余、负样本构建、训练/验证/测试划分，以及 ESM-C embedding 缓存流程 。
- 实现 baseline 和初版 PPI-Sifter 原型，确保训练与推理链路可跑通 。

### 第二阶段：模型定型

- 完成 fast filter、residue-level attentive reranker、联合 loss 与 attention 正则设计 。
- 在 benchmark 上比较不同配置，确定最优超参数与推理策略 。

### 第三阶段：消融与解释

- 完成冻结策略、模块结构和 loss 设计的系统消融，整理 attention map 解释分析结果 。
- 形成可以提交的完整实验表格、图示和统计结论 。

### 第四阶段：工程化与交付

- 整理 CLI、配置文件、模型权重、训练脚本、推理脚本与可视化脚本，形成可运行版本 。
- 最终输出项目设计书、实验报告和部署说明 。

***

## 十六、风险与对策

### . 精度不足风险

- 风险表现为高通量筛查时 false positive 偏高，导致实验验证成本上升 。
- 对策是采用 Weighted BCE + Focal Loss，并加强对 attention map 的稀疏与一致性正则 。

### . 泛化不足风险

- 风险表现为跨物种、低同源或非模式物种上性能下降 。
- 对策是采用 protein-disjoint 和 species split 评测，必要时在同类物种上做轻量适配，但仍保持主模型冻结 ESM-C 的设定 。

### . 速度瓶颈风险

- 风险表现为长序列 pair 在 attention 计算中占用显存过高 。
- 对策是引入 embedding 缓存、长度裁剪、top-k attention、AMP 与 batch 分级推理 。

### . 解释失真风险

- 风险表现为 attention map 与真实界面不一致，导致解释结论不可靠 。
- 对策是加入 symmetry、sparsity 与 hotspot consistency 约束，并用结构复合物数据做统计验证 。

***

## 十七、预期创新点

- **单阶段统一建模。** 在同一模型内同时完成高通量筛选、pair 分类与 residue hotspot 输出，避免预测与解释割裂 。
- **冻结 ESM-C 的高通量方案。** 以冻结 PLM 作为主干，最大化吞吐并降低训练成本 。
- **残基级注意力可解释输出。** 直接将 attention map 用于热点残基对筛选，而非仅做可视化展示 。
- **面向高通量的评价体系。** 同时报告 AUPRC、Precision、F、MCC、pair/s、GPU 占用和热点富集指标，确保结论围绕“快”和“准”展开 。

***

## 十八、预期成果

- 项目完成后，预期形成一套可直接运行的 PPI 高通量筛选模型、完整 benchmark、系统消融实验和 attention 可解释性分析报告 。
- 成果应能支撑“高通量筛选效率提升、误报控制[references.txt](..%2Freferences%2Freferences.txt)、热点位点定位”三类结论，并具备可复现、可部署、可扩展的工程基础 。

投稿 ICDM 2026、BIBM 2026、bioinformatics、TCBB 等。

## 十九、参考文献
- [1] AttnSeq-PPI: Attention-based Sequence Model for Protein-Protein Interaction Prediction [R].2024.
- [2] B-PPI: A Bi-directional Feature Fusion Network for Protein-Protein Interaction Prediction [EB/OL].bioRxiv,2025.
- [3] CmhAttCPI: Compound-Multihead Attention Model for Compound-Protein Interaction Prediction [R].2024.
- [4] DeepPFP-CO: Deep Learning Based Protein Function Prediction with Co-annotation Constraint [J].IEEE/ACM Transactions on Computational Biology and Bioinformatics,2022.
- [5] D-SCRIPT: Deep Sequence-driven Interaction Prediction for Protein Pairs [J].Cell Systems,2021.
- [6] ICAN: Interpretable Cross-Attention Network for Protein-Protein Interaction Prediction [J].PLOS ONE,2022.
- [7] SENSE-PPI: Sparse Evolutionary Feature Integrated Attention Network for PPI Prediction [J].iScience,2024.
- [8] Topsy-Turvy: Cross-species Protein Interaction Prediction with Contrastive Learning [J].Bioinformatics,2022.

