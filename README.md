# PPI-Sifter

PPI-Sifter 是一个面向高通量蛋白互作筛选与残基热点解释的一体化序列模型。该版本已完全更正为“前半段与 B-PPI 一致”的设计：输入为冻结 ESM-C residue embeddings，经输入投影、双向 cross-attention、gated FFN、attention pooling、对称共享表示和 MLP 分类头输出 pair-level interaction probability；在此基础上，额外导出 residue-level attention map 与 top-k residue pairs。

## 设计原则

- 保持前半段与 B-PPI 主干一致。
- 将 PPI-Sifter 的创新集中在解释性输出，而不是改变前端判别框架。
- 支持训练、评估、推理、attention 导出与热图可视化。

