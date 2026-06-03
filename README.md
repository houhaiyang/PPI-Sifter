# PPI-Sifter

PPI-Sifter 是一个面向高通量蛋白互作筛选与残基热点解释的一体化序列模型工程实现。项目遵循“冻结 ESM-C + protein-level fast filter + residue-level attentive reranker”的设计主线，支持嵌入缓存、1:10 正负样本构造、训练评估、批量推理与基于注意力的可解释性分析。

## 1. 项目特点

- 冻结 residue embedding 主干，避免大模型全量微调的高训练成本。
- 采用 protein-level 快速筛选与 residue-level 重排序的两阶段头部结构。
- 支持 attention map 导出、top-k 残基对热点提取与热图可视化。
- 所有路径均通过 YAML 配置或 CLI 参数传入，避免硬编码。
- 对 `data/BIOGRID/embeddings/*.residue_emb.npy` 的分批落地场景友好，预处理脚本会自动过滤缺失嵌入并记录状态。

## 2. 环境依赖

建议环境：

- Python >= 3.10
- PyTorch >= 2.2
- numpy
- pandas
- PyYAML
- scikit-learn
- matplotlib
- seaborn
- tqdm
- biopython

安装示例：

```bash
pip install torch numpy pandas pyyaml scikit-learn matplotlib seaborn tqdm biopython
```

如需重新抽取 ESM-C 嵌入，可额外安装与你本地环境兼容的 `esm` 相关依赖。

## 3. 目录结构

```text
PPI-Sifter/
├── README.md
├── configs/
│   ├── default.yaml
│   ├── train.yaml
│   ├── infer.yaml
│   └── interpret.yaml
├── data/
│   └── BIOGRID/
│       ├── embeddings/
│       ├── pairs/
│       └── splits/
├── docs/
│   └── PPI-Sifter_项目设计书-0603-残基互作.md
├── ppisifter/
│   ├── __init__.py
│   ├── constants.py
│   ├── utils.py
│   ├── data.py
│   ├── losses.py
│   ├── model.py
│   ├── attention.py
│   └── interpret.py
├── scripts/
│   ├── __init__.py
│   ├── emb/
│   │   ├── extract_embeddings.py
│   │   ├── merge_embeddings.py
│   │   ├── build_pairs.py
│   │   └── split_dataset.py
│   ├── train/
│   │   ├── train.py
│   │   ├── eval.py
│   │   ├── infer.py
│   │   └── save_checkpoint.py
│   └── interpretability/
│       ├── export_attention.py
│       ├── visualize_attention.py
│       └── feature_importance.py
└── outputs/
    ├── checkpoints/
    ├── preds/
    └── interpret/
```

## 4. 数据预处理流程

### 4.1 前置假设

项目默认 `data/BIOGRID/embeddings/` 下已经逐步生成了所有或部分蛋白的 residue embedding 文件：

```text
{protein_id}.residue_emb.npy
```

每个文件 shape 为 `[L, D]`，其中 `L` 为蛋白长度，`D` 为 embedding 维度。

### 4.2 可选：从 FASTA 重新抽取嵌入

```bash
python scripts/emb/extract_embeddings.py   --config configs/default.yaml   --input_fasta data/BIOGRID/all.clean.faa   --output_dir data/BIOGRID/embeddings
```

### 4.3 合并已存在嵌入索引

```bash
python scripts/emb/merge_embeddings.py   --config configs/default.yaml   --output data/BIOGRID/embeddings/merged_embeddings.pt
```

该脚本不会强制要求嵌入齐全，而是会输出可用蛋白索引、缺失列表与维度统计，便于分批补跑。

### 4.4 构造正负样本对

```bash
python scripts/emb/build_pairs.py   --config configs/default.yaml   --biogrid_csv data/BIOGRID/BIOGRID-ALL.csv   --output_dir data/BIOGRID/pairs
```

该脚本会：

- 清洗并规范化蛋白 ID；
- 保留两端均存在 embedding 的正样本；
- 构造 1:10 的负样本；
- 输出 `all_pairs.csv`、`build_report.json` 等文件。

### 4.5 划分训练/验证/测试集

```bash
python scripts/emb/split_dataset.py   --config configs/default.yaml   --pairs_csv data/BIOGRID/pairs/all_pairs.csv   --output_dir data/BIOGRID/splits
```

默认支持 `protein_disjoint` 与 `random` 两种划分策略，推荐优先使用 `protein_disjoint`。

## 5. 训练

```bash
python scripts/train/train.py   --config configs/train.yaml
```

训练输出：

- `outputs/checkpoints/best.pt`
- `outputs/checkpoints/last.pt`
- `outputs/checkpoints/train_history.json`

## 6. 测试与推理

评估测试集：

```bash
python scripts/train/eval.py   --config configs/train.yaml   --checkpoint outputs/checkpoints/best.pt   --split test
```

批量推理：

```bash
python scripts/train/infer.py   --config configs/infer.yaml   --checkpoint outputs/checkpoints/best.pt   --input_csv data/BIOGRID/splits/test.csv   --output_csv outputs/preds/test_predictions.csv
```

## 7. 可解释性分析

导出 attention：

```bash
python scripts/interpretability/export_attention.py   --config configs/interpret.yaml   --checkpoint outputs/checkpoints/best.pt   --input_csv data/BIOGRID/splits/test.csv
```

生成热图：

```bash
python scripts/interpretability/visualize_attention.py   --attention_csv outputs/interpret/sample_attention.csv   --output_png outputs/interpret/sample_attention.png
```

计算特征重要性：

```bash
python scripts/interpretability/feature_importance.py   --config configs/interpret.yaml   --checkpoint outputs/checkpoints/best.pt   --input_csv data/BIOGRID/splits/test.csv
```

## 8. 部署说明

- 推荐流程为“离线缓存 embedding -> 批量构造 pair -> 训练/推理 -> 导出解释结果”。
- 线上服务化时，建议仅暴露 pair 预测和 attention 导出接口，并将 embedding 计算前置为异步任务。
- 对超长序列可使用配置中的长度截断参数，或在推理时启用 top-k attention 近似。
- 若 embedding 文件仍在持续生成，可周期性重新运行 `merge_embeddings.py` 与 `build_pairs.py`，无需删除历史结果。

## 9. 复现实验建议

- 主验证指标建议使用 AUPRC。
- 样本不均衡时建议启用 `weighted_bce + focal` 联合损失。
- 可解释性建议同时关注 attention entropy、symmetry gap 与 top-k residue pairs。
