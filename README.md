# PPI-Sifter

PPI-Sifter 是一个面向高通量蛋白互作筛选与残基热点解释的一体化序列模型。该版本已完全更正为“前半段与 B-PPI 一致”的设计：输入为冻结 ESM-C residue embeddings，经输入投影、双向 cross-attention、gated FFN、attention pooling、对称共享表示和 MLP 分类头输出 pair-level interaction probability；在此基础上，额外导出 residue-level attention map 与 top-k residue pairs。

## 设计原则

- 保持前半段与 B-PPI 主干一致。
- 将 PPI-Sifter 的创新集中在解释性输出，而不是改变前端判别框架。
- 支持训练、评估、推理、attention 导出与热图可视化。


根据你的项目现状，整个流程分为以下几步。

## 前提：确认嵌入已就绪

```bash
ls data/BIOGRID/embeddings/residue/batch_*.npz | wc -l
# 应该有若干个 batch 文件
```

如果嵌入还没跑完，先补全：

```bash
python faa_2_residue_level_embedding_optimized_v2.py \
  --input data/BIOGRID/all.clean.faa \
  --outdir data/BIOGRID/embeddings/residue/ \
  --batch_size 200 --dtype float16 --compresslevel 1
```

***

## Step 1：确认 pair CSV 已存在

你的 config 里需要：

```
data/BIOGRID/pairs/train.csv
data/BIOGRID/pairs/valid.csv
data/BIOGRID/pairs/test.csv
```

每个 CSV 至少包含列 `protein_a`, `protein_b`, `label`（0/1）。如果还没有，先用你已有的 `scripts/biogrid/` 里的脚本生成。

***

## Step 2：安装包

```bash
cd /path/to/PPI-Sifter
pip install -e .
```

***

## Step 3：训练

```bash
python scripts/train/train.py --config configs/default.yaml
```

启动后会先扫描所有 `batch_*.npz` 建索引（约几秒），然后开始训练，checkpoint 保存到 `outputs/checkpoints/`。

***

## Step 4：评估

```bash
python scripts/train/eval.py \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --pairs data/BIOGRID/pairs/test.csv
```

输出 AUROC、AUPRC、F1 等指标到终端。

***

## Step 5：推理（无标签预测）

```bash
python scripts/train/infer.py \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --pairs data/BIOGRID/pairs/test.csv \
  --output outputs/preds/test_pred.csv
```

结果保存为 `protein_a, protein_b, interaction_prob` 的 CSV。

***

## 如果中途报错

- `No batch_*.npz found`：`embeddings_dir` 路径不对，检查 `configs/default.yaml` 里的 `embeddings_dir`。
- `[PairDataset] dropping N rows`：有蛋白在 pair CSV 里但没有嵌入，属于正常现象（嵌入未跑完的蛋白会被跳过）。
- CUDA OOM：把 `configs/default.yaml` 里 `train.batch_size` 调小，比如改成 `4` 或 `8`。
- `num_workers > 0` 报错：把 config 里 `num_workers` 保持 `0`，原因是 `NPZStore` 的 zip 句柄不能跨进程复用。