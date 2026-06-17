## PPI-Sifter 完整流水线说明

***

### Step 1 — `scripts/biogrid/a__get_biogrid_faa.py`

**功能**：从 BioGRID 原始交互数据库文件出发，通过 UniProt API 批量下载所有涉及蛋白的氨基酸序列，生成标准 FASTA 文件。

**输入**：

- `data/BIOGRID/BIOGRID-*.tab3.txt`（BioGRID 官网下载的交互记录，含 UniProt AC 列）

**输出**：

- `data/BIOGRID/sequences.faa`（全量蛋白序列，FASTA 格式，header 为纯 UniProt AC，如 `>P27695`）

***

### Step 2 — `scripts/biogrid/faa_2_residue_level_embedding_optimized_v2.py`

**功能**：调用冻结的 ESM-C 600M 模型，逐批次将 FASTA 中每条蛋白序列转为 residue-level embedding，写入分批压缩 npz 文件。支持断点续跑（`done_ids.txt` 记录已完成的蛋白 ID）。

**输入**：

- `data/BIOGRID/sequences.faa`（Step 1 输出）
- ESM-C 600M 模型权重（`MODEL_PATH` 硬编码，需按实际路径修改）

**输出**：

- `data/BIOGRID/embeddings/residue/batch_000000.npz`
- `data/BIOGRID/embeddings/residue/batch_000001.npz`
- …（每批 200 条，npz 内 key=UniProt AC，value=`ndarray(L, 1152)`，dtype=float16）
- `data/BIOGRID/embeddings/residue/done_ids.txt`（已完成蛋白 ID 列表，断点续跑凭证）
- `data/BIOGRID/embeddings/residue/generate_meta.json`（运行统计）

```bash
python scripts/biogrid/faa_2_residue_level_embedding_optimized_v2.py \
    --input  data/BIOGRID/sequences.faa \
    --outdir data/BIOGRID/embeddings/residue \
    --device cuda \
    --batch_size 200 \
    --dtype float16
```


***

### Step 3 — `scripts/biogrid/merge_existing_embeddings_compressed_v2.py`

**功能**：将多个 `batch_*.npz` 文件合并为一个统一的大 npz，便于后续统计与校验。**此步可选**——`npz_to_hdf5.py` 可直接读取多个分批 npz，无需先合并。

**输入**：

- `data/BIOGRID/embeddings/residue/batch_*.npz`（Step 2 输出）

**输出**：

- `data/BIOGRID/embeddings/residue/merged.npz`（全量合并，key=UniProt AC）

***

### Step 4 — `scripts/biogrid/b__faa_record_id.py`

**功能**：校验 `sequences.faa` 中的 record ID 格式是否为纯 UniProt AC（如 `P27695`），统计不符合格式的条目并输出报告。

**输入**：

- `data/BIOGRID/sequences.faa`

**输出**：

- 控制台报告：格式不合格的 record ID 列表及数量
- （可选）`data/BIOGRID/id_check_report.txt`

***

### Step 5 — `scripts/biogrid/b__faa_record_id_sup.py`

**功能**：补充修复 Step 4 发现的 ID 格式问题。对于 header 带前缀（如 `sp|P27695|APEX1_HUMAN`）的序列，截取第二段作为纯 AC，重写 faa 文件保证 key 统一。

**输入**：

- `data/BIOGRID/sequences.faa`（含不合格 ID 的原始文件）

**输出**：

- `data/BIOGRID/sequences_clean.faa`（所有 record ID 统一为纯 UniProt AC）

***

### Step 6 — `scripts/biogrid/c__build_pair_csv_enhanced.py`

**功能**：从 BioGRID 交互记录生成训练用蛋白对 CSV，实现 protein-disjoint 划分（训练集/验证集/测试集之间无蛋白重叠），同时生成负样本（非交互对）。

**输入**：

- `data/BIOGRID/BIOGRID-*.tab3.txt`（BioGRID 原始交互记录）
- `data/BIOGRID/embeddings/residue/done_ids.txt` 或 HDF5 key 列表（**只保留有 embedding 的蛋白对**）

**输出**：

- `data/BIOGRID/pairs/proteindisjoint/train.csv`
- `data/BIOGRID/pairs/proteindisjoint/valid.csv`
- `data/BIOGRID/pairs/proteindisjoint/test.csv`
- 每个 CSV 含列：`protein_a, protein_b, label`（1=交互，0=非交互）

***

### Step 7 — `scripts/emb/npz_to_hdf5.py`

**功能**：将所有分批 npz 文件转换为单个 HDF5 文件，供训练时随机访问（避免 OOM）。

**输入**：

- `data/BIOGRID/embeddings/residue/batch_*.npz`（Step 2 输出）

**输出**：

- `data/BIOGRID/embeddings/embeddings.h5`（HDF5，key=UniProt AC，value=`ndarray(L, 1152)`，dtype=float32）

**运行命令**：

```bash
python scripts/emb/npz_to_hdf5.py \
    --npz_dir data/BIOGRID/embeddings/residue \
    --out_h5  data/BIOGRID/embeddings/embeddings.h5
```


***

### Step 8 — 训练

```bash
python scripts/train/train.py
```


***


