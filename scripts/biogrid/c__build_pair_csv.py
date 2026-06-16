#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
c__build_pair_csv.py
====================
用途：
1. 读取 BioGRID 原始互作表。
2. 提取蛋白对 accession，构建正样本对。
3. 优先读取 b__faa_record_id_sup.py 生成的 embedding ID 索引文件；
   若索引文件不存在，则自动回退为扫描 batch_*.npz。
4. 仅保留双端蛋白都存在 embedding 的正样本对。
5. 基于正样本构建负样本。
6. 输出 train / valid / test 三个 CSV 与 stats.json。

说明：
- 本版本取消命令行传参，直接按脚本顶部“配置区”执行。
- 若你已经运行过 b__faa_record_id_sup.py，通常无需重新扫描 npz。
- 默认行为：优先读取 embedding 索引 txt；只有索引不存在时才扫描 npz。
"""

import json
import random
import re
import sys
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd


# ── 配置区（建议只在这里修改）─────────────────────────────────────────────────
BIOGRID_PATH = Path("data/BIOGRID/BIOGRID-ALL-4.4.240.csv.gz")
# BioGRID 原始互作表路径。
# 该文件通常来自官方下载的 .csv.gz，包含 Interactor A/B 的 accession、物种、实验类型等字段。

EMBDIR = Path("data/BIOGRID/embeddings/residue")
# residue embedding 所在目录。
# 仅在“索引文件不存在”时才会使用该目录，逐个扫描其中的 batch_*.npz。

OUTDIR = Path("data/BIOGRID/pairs")
# 输出目录。
# 最终会写出 train.csv、valid.csv、test.csv、stats.json。

EMB_INDEX_TXT = Path(
    "data/BIOGRID/faa_chunks/"
    "BIOGRID-ALL-4.4.240.uniprot.embedding.uniprot_ids.txt"
)
# embedding ID 索引文件。
# 该文件由 b__faa_record_id_sup.py 生成，内容通常是：
#   1) 从 batch_*.npz 提取到的 embedding accession
#   2) 再与 all.clean.faa 取交集后的 accession 列表
# 本脚本默认优先读取它，以避免每次都重新扫描全部 npz 文件。

NEG_RATIO = 1.0
# 负样本比例。
# 定义：负样本数 = 正样本数 * NEG_RATIO。
# - 1.0 表示正负样本 1:1 平衡。
# - 若设为 2.0，则负样本数量是正样本的两倍。
# 一般在二分类训练中，1.0 是最稳妥、最容易对比实验的默认值。

TRAIN_RATIO = 0.8
# 训练集比例。
# 表示全部样本（正+负）中有 80% 分配给 train.csv。

VALID_RATIO = 0.1
# 验证集比例。
# 表示全部样本中有 10% 分配给 valid.csv。
# 剩余部分会自动分给 test.csv，因此当前配置等价于：
#   train : valid : test = 0.8 : 0.1 : 0.1

SEED = 42
# 随机种子。
# 用于：
# 1) 负样本随机采样
# 2) 全量样本随机打乱后再切分 train / valid / test
# 固定该值可以保证多次运行结果可复现。

ORGANISM_IDS = {9606}
# 物种过滤条件。
# 9606 = Homo sapiens（人类）。
# 若只做人类 PPI，就保留该设置。
# 若不想按物种过滤，可设为 None 或空集合（需同时调整下方判断逻辑的使用习惯）。

EXP_TYPE = "physical"
# 实验类型过滤条件。
# BioGRID 中常见如："physical"、"genetic"。
# PPI 建模通常只保留 physical interaction，更符合蛋白质物理互作定义。
# 若不想过滤实验类型，可设为 None。

STRICT_MODE = False
# 预留开关，当前脚本未实际使用。
# 未来可扩展为：
# - 更严格的 accession 清洗
# - 更严格的 BioGRID 字段校验
# - 损坏样本直接报错而不是跳过
# 当前保留该变量是为了后续扩展与配置风格统一。
# ─────────────────────────────────────────────────────────────────────────────


def normalize_id(x) -> Optional[str]:
    """归一化为 UniProt Accession，保持与 b__faa_record_id_sup.py 一致。"""
    if x is None:
        return None
    s = str(x).strip()
    if s in ("", "-", "nan", "NaN", "None"):
        return None
    if s.endswith(".npy"):
        s = s[:-4]
    m = re.match(r"[a-z]{2}\|([A-Z0-9]{1,10}(?:-\d+)?)\|", s)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-Z][A-Z0-9]{4,9}(?:-\d+)?)\b", s)
    if m:
        return m.group(1)
    return None


def parse_accession_cell(cell) -> list[str]:
    """解析 BioGRID 单元格中的 accession 列，支持多值拆分。"""
    if cell is None:
        return []
    parts = re.split(r"[|;,\s]+", str(cell).strip())
    return [acc for p in parts if (acc := normalize_id(p))]


def get_best_accession(sp_cell, tr_cell) -> Optional[str]:
    """
    Swiss-Prot 优先，其次 TrEMBL。
    这样做的目的是优先使用人工校验程度更高的 accession。
    """
    for cell in (sp_cell, tr_cell):
        accs = parse_accession_cell(cell)
        if accs:
            return accs[0]
    return None


def _load_from_index(index_txt: Path) -> set[str]:
    """快速路径：从预建索引文件逐行读取 embedding IDs。"""
    print(f"  [快速路径] 从索引文件加载: {index_txt}")
    ids: set[str] = set()
    with open(index_txt, encoding="utf-8") as f:
        for line in f:
            acc = line.strip()
            if acc:
                ids.add(acc)
    print(f"  [快速路径] 加载完成，共 {len(ids):,} 个 embedding IDs")
    return ids


def _load_from_npz(embdir: Path) -> set[str]:
    """
    慢速路径：扫描所有 batch_*.npz，提取内部 key 对应的 accession。
    只有在索引文件缺失时才会使用。
    """
    print(f"  [扫描路径] 扫描 {embdir} 下 batch_*.npz ...")
    emb_ids: set[str] = set()
    batch_files = sorted(embdir.glob("batch_*.npz"))
    if not batch_files:
        print(f"  [WARNING] 未找到 batch_*.npz in {embdir}", file=sys.stderr)
        return emb_ids

    total = len(batch_files)
    for i, bf in enumerate(batch_files, 1):
        try:
            with zipfile.ZipFile(bf, "r") as zf:
                for name in zf.namelist():
                    if name.endswith(".npy"):
                        acc = normalize_id(name[:-4])
                        if acc:
                            emb_ids.add(acc)
        except Exception as e:
            print(f"  [WARNING] 跳过 {bf.name}: {e}", file=sys.stderr)

        if i % 50 == 0 or i == total:
            print(f"  扫描进度 {i}/{total}，累计 IDs: {len(emb_ids):,}")

    print(f"  [扫描路径] 完成，共 {len(emb_ids):,} 个 embedding IDs")
    return emb_ids


def load_embedding_ids(embdir: Path, index_txt: Path) -> set[str]:
    """
    STEP 3 入口：
    - 若索引文件存在，优先读取索引（推荐、快速）
    - 若索引文件不存在，回退为扫描 npz（兼容旧流程）
    """
    if index_txt.exists():
        return _load_from_index(index_txt)

    print(f"  [INFO] 索引文件不存在 ({index_txt.name})，自动扫描 npz")
    return _load_from_npz(embdir)


def build_negative_pairs(pos_pairs, neg_ratio, rng):
    """
    随机负采样。

    逻辑：
    - 从正样本涉及到的全部蛋白集合中随机抽两条蛋白，组成候选 pair。
    - 若该 pair 不在正样本中，且此前未被采过，就收为负样本。
    - 直到负样本数达到 len(pos_pairs) * neg_ratio，或达到最大尝试次数。

    注意：
    - 这是“随机非互作近似”策略，不保证是真负样本，只保证“不在当前正样本表中”。
    - 若后续要做更严格评估，可改成 protein-disjoint / cluster-aware / degree-aware 负采样。
    """
    pos_set = set(pos_pairs)
    proteins = sorted({p for a, b in pos_pairs for p in (a, b)})
    n_neg = int(len(pos_pairs) * neg_ratio)
    neg_pairs: set[tuple[str, str]] = set()

    max_attempts = max(n_neg * 20, 1000)
    attempts = 0
    while len(neg_pairs) < n_neg and attempts < max_attempts:
        a, b = rng.sample(proteins, 2)
        if a > b:
            a, b = b, a
        pair = (a, b)
        if pair not in pos_set and pair not in neg_pairs:
            neg_pairs.add(pair)
        attempts += 1

    return list(neg_pairs)


def save_splits(pos_pairs, neg_pairs, outdir, rng):
    """
    将正负样本合并、打乱，并按 TRAIN_RATIO / VALID_RATIO 切分。
    test 比例不单独设置，而是使用剩余样本自动得到。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_pairs = [(a, b, 1) for a, b in pos_pairs] + [(a, b, 0) for a, b in neg_pairs]
    rng.shuffle(all_pairs)

    n = len(all_pairs)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)
    splits = {
        "train": all_pairs[:n_train],
        "valid": all_pairs[n_train:n_train + n_valid],
        "test": all_pairs[n_train + n_valid:],
    }

    stats = {"total": n, "splits": {}}
    for split_name, rows in splits.items():
        df_out = pd.DataFrame(rows, columns=["protein_a", "protein_b", "label"])
        out_csv = outdir / f"{split_name}.csv"
        df_out.to_csv(out_csv, index=False)

        n_pos = int(df_out["label"].sum())
        stats["splits"][split_name] = {
            "total": len(df_out),
            "positive": n_pos,
            "negative": len(df_out) - n_pos,
        }
        print(f"[INFO] {split_name}: {len(df_out):,} pairs -> {out_csv}")

    stats_path = outdir / "stats.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[INFO] 统计文件 -> {stats_path}")


def build_pairs():
    """主流程入口。"""
    rng = random.Random(SEED)

    # STEP 1: 读取并过滤 BioGRID
    print("[STEP 1] 读取 BioGRID CSV ...")
    df = pd.read_csv(BIOGRID_PATH, compression="gzip", low_memory=False)
    print(f"[INFO] 原始记录: {len(df):,}")

    if ORGANISM_IDS:
        df = df[
            df["Organism ID Interactor A"].isin(ORGANISM_IDS)
            & df["Organism ID Interactor B"].isin(ORGANISM_IDS)
        ]
        print(f"[INFO] 过滤物种后: {len(df):,}")

    if EXP_TYPE:
        df = df[
            df["Experimental System Type"].fillna("").str.lower() == EXP_TYPE.lower()
        ]
        print(f"[INFO] 过滤实验类型后: {len(df):,}")

    # STEP 2: 提取正样本对
    print("[STEP 2] 提取 accession 正样本对 ...")
    records = []
    for _, row in df.iterrows():
        a = get_best_accession(
            row.get("SWISS-PROT Accessions Interactor A"),
            row.get("TREMBL Accessions Interactor A"),
        )
        b = get_best_accession(
            row.get("SWISS-PROT Accessions Interactor B"),
            row.get("TREMBL Accessions Interactor B"),
        )
        if a and b and a != b:
            if a > b:
                a, b = b, a
            records.append((a, b))

    pos_pairs_raw = sorted(set(records))
    print(f"[INFO] 正样本对（去重）: {len(pos_pairs_raw):,}")

    # STEP 3: 加载 embedding IDs
    print("[STEP 3] 加载 embedding IDs ...")
    emb_ids = load_embedding_ids(EMBDIR, EMB_INDEX_TXT)

    # STEP 4: 过滤双端均有 embedding 的正样本对
    print("[STEP 4] 过滤正样本对（双端均有 embedding）...")
    if emb_ids:
        before = len(pos_pairs_raw)
        pos_pairs = [
            (a, b) for a, b in pos_pairs_raw
            if a in emb_ids and b in emb_ids
        ]
        after = len(pos_pairs)
        print(
            f"[INFO] 过滤前: {before:,}  过滤后: {after:,}  保留率: {after / before * 100:.1f}%"
        )

        if after == 0:
            biogrid_ids = {p for a, b in pos_pairs_raw for p in (a, b)}
            overlap = biogrid_ids & emb_ids
            print(f"[DIAGNOSIS] BioGRID 唯一蛋白数: {len(biogrid_ids):,}")
            print(f"[DIAGNOSIS] Embedding ID 数:    {len(emb_ids):,}")
            print(f"[DIAGNOSIS] 交集大小:            {len(overlap):,}")
            print(f"[DIAGNOSIS] BioGRID 样本: {sorted(biogrid_ids)[:20]}")
            print(f"[DIAGNOSIS] Emb ID 样本:  {sorted(emb_ids)[:20]}")
            print(f"[DIAGNOSIS] 交集样本:      {sorted(overlap)[:20]}")
    else:
        pos_pairs = pos_pairs_raw
        print("[WARNING] embedding IDs 为空，STEP 4 不过滤")

    # STEP 5: 构建负样本
    print("[STEP 5] 构建负样本对 ...")
    neg_pairs = build_negative_pairs(pos_pairs, NEG_RATIO, rng)
    print(f"[INFO] 负样本对: {len(neg_pairs):,}")

    # STEP 6: 保存划分结果
    print("[STEP 6] 保存 train/valid/test CSV ...")
    save_splits(pos_pairs, neg_pairs, OUTDIR, rng)
    print("[DONE]")


if __name__ == "__main__":
    build_pairs()
