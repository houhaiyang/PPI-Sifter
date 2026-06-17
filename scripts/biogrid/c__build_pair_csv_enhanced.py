#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
c__build_pair_csv.py  (v2 — benchmark + ablation 版)
=====================================================
用途：
  1. 读取 BioGRID 原始互作表，提取正样本对（带 species / taxa 元信息）。
  2. 加载 embedding ID 索引，过滤双端均有 embedding 的正样本。
  3. 按指定 SPLIT_MODE 划分 train / valid / test：
       - random           : 按蛋白对随机划分（附录 / smoke test 用）
       - protein_disjoint : 按蛋白节点划分，train/test 不共享任何蛋白
       - cluster_disjoint : 按 MMseqs2 cluster 划分（需提供 cluster TSV）
       - species_disjoint : 按 taxonomy 物种划分，train/test 不共享物种
  4. 各 split 内独立进行负样本采样，避免 test 信息反向污染 train。
  5. 输出各 split 的 train/valid/test CSV，并自动生成 leakage_report.json。

运行依赖：
  pip install pandas

入口：
  python c__build_pair_csv.py

注意：
  - cluster_disjoint 模式需提前提供 CLUSTER_TSV（MMseqs2 easy-cluster 结果）。
  - species_disjoint 模式需 BioGRID 中包含 "Organism ID Interactor A/B" 字段。
  - 所有输出写入 OUTDIR/<split_mode>/ 子目录，互不干扰，便于多模式并行实验。
"""

import json
import random
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd


# ── 配置区（建议只在这里修改）─────────────────────────────────────────────────

BIOGRID_PATH = Path("data/BIOGRID/BIOGRID-ALL-4.4.240.csv.gz")
# BioGRID 原始互作表路径（.csv.gz）。

EMBDIR = Path("data/BIOGRID/embeddings/residue")
# residue embedding 目录，仅索引文件缺失时扫描。

OUTDIR = Path("data/BIOGRID/pairs")
# 输出根目录；各 split 模式写入 OUTDIR/<split_mode>/ 子目录。

EMB_INDEX_TXT = Path(
    "data/BIOGRID/faa_chunks/"
    "BIOGRID-ALL-4.4.240.uniprot.embedding.uniprot_ids.txt"
)
# b__faa_record_id_sup.py 生成的 embedding accession 索引文件。

CLUSTER_TSV = Path("data/BIOGRID/mmseqs2/cluster_result_cluster.tsv")
# MMseqs2 easy-cluster 输出的两列 TSV（representative\tmember）。
# 仅 cluster_disjoint 模式使用；其他模式可忽略。

# ── 划分模式 ──────────────────────────────────────────────────────────────────
SPLIT_MODE = "protein_disjoint"
# 支持四种取值：
#   "random"           — 直接按 pair 随机划分，适合 smoke test / 附录
#   "protein_disjoint" — 按蛋白节点划分，主 benchmark 之一（推荐优先跑）
#   "cluster_disjoint" — 按序列簇划分，需提供 CLUSTER_TSV
#   "species_disjoint" — 按物种 / taxa 划分，跨物种泛化 benchmark

SPLIT_VERSION = "v1"
# 版本号，写入 stats 文件，便于追踪实验版本。

# ── 比例与采样 ────────────────────────────────────────────────────────────────
NEG_RATIO = 1.0
# 负样本数 = 正样本数 × NEG_RATIO；1.0 表示 1:1 平衡。

TRAIN_RATIO = 0.7
# 训练集占比（按蛋白/簇/物种数量划分时，指对应分组的比例）。

VALID_RATIO = 0.15
# 验证集占比；剩余自动分给 test。
# 默认 train:valid:test = 0.7:0.15:0.15。

SEED = 42
# 全局随机种子，保证实验可复现。

# ── 数据过滤 ──────────────────────────────────────────────────────────────────
# ORGANISM_IDS = {9606}
ORGANISM_IDS = None
# 物种过滤。9606 = Homo sapiens。
# 注意：species_disjoint 模式会忽略此过滤，以保留足够多的物种分组。
# 若不过滤，设为 None。

EXP_TYPE = "physical"
# 实验类型过滤，保留物理互作。None 表示不过滤。

# ── 泄漏审计阈值 ──────────────────────────────────────────────────────────────
AUDIT_MAX_OVERLAP_RATIO = 0.0
# leakage_report 中，若 train/test 蛋白/簇/物种重叠比例超过此值则标记 FAIL。
# 0.0 表示严格零泄漏。

# ─────────────────────────────────────────────────────────────────────────────


# ==============================================================================
# Section 1: ID 归一化与 accession 解析
# ==============================================================================

def normalize_id(x) -> Optional[str]:
    """
    归一化为 UniProt Accession，规则与 b__faa_record_id_sup.py 保持一致。

    Args:
        x: 原始字符串或 None。
    Returns:
        归一化后的 accession 字符串，或 None（无法解析时）。
    """
    if x is None:
        return None
    s = str(x).strip()
    if s in ("", "-", "nan", "NaN", "None"):
        return None
    if s.endswith(".npy"):
        s = s[:-4]
    # sp|P12345|GENE 格式
    m = re.match(r"[a-z]{2}\|([A-Z0-9]{1,10}(?:-\d+)?)\|", s)
    if m:
        return m.group(1)
    # 纯 accession 格式
    m = re.search(r"\b([A-Z][A-Z0-9]{4,9}(?:-\d+)?)\b", s)
    if m:
        return m.group(1)
    return None


def parse_accession_cell(cell) -> list[str]:
    """
    解析 BioGRID 单元格，支持 |;, 分隔的多值 accession。

    Args:
        cell: BioGRID 原始单元格值。
    Returns:
        解析出的 accession 列表。
    """
    if cell is None:
        return []
    parts = re.split(r"[|;,\s]+", str(cell).strip())
    return [acc for p in parts if (acc := normalize_id(p))]


def get_best_accession(sp_cell, tr_cell) -> Optional[str]:
    """
    Swiss-Prot 优先，其次 TrEMBL，优先使用人工校验程度更高的 accession。

    Args:
        sp_cell: Swiss-Prot accession 列值。
        tr_cell: TrEMBL accession 列值。
    Returns:
        最优 accession 字符串，或 None。
    """
    for cell in (sp_cell, tr_cell):
        accs = parse_accession_cell(cell)
        if accs:
            return accs[0]
    return None


# ==============================================================================
# Section 2: Embedding ID 加载
# ==============================================================================

def _load_from_index(index_txt: Path) -> set[str]:
    """从预建索引文件逐行读取 embedding IDs（快速路径）。"""
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
    """扫描 batch_*.npz 提取 accession（慢速路径，仅索引缺失时使用）。"""
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
    加载 embedding IDs。优先读索引文件，不存在则扫描 npz。

    Args:
        embdir:    batch_*.npz 所在目录（慢速路径）。
        index_txt: 预建索引文件路径（快速路径）。
    Returns:
        embedding accession 集合。
    """
    if index_txt.exists():
        return _load_from_index(index_txt)
    print(f"  [INFO] 索引文件不存在 ({index_txt.name})，自动扫描 npz")
    return _load_from_npz(embdir)


# ==============================================================================
# Section 3: BioGRID 正样本构建
# ==============================================================================

def load_biogrid_pos_pairs(
    biogrid_path: Path,
    emb_ids: set[str],
    organism_ids: Optional[set[int]],
    exp_type: Optional[str],
    for_species_split: bool = False,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """
    读取 BioGRID，提取正样本对与蛋白→物种映射。

    Args:
        biogrid_path:      BioGRID CSV.gz 路径。
        emb_ids:           有 embedding 的 accession 集合，用于过滤。
        organism_ids:      物种 ID 过滤集合，None 表示不过滤。
        exp_type:          实验类型过滤，None 表示不过滤。
        for_species_split: True 时跳过物种过滤，保留所有物种以支持 species split。
    Returns:
        (pos_pairs, protein_to_taxid)
        pos_pairs:        去重后的正样本对列表，每对 (a, b) 满足 a < b。
        protein_to_taxid: accession → Organism ID 映射（仅含正样本中出现的蛋白）。
    """
    print("[STEP 1] 读取 BioGRID CSV ...")
    df = pd.read_csv(biogrid_path, compression="gzip", low_memory=False)
    print(f"  原始记录: {len(df):,}")

    # species_disjoint 模式不在此处过滤物种，保留所有物种用于分组
    if organism_ids and not for_species_split:
        df = df[
            df["Organism ID Interactor A"].isin(organism_ids)
            & df["Organism ID Interactor B"].isin(organism_ids)
        ]
        print(f"  过滤物种后: {len(df):,}")

    if exp_type:
        df = df[
            df["Experimental System Type"].fillna("").str.lower()
            == exp_type.lower()
        ]
        print(f"  过滤实验类型后: {len(df):,}")

    print("[STEP 2] 提取 accession 正样本对与物种映射 ...")
    records: list[tuple[str, str]] = []
    protein_to_taxid: dict[str, int] = {}

    for _, row in df.iterrows():
        a = get_best_accession(
            row.get("SWISS-PROT Accessions Interactor A"),
            row.get("TREMBL Accessions Interactor A"),
        )
        b = get_best_accession(
            row.get("SWISS-PROT Accessions Interactor B"),
            row.get("TREMBL Accessions Interactor B"),
        )
        if not (a and b and a != b):
            continue

        # 记录物种映射（用于 species_disjoint）
        try:
            tax_a = int(row.get("Organism ID Interactor A", 0))
            tax_b = int(row.get("Organism ID Interactor B", 0))
            if tax_a:
                protein_to_taxid[a] = tax_a
            if tax_b:
                protein_to_taxid[b] = tax_b
        except (ValueError, TypeError):
            pass

        if a > b:
            a, b = b, a
        records.append((a, b))

    pos_pairs_raw = sorted(set(records))
    print(f"  正样本对（去重）: {len(pos_pairs_raw):,}")

    # 过滤双端均有 embedding 的正样本
    if emb_ids:
        before = len(pos_pairs_raw)
        pos_pairs = [
            (a, b) for a, b in pos_pairs_raw
            if a in emb_ids and b in emb_ids
        ]
        print(
            f"  embedding 过滤前: {before:,}  过滤后: {len(pos_pairs):,}"
            f"  保留率: {len(pos_pairs) / max(before, 1) * 100:.1f}%"
        )
    else:
        pos_pairs = pos_pairs_raw
        print("  [WARNING] embedding IDs 为空，跳过 embedding 过滤")

    return pos_pairs, protein_to_taxid


# ==============================================================================
# Section 4: 负样本采样（split 内独立进行）
# ==============================================================================

def sample_negatives_within_split(
    pos_pairs: list[tuple[str, str]],
    neg_ratio: float,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """
    在给定 split 的正样本范围内独立构建负样本。
    负样本仅从当前 split 涉及的蛋白中抽取，避免跨 split 采样污染。

    Args:
        pos_pairs: 当前 split 的正样本对列表。
        neg_ratio: 负样本倍率。
        rng:       已初始化的 random.Random 实例。
    Returns:
        负样本对列表。
    """
    pos_set = set(pos_pairs)
    proteins = sorted({p for a, b in pos_pairs for p in (a, b)})
    n_neg = int(len(pos_pairs) * neg_ratio)

    if len(proteins) < 2:
        print("  [WARNING] 蛋白数不足 2，无法构建负样本")
        return []

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

    if len(neg_pairs) < n_neg:
        print(
            f"  [WARNING] 负样本采样不足：目标 {n_neg}，实际 {len(neg_pairs)}"
        )

    return list(neg_pairs)


# ==============================================================================
# Section 5: 泄漏审计
# ==============================================================================

def audit_leakage(
    train_proteins: set[str],
    valid_proteins: set[str],
    test_proteins: set[str],
    train_groups: Optional[set] = None,
    valid_groups: Optional[set] = None,
    test_groups: Optional[set] = None,
    group_label: str = "cluster/species",
    max_overlap_ratio: float = AUDIT_MAX_OVERLAP_RATIO,
) -> dict:
    """
    对 train/valid/test 蛋白集合与可选分组做泄漏检查，返回 audit 报告 dict。

    Args:
        train_proteins: 训练集蛋白 accession 集合。
        valid_proteins: 验证集蛋白 accession 集合。
        test_proteins:  测试集蛋白 accession 集合。
        train_groups:   训练集分组 ID 集合（cluster ID 或 taxid），可选。
        valid_groups:   验证集分组 ID 集合，可选。
        test_groups:    测试集分组 ID 集合，可选。
        group_label:    分组名称，用于报告输出。
        max_overlap_ratio: 超过此比例视为 FAIL。
    Returns:
        leakage audit 报告字典。
    """
    report = {"checks": [], "overall": "PASS"}

    def _check(name, set_a, set_b, label_a, label_b):
        overlap = set_a & set_b
        ratio = len(overlap) / max(len(set_b), 1)
        status = "PASS" if ratio <= max_overlap_ratio else "FAIL"
        if status == "FAIL":
            report["overall"] = "FAIL"
        report["checks"].append({
            "check": name,
            f"{label_a}_size": len(set_a),
            f"{label_b}_size": len(set_b),
            "overlap": len(overlap),
            "overlap_ratio_in_b": round(ratio, 6),
            "status": status,
        })

    _check("protein: train ∩ valid", train_proteins, valid_proteins,
           "train", "valid")
    _check("protein: train ∩ test",  train_proteins, test_proteins,
           "train", "test")
    _check("protein: valid ∩ test",  valid_proteins, test_proteins,
           "valid", "test")

    if train_groups and test_groups:
        _check(
            f"{group_label}: train ∩ valid",
            train_groups, valid_groups or set(),
            "train", "valid",
        )
        _check(
            f"{group_label}: train ∩ test",
            train_groups, test_groups,
            "train", "test",
        )

    return report


# ==============================================================================
# Section 6: Split 实现
# ==============================================================================

def _assign_groups_to_splits(
    groups: list,
    train_ratio: float,
    valid_ratio: float,
    rng: random.Random,
) -> tuple[set, set, set]:
    """
    将分组列表（蛋白、cluster ID 或物种 ID）随机打乱后按比例分配。

    Args:
        groups:      分组 ID 列表。
        train_ratio: 训练集比例。
        valid_ratio: 验证集比例。
        rng:         random.Random 实例。
    Returns:
        (train_set, valid_set, test_set) 三个集合。
    """
    shuffled = list(groups)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(n * train_ratio))
    n_valid = max(1, int(n * valid_ratio))
    return (
        set(shuffled[:n_train]),
        set(shuffled[n_train: n_train + n_valid]),
        set(shuffled[n_train + n_valid:]),
    )


def split_random(
    pos_pairs: list[tuple[str, str]],
    rng: random.Random,
) -> dict[str, list[tuple[str, str]]]:
    """
    Random pair split：直接对正样本对随机打乱后按比例切分。
    仅用于 smoke test / 附录，不作为主 benchmark。

    Args:
        pos_pairs: 全量正样本对。
        rng:       random.Random 实例。
    Returns:
        {"train": [...], "valid": [...], "test": [...]} 正样本对字典。
    """
    shuffled = list(pos_pairs)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)
    return {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train: n_train + n_valid],
        "test":  shuffled[n_train + n_valid:],
    }


def split_protein_disjoint(
    pos_pairs: list[tuple[str, str]],
    rng: random.Random,
) -> dict[str, list[tuple[str, str]]]:
    """
    Protein-disjoint split：先在蛋白节点级别划分，再过滤 pair。
    测试集中的任何蛋白在训练集和验证集中不出现。
    跨 split 的 pair 直接丢弃。

    Args:
        pos_pairs: 全量正样本对。
        rng:       random.Random 实例。
    Returns:
        {"train": [...], "valid": [...], "test": [...]} 正样本对字典。
    """
    all_proteins = sorted({p for a, b in pos_pairs for p in (a, b)})
    train_p, valid_p, test_p = _assign_groups_to_splits(
        all_proteins, TRAIN_RATIO, VALID_RATIO, rng
    )

    result: dict[str, list] = {"train": [], "valid": [], "test": []}
    for a, b in pos_pairs:
        if a in train_p and b in train_p:
            result["train"].append((a, b))
        elif a in valid_p and b in valid_p:
            result["valid"].append((a, b))
        elif a in test_p and b in test_p:
            result["test"].append((a, b))
        # 跨 split 的 pair 丢弃

    print(
        f"  蛋白总数: {len(all_proteins):,}  "
        f"train: {len(train_p):,}  valid: {len(valid_p):,}  "
        f"test: {len(test_p):,}"
    )
    return result


def split_cluster_disjoint(
    pos_pairs: list[tuple[str, str]],
    cluster_tsv: Path,
    rng: random.Random,
) -> dict[str, list[tuple[str, str]]]:
    """
    Cluster-disjoint split：按 MMseqs2 cluster 划分，
    测试集 cluster 中的蛋白不与训练集 cluster 重叠。

    Args:
        pos_pairs:   全量正样本对。
        cluster_tsv: MMseqs2 easy-cluster 两列 TSV（representative\tmember）。
        rng:         random.Random 实例。
    Returns:
        {"train": [...], "valid": [...], "test": [...]} 正样本对字典。

    【待验证】cluster_tsv 格式需为无表头两列 TSV，第一列为 cluster 代表序列 ID。
    """
    if not cluster_tsv.exists():
        print(
            f"  [ERROR] cluster_disjoint 模式需提供 CLUSTER_TSV: {cluster_tsv}",
            file=sys.stderr,
        )
        sys.exit(1)

    # 读取 cluster 映射：蛋白 → cluster_id（以 representative 为 cluster ID）
    protein_to_cluster: dict[str, str] = {}
    with open(cluster_tsv, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep, member = parts[0], parts[1]
                protein_to_cluster[normalize_id(member) or member] = (
                    normalize_id(rep) or rep
                )

    # 仅保留正样本中出现的蛋白
    all_clusters = sorted({
        protein_to_cluster[p]
        for a, b in pos_pairs
        for p in (a, b)
        if p in protein_to_cluster
    })

    if not all_clusters:
        print("  [ERROR] cluster TSV 与正样本无交集，请检查 accession 格式",
              file=sys.stderr)
        sys.exit(1)

    train_c, valid_c, test_c = _assign_groups_to_splits(
        all_clusters, TRAIN_RATIO, VALID_RATIO, rng
    )

    result: dict[str, list] = {"train": [], "valid": [], "test": []}
    for a, b in pos_pairs:
        ca = protein_to_cluster.get(a)
        cb = protein_to_cluster.get(b)
        if ca is None or cb is None:
            continue  # 没有 cluster 信息的蛋白对丢弃
        if ca in train_c and cb in train_c:
            result["train"].append((a, b))
        elif ca in valid_c and cb in valid_c:
            result["valid"].append((a, b))
        elif ca in test_c and cb in test_c:
            result["test"].append((a, b))

    print(
        f"  Cluster 总数: {len(all_clusters):,}  "
        f"train: {len(train_c):,}  valid: {len(valid_c):,}  "
        f"test: {len(test_c):,}"
    )
    return result, protein_to_cluster


def split_species_disjoint(
    pos_pairs: list[tuple[str, str]],
    protein_to_taxid: dict[str, int],
    rng: random.Random,
) -> dict[str, list[tuple[str, str]]]:
    """
    Species-disjoint split：按物种 / taxa 划分，
    测试集物种在训练集中完全不出现。

    Args:
        pos_pairs:        全量正样本对。
        protein_to_taxid: accession → Organism ID 映射。
        rng:              random.Random 实例。
    Returns:
        {"train": [...], "valid": [...], "test": [...]} 正样本对字典。
    """
    all_species = sorted({
        protein_to_taxid[p]
        for a, b in pos_pairs
        for p in (a, b)
        if p in protein_to_taxid
    })

    if len(all_species) < 3:
        print(
            f"  [WARNING] 物种数量不足（{len(all_species)} 个），"
            "species_disjoint 效果有限",
            file=sys.stderr,
        )

    train_s, valid_s, test_s = _assign_groups_to_splits(
        all_species, TRAIN_RATIO, VALID_RATIO, rng
    )

    result: dict[str, list] = {"train": [], "valid": [], "test": []}
    for a, b in pos_pairs:
        ta = protein_to_taxid.get(a)
        tb = protein_to_taxid.get(b)
        if ta is None or tb is None:
            continue
        if ta in train_s and tb in train_s:
            result["train"].append((a, b))
        elif ta in valid_s and tb in valid_s:
            result["valid"].append((a, b))
        elif ta in test_s and tb in test_s:
            result["test"].append((a, b))

    print(
        f"  物种总数: {len(all_species):,}  "
        f"train: {len(train_s):,}  valid: {len(valid_s):,}  "
        f"test: {len(test_s):,}"
    )
    return result


# ==============================================================================
# Section 7: CSV 输出与泄漏报告
# ==============================================================================

def save_split_csvs(
    pos_split: dict[str, list[tuple[str, str]]],
    outdir: Path,
    rng: random.Random,
    split_mode: str,
    split_version: str,
    protein_to_cluster: Optional[dict[str, str]] = None,
    protein_to_taxid: Optional[dict[str, int]] = None,
) -> None:
    """
    对每个 split（train/valid/test）在 split 内独立采样负样本，
    合并后输出 CSV，并生成 leakage_report.json。

    Args:
        pos_split:          各 split 的正样本对字典。
        outdir:             输出目录（自动创建）。
        rng:                random.Random 实例。
        split_mode:         当前 split 模式名称，写入统计文件。
        split_version:      split 版本号，写入统计文件。
        protein_to_cluster: accession → cluster_id 映射（cluster 模式使用）。
        protein_to_taxid:   accession → taxid 映射（species 模式使用）。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stats = {
        "split_mode": split_mode,
        "split_version": split_version,
        "neg_ratio": NEG_RATIO,
        "seed": SEED,
        "splits": {},
    }
    all_split_proteins: dict[str, set[str]] = {}
    all_split_groups: dict[str, set] = {}

    for split_name, pos_pairs in pos_split.items():
        if not pos_pairs:
            print(f"  [WARNING] {split_name} 正样本为空，跳过")
            continue

        # split 内独立负采样（不跨 split）
        neg_pairs = sample_negatives_within_split(pos_pairs, NEG_RATIO, rng)

        all_rows = (
            [(a, b, 1) for a, b in pos_pairs]
            + [(a, b, 0) for a, b in neg_pairs]
        )
        rng.shuffle(all_rows)

        df_out = pd.DataFrame(all_rows, columns=["protein_a", "protein_b", "label"])
        out_csv = outdir / f"{split_name}.csv"
        df_out.to_csv(out_csv, index=False)

        n_pos = int(df_out["label"].sum())
        stats["splits"][split_name] = {
            "total": len(df_out),
            "positive": n_pos,
            "negative": len(df_out) - n_pos,
            "unique_proteins": len(
                {p for a, b in pos_pairs for p in (a, b)}
            ),
        }
        print(
            f"  [{split_name}] {len(df_out):,} pairs "
            f"(pos={n_pos:,}, neg={len(df_out) - n_pos:,}) -> {out_csv}"
        )

        # 收集蛋白集合，供泄漏审计使用
        all_split_proteins[split_name] = {
            p for a, b in pos_pairs for p in (a, b)
        }

        # 收集分组集合
        if protein_to_cluster:
            all_split_groups[split_name] = {
                protein_to_cluster[p]
                for p in all_split_proteins[split_name]
                if p in protein_to_cluster
            }
        elif protein_to_taxid:
            all_split_groups[split_name] = {
                protein_to_taxid[p]
                for p in all_split_proteins[split_name]
                if p in protein_to_taxid
            }

    # 泄漏审计
    group_label = (
        "cluster" if protein_to_cluster
        else ("species" if protein_to_taxid else "")
    )
    leakage = audit_leakage(
        train_proteins=all_split_proteins.get("train", set()),
        valid_proteins=all_split_proteins.get("valid", set()),
        test_proteins=all_split_proteins.get("test", set()),
        train_groups=all_split_groups.get("train"),
        valid_groups=all_split_groups.get("valid"),
        test_groups=all_split_groups.get("test"),
        group_label=group_label,
    )
    stats["leakage_audit"] = leakage

    # 写出统计文件
    stats_path = outdir / "stats.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  统计与泄漏报告 -> {stats_path}")
    print(f"  泄漏审计结果: {leakage['overall']}")

    # 如果审计失败，在终端醒目提示
    if leakage["overall"] == "FAIL":
        print(
            "  [ALERT] 泄漏审计 FAIL！请检查 stats.json 中 leakage_audit 字段。",
            file=sys.stderr,
        )


# ==============================================================================
# Section 8: 主流程
# ==============================================================================

def build_pairs() -> None:
    """主流程入口：按配置区参数执行完整数据构建流水线。"""
    rng = random.Random(SEED)
    for_species = (SPLIT_MODE == "species_disjoint")

    # STEP 1–2: 读取 BioGRID 并提取正样本对
    emb_ids = load_embedding_ids(EMBDIR, EMB_INDEX_TXT)
    pos_pairs, protein_to_taxid = load_biogrid_pos_pairs(
        biogrid_path=BIOGRID_PATH,
        emb_ids=emb_ids,
        organism_ids=ORGANISM_IDS,
        exp_type=EXP_TYPE,
        for_species_split=for_species,
    )

    if not pos_pairs:
        print("[ERROR] 正样本为空，请检查数据路径与过滤配置。", file=sys.stderr)
        sys.exit(1)

    # STEP 3: 按 split 模式分配正样本
    print(f"[STEP 3] 执行 split 模式: {SPLIT_MODE} ...")
    protein_to_cluster: Optional[dict[str, str]] = None

    if SPLIT_MODE == "random":
        pos_split = split_random(pos_pairs, rng)

    elif SPLIT_MODE == "protein_disjoint":
        pos_split = split_protein_disjoint(pos_pairs, rng)

    elif SPLIT_MODE == "cluster_disjoint":
        pos_split, protein_to_cluster = split_cluster_disjoint(
            pos_pairs, CLUSTER_TSV, rng
        )

    elif SPLIT_MODE == "species_disjoint":
        pos_split = split_species_disjoint(pos_pairs, protein_to_taxid, rng)

    else:
        print(
            f"[ERROR] 未知 SPLIT_MODE: {SPLIT_MODE}。"
            "可选: random / protein_disjoint / cluster_disjoint / species_disjoint",
            file=sys.stderr,
        )
        sys.exit(1)

    # 输出各 split 正样本数量
    for name, pairs in pos_split.items():
        print(f"  {name} 正样本: {len(pairs):,}")

    # STEP 4: split 内独立负采样 + 写出 CSV + 泄漏审计
    print(f"[STEP 4] 输出 CSV 与泄漏审计 -> {OUTDIR / SPLIT_MODE}")
    save_split_csvs(
        pos_split=pos_split,
        outdir=OUTDIR / SPLIT_MODE,
        rng=rng,
        split_mode=SPLIT_MODE,
        split_version=SPLIT_VERSION,
        protein_to_cluster=protein_to_cluster,
        protein_to_taxid=protein_to_taxid if for_species else None,
    )

    print("[DONE]")


if __name__ == "__main__":
    build_pairs()