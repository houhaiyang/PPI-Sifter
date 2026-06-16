#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
b__faa_record_id_sup.py
=======================
辅助脚本：扫描 batch_*.npz 建立 embedding ID 索引，
与 data/BIOGRID/all.clean.faa 中的序列 ID 进行核验比对，
**只保留两者的交集**，输出：
  data/BIOGRID/faa_chunks/BIOGRID-ALL-4.4.240.uniprot.embedding.uniprot_ids.txt
  data/BIOGRID/faa_chunks/BIOGRID-ALL-4.4.240.uniprot.embedding_index.json

设计特点
--------
* 节省内存：zipfile.namelist() 只读 key 列表，不加载 array。
* all.clean.faa 用流式逐行解析（只读 >header 行），无需 Biopython。
* normalize_id 逻辑与 c__build_pair_csv.py 完全一致，确保 ID 对齐。
* 容错：损坏 npz 记录警告并跳过；faa 不存在时打印警告继续输出空交集。

目录结构
--------
  data/BIOGRID/
  ├── all.clean.faa                <- FAA_PATH（由 b__faa_record_id.py 生成）
  ├── embeddings/residue/          <- EMBDIR（batch_*.npz）
  └── faa_chunks/                  <- 输出目录
"""

import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Optional

# ── 路径配置 ──────────────────────────────────────────────────────────────────
EMBDIR   = Path("data/BIOGRID/embeddings/residue")
FAA_PATH = Path("data/BIOGRID/all.clean.faa")
OUT_DIR  = Path("data/BIOGRID/faa_chunks")
OUT_TXT  = OUT_DIR / "BIOGRID-ALL-4.4.240.uniprot.embedding.uniprot_ids.txt"
OUT_JSON = OUT_DIR / "BIOGRID-ALL-4.4.240.uniprot.embedding_index.json"
# ─────────────────────────────────────────────────────────────────────────────


def normalize_id(x) -> Optional[str]:
    """
    归一化为 UniProt Accession（与 c__build_pair_csv.py 完全一致）。

    示例：
      'sp|P12345|ABCD_HUMAN.npy' -> 'P12345'
      'P12345.npy'               -> 'P12345'
      '>sp|P12345|ABCD_HUMAN'    -> 'P12345'  (faa header)
      'P12345'                   -> 'P12345'
    """
    if x is None:
        return None
    s = str(x).strip()
    if s in ("", "-", "nan", "NaN", "None"):
        return None
    # 去掉 FASTA > 前缀与 .npy 后缀
    s = s.lstrip(">").strip()
    if s.endswith(".npy"):
        s = s[:-4]
    # Swiss-Prot / TrEMBL: sp|ACC|ENTRY 或 tr|ACC|ENTRY
    m = re.match(r"[a-z]{2}\|([A-Z0-9]{1,10}(?:-\d+)?)\|", s)
    if m:
        return m.group(1)
    # 裸 accession（6-10 位大写字母数字，可含 -isoform 后缀）
    m = re.search(r"\b([A-Z][A-Z0-9]{4,9}(?:-\d+)?)\b", s)
    if m:
        return m.group(1)
    return None


def scan_embedding_ids(embdir: Path) -> tuple[dict[str, str], list[str]]:
    """
    扫描所有 batch_*.npz，提取 accession。
    仅读 .namelist()，不加载 array 数据。

    Returns
    -------
    id_to_npz : dict  accession -> 首次出现的 npz 文件名（调试溯源）
    errors    : list  读取失败的文件路径
    """
    id_to_npz: dict[str, str] = {}
    errors: list[str] = []

    batch_files = sorted(embdir.glob("batch_*.npz"))
    total = len(batch_files)
    if total == 0:
        print(f"[WARNING] 在 {embdir} 未找到任何 batch_*.npz！", file=sys.stderr)
        return id_to_npz, errors

    print(f"[INFO] 发现 {total} 个 batch_*.npz，开始扫描 key 列表...")
    for i, bf in enumerate(batch_files, 1):
        try:
            with zipfile.ZipFile(bf, "r") as zf:
                for name in zf.namelist():
                    if name.endswith(".npy"):
                        acc = normalize_id(name[:-4])
                        if acc and acc not in id_to_npz:
                            id_to_npz[acc] = bf.name
        except Exception as e:
            print(f"[WARNING] 读取失败 {bf.name}: {e}", file=sys.stderr)
            errors.append(str(bf))

        if i % 50 == 0 or i == total:
            print(f"  已处理 {i}/{total} 个 npz，累计 embedding IDs: {len(id_to_npz):,}")

    return id_to_npz, errors


def load_faa_ids(faa_path: Path) -> set[str]:
    """
    流式解析 all.clean.faa，只读 '>' 开头的 header 行，提取 accession。
    all.clean.faa 由 b__faa_record_id.py 生成，record.id 已是归一化 accession，
    header 格式为 '>P12345' 或 '>sp|P12345|...'。
    """
    if not faa_path.exists():
        print(f"[WARNING] FAA 文件不存在：{faa_path}", file=sys.stderr)
        return set()

    faa_ids: set[str] = set()
    with open(faa_path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            # header 第一个空格前为 record.id
            header = line[1:].split()[0] if line[1:].strip() else ""
            acc = normalize_id(header)
            if acc:
                faa_ids.add(acc)

    print(f"[INFO] all.clean.faa 解析完毕，共 {len(faa_ids):,} 个序列 ID")
    return faa_ids


def compare_and_report(emb_ids: set[str],
                       faa_ids: set[str],
                       intersection: set[str]) -> None:
    """打印三方统计比对报告。"""
    only_in_emb = emb_ids - faa_ids
    only_in_faa = faa_ids - emb_ids

    sep = "=" * 62
    print(f"\n{sep}")
    print("  Embedding IDs  vs.  all.clean.faa  比对报告")
    print(sep)
    print(f"  Embedding 中 ID 总数  : {len(emb_ids):>10,}")
    print(f"  all.clean.faa ID 总数 : {len(faa_ids):>10,}")
    print(f"  交集（输出保留）      : {len(intersection):>10,}")
    print(f"  仅存在于 Embedding    : {len(only_in_emb):>10,}")
    print(f"  仅存在于 FAA          : {len(only_in_faa):>10,}")

    if faa_ids:
        cov_emb = len(intersection) / len(faa_ids) * 100
        print(f"  FAA 覆盖率（emb/faa） : {cov_emb:>9.2f}%")
    if emb_ids:
        cov_faa = len(intersection) / len(emb_ids) * 100
        print(f"  Emb 利用率（∩/emb）   : {cov_faa:>9.2f}%")

    SAMPLE_N = 10
    if only_in_faa:
        print(f"\n  [FAA 中有但 Embedding 缺失的样本（前{SAMPLE_N}）]")
        for s in sorted(only_in_faa)[:SAMPLE_N]:
            print(f"    {s}")
    if only_in_emb:
        print(f"\n  [Embedding 中多余（不在 FAA）的样本（前{SAMPLE_N}）]")
        for s in sorted(only_in_emb)[:SAMPLE_N]:
            print(f"    {s}")

    print(f"{sep}\n")


def main():
    # STEP 1: 扫描 npz
    print("[STEP 1] 扫描 batch_*.npz，建立 embedding ID 索引...")
    id_to_npz, errors = scan_embedding_ids(EMBDIR)
    emb_ids = set(id_to_npz.keys())
    print(f"[INFO] 共获取有效 embedding IDs: {len(emb_ids):,}")
    if errors:
        print(f"[WARNING] {len(errors)} 个文件读取失败，请检查日志")

    # STEP 2: 解析 all.clean.faa
    print(f"\n[STEP 2] 解析 {FAA_PATH} 提取序列 IDs...")
    faa_ids = load_faa_ids(FAA_PATH)

    # STEP 3: 取交集
    print("\n[STEP 3] 计算交集，生成比对报告...")
    intersection = emb_ids & faa_ids
    compare_and_report(emb_ids, faa_ids, intersection)

    # STEP 4: 输出交集 txt（每行一个 accession）
    print("[STEP 4] 输出交集 ID 列表...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sorted_ids = sorted(intersection)
    OUT_TXT.write_text("\n".join(sorted_ids) + "\n", encoding="utf-8")
    print(f"[INFO] 已写入 {len(sorted_ids):,} 个交集 ID -> {OUT_TXT}")

    # STEP 5: 输出 JSON 索引（仅保留交集条目）
    print("[STEP 5] 输出 JSON 索引（accession -> npz 来源，仅交集）...")
    filtered_index = {k: v for k, v in id_to_npz.items() if k in intersection}
    OUT_JSON.write_text(
        json.dumps(filtered_index, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")),
        encoding="utf-8"
    )
    print(f"[INFO] JSON 索引已写入 {len(filtered_index):,} 条 -> {OUT_JSON}")
    print("[DONE]")


if __name__ == "__main__":
    main()
