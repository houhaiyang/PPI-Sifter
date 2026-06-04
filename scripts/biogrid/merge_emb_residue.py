#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm


def merge_residue_embeddings(input_faa: str, embeddings_dir: str, output_file: str):
    """
    合并 residue-level embedding 文件为一个 .pt 文件。

    输入:
        input_faa: 原始 FASTA 文件
        embeddings_dir: 包含单条蛋白 residue embedding 的目录
        output_file: 输出 .pt 文件路径

    约定:
        每条蛋白对应一个文件:
            {seq_id}.residue_emb.npy
        每个文件 shape 为:
            [L, D]
        其中 L 为该蛋白残基长度，D 为 embedding 维度
    """
    seq_records = list(SeqIO.parse(input_faa, "fasta"))
    seq_ids = [record.id for record in seq_records]

    merged = {}
    missing = []
    bad_shape = []

    found_files = 0

    for seq_id in tqdm(seq_ids, desc="Merging residue embeddings", unit="seq"):
        file_path = Path(embeddings_dir) / f"{seq_id}.residue_emb.npy"

        if not file_path.exists():
            missing.append(seq_id)
            continue

        try:
            emb = np.load(file_path)

            if emb.ndim != 2:
                bad_shape.append((seq_id, tuple(emb.shape)))
                continue

            merged[seq_id] = {
                "embedding": torch.from_numpy(emb).float(),
                "seq_len": emb.shape[0],
                "emb_dim": emb.shape[1],
            }
            found_files += 1

        except Exception as e:
            print(f"\nError reading {file_path}: {e}")

    payload = {
        "data": merged,
        "num_sequences_in_fasta": len(seq_ids),
        "num_embeddings_found": found_files,
        "missing_seq_ids": missing,
        "bad_shape_seq_ids": bad_shape,
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    print(f"Total sequences in FASTA: {len(seq_ids)}")
    print(f"Found residue embedding files: {found_files}")
    print(f"Missing files: {len(missing)}")
    print(f"Bad shape files: {len(bad_shape)}")
    print(f"Saved merged residue embeddings to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge individual residue embedding .npy files into one .pt file"
    )
    parser.add_argument("--input", required=True, help="Original input FASTA file (.faa)")
    parser.add_argument("--embeddings_dir", required=True, help="Directory containing .residue_emb.npy files")
    parser.add_argument("--output", required=True, help="Output merged .pt file path")

    args = parser.parse_args()

    merge_residue_embeddings(
        input_faa=args.input,
        embeddings_dir=args.embeddings_dir,
        output_file=args.output,
    )