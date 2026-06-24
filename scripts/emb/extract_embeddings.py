#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO

from ppisifter.utils import load_config, ensure_dir


def load_model_stub(device: torch.device):
    raise RuntimeError(
        '本脚本预留为 ESM-C 嵌入抽取入口。请根据本地 ESM-C 安装方式补充模型加载逻辑，或直接使用已存在的 embeddings 目录。'
    )


def sequence_to_embedding(model, device: torch.device, sequence: str) -> np.ndarray:
    raise RuntimeError('请根据本地 ESM-C API 实现 residue embedding 抽取。')


def main():
    parser = argparse.ArgumentParser(description='Extract residue embeddings into .residue_emb.npy files.')
    parser.add_argument('--config', required=True, help='Path to YAML config.')
    parser.add_argument('--input_fasta', required=True, help='Input FASTA file.')
    parser.add_argument('--output_dir', required=True, help='Output embedding directory.')
    parser.add_argument('--device', default=None, help='Override device from config.')
    args = parser.parse_args()

    config = load_config(args.config)
    device_name = args.device or config['project'].get('device', 'cuda')
    device = torch.device(device_name if device_name == 'cuda' and torch.cuda.is_available() else 'cpu')
    output_dir = ensure_dir(args.output_dir)

    model = load_model_stub(device)
    suffix = config['data'].get('embedding_suffix', '.residue_emb.npy')

    for record in SeqIO.parse(args.input_fasta, 'fasta'):
        protein_id = record.id
        out_path = output_dir / f'{protein_id}{suffix}'
        if out_path.exists():
            continue
        array = sequence_to_embedding(model, device, str(record.seq))
        np.save(out_path, array)


if __name__ == '__main__':
    main()
