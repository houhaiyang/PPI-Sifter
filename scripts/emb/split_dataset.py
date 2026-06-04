#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Set

import pandas as pd

from ppisifter.utils import load_config, save_json, set_seed


def protein_disjoint_split(df: pd.DataFrame, train_ratio: float, valid_ratio: float, seed: int):
    proteins = pd.Index(sorted(set(df['protein_a']).union(set(df['protein_b'])))).to_series().sample(frac=1.0, random_state=seed).tolist()
    n = len(proteins)
    train_proteins = set(proteins[: int(n * train_ratio)])
    valid_proteins = set(proteins[int(n * train_ratio): int(n * (train_ratio + valid_ratio))])
    test_proteins = set(proteins[int(n * (train_ratio + valid_ratio)):])

    train_df = df[df['protein_a'].isin(train_proteins) & df['protein_b'].isin(train_proteins)].copy()
    valid_df = df[df['protein_a'].isin(valid_proteins) & df['protein_b'].isin(valid_proteins)].copy()
    test_df = df[df['protein_a'].isin(test_proteins) & df['protein_b'].isin(test_proteins)].copy()
    return train_df, valid_df, test_df


def random_split(df: pd.DataFrame, train_ratio: float, valid_ratio: float, seed: int):
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    return shuffled.iloc[:train_end].copy(), shuffled.iloc[train_end:valid_end].copy(), shuffled.iloc[valid_end:].copy()


def main():
    parser = argparse.ArgumentParser(description='Split pair dataset into train/valid/test CSV files.')
    parser.add_argument('--config', required=True, help='Path to YAML config.')
    parser.add_argument('--pairs_csv', required=True, help='Path to all_pairs.csv.')
    parser.add_argument('--output_dir', required=True, help='Output directory.')
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config['project'].get('seed', 42))
    set_seed(seed)
    strategy = config['data'].get('split_strategy', 'protein_disjoint')
    train_ratio = float(config['data'].get('train_ratio', 0.8))
    valid_ratio = float(config['data'].get('valid_ratio', 0.1))

    df = pd.read_csv(args.pairs_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if strategy == 'protein_disjoint':
        train_df, valid_df, test_df = protein_disjoint_split(df, train_ratio, valid_ratio, seed)
    else:
        train_df, valid_df, test_df = random_split(df, train_ratio, valid_ratio, seed)

    train_df.to_csv(output_dir / 'train.csv', index=False)
    valid_df.to_csv(output_dir / 'valid.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    save_json({
        'strategy': strategy,
        'train_size': int(len(train_df)),
        'valid_size': int(len(valid_df)),
        'test_size': int(len(test_df)),
    }, output_dir / 'split_report.json')


if __name__ == '__main__':
    main()
