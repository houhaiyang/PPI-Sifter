#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Set, Tuple

import pandas as pd

from ppisifter.utils import load_config, save_json, set_seed


BIOGRID_CANDIDATE_A = [
    'Official Symbol Interactor A',
    'Alt IDs Interactor A',
    'SWISS-PROT Accessions Interactor A',
    'BioGRID ID Interactor A',
]
BIOGRID_CANDIDATE_B = [
    'Official Symbol Interactor B',
    'Alt IDs Interactor B',
    'SWISS-PROT Accessions Interactor B',
    'BioGRID ID Interactor B',
]


def choose_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f'Cannot find suitable BioGRID column in: {candidates}')


def normalize_id(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == 'nan' or text == '-':
        return ''
    return text.split('|')[0].split(';')[0].split(',')[0].strip()


def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


def sample_negative_pairs(proteins: List[str], positive_set: Set[Tuple[str, str]], target_count: int, seed: int) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    negatives: Set[Tuple[str, str]] = set()
    if len(proteins) < 2:
        return []
    max_trials = max(target_count * 50, 1000)
    trials = 0
    while len(negatives) < target_count and trials < max_trials:
        a, b = rng.sample(proteins, 2)
        pair = canonical_pair(a, b)
        if pair in positive_set or pair in negatives or a == b:
            trials += 1
            continue
        negatives.add(pair)
        trials += 1
    return sorted(negatives)


def main():
    parser = argparse.ArgumentParser(description='Build positive and 1:10 negative PPI pairs.')
    parser.add_argument('--config', required=True, help='Path to YAML config.')
    parser.add_argument('--biogrid_csv', required=True, help='BioGRID interaction CSV path.')
    parser.add_argument('--output_dir', required=True, help='Directory to save pair files.')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config['project'].get('seed', 42)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = config['data'].get('embedding_suffix', '.residue_emb.npy')
    embeddings_dir = Path(config['data']['embeddings_dir'])
    available = {p.name[: -len(suffix)] for p in embeddings_dir.glob(f'*{suffix}')}

    df = pd.read_csv(args.biogrid_csv, low_memory=False)
    col_a = choose_column(df, BIOGRID_CANDIDATE_A)
    col_b = choose_column(df, BIOGRID_CANDIDATE_B)
    df = df[[col_a, col_b]].copy()
    df.columns = ['protein_a', 'protein_b']
    df['protein_a'] = df['protein_a'].map(normalize_id)
    df['protein_b'] = df['protein_b'].map(normalize_id)
    df = df[(df['protein_a'] != '') & (df['protein_b'] != '')]
    df = df[df['protein_a'] != df['protein_b']]
    df = df[df['protein_a'].isin(available) & df['protein_b'].isin(available)]

    positive_pairs = sorted({canonical_pair(a, b) for a, b in df[['protein_a', 'protein_b']].itertuples(index=False, name=None)})
    proteins = sorted(set([x for pair in positive_pairs for x in pair]))
    negative_ratio = int(config['data'].get('negative_ratio', 10))
    negative_pairs = sample_negative_pairs(proteins, set(positive_pairs), len(positive_pairs) * negative_ratio, int(config['project'].get('seed', 42)))

    pos_df = pd.DataFrame(positive_pairs, columns=['protein_a', 'protein_b'])
    pos_df['label'] = 1
    neg_df = pd.DataFrame(negative_pairs, columns=['protein_a', 'protein_b'])
    neg_df['label'] = 0
    all_df = pd.concat([pos_df, neg_df], ignore_index=True)
    all_df['pair_id'] = all_df['protein_a'] + '__' + all_df['protein_b']
    all_df = all_df[['pair_id', 'protein_a', 'protein_b', 'label']].sample(frac=1.0, random_state=int(config['project'].get('seed', 42))).reset_index(drop=True)

    all_df.to_csv(output_dir / 'all_pairs.csv', index=False)
    pos_df.to_csv(output_dir / 'positive_pairs.csv', index=False)
    neg_df.to_csv(output_dir / 'negative_pairs.csv', index=False)
    report = {
        'num_available_embeddings': len(available),
        'num_positive_pairs': int(len(pos_df)),
        'num_negative_pairs': int(len(neg_df)),
        'negative_ratio': negative_ratio,
        'num_unique_proteins': len(proteins),
    }
    save_json(report, output_dir / 'build_report.json')


if __name__ == '__main__':
    main()
