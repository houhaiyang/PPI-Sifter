#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description='Visualize attention matrix as heatmap.')
    parser.add_argument('--attention_csv', required=True, help='CSV file containing attention matrix.')
    parser.add_argument('--output_png', required=True, help='Output PNG path.')
    parser.add_argument('--dpi', type=int, default=200, help='Figure DPI.')
    args = parser.parse_args()

    df = pd.read_csv(args.attention_csv)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, cmap='viridis')
    plt.title('Residue-Residue Attention Heatmap')
    plt.xlabel('Residues of protein B')
    plt.ylabel('Residues of protein A')
    output_path = Path(args.output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi)
    plt.close()


if __name__ == '__main__':
    main()
