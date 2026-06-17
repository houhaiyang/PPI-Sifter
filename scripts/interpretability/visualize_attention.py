import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix_csv', required=True)
    parser.add_argument('--output_png', required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.matrix_csv, header=None)
    Path(args.output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(df.values, cmap='Purples')
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=200)
    plt.close()

if __name__ == '__main__':
    main()
