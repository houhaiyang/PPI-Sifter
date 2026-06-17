"""
脚本: split_dataset.py
功能: 实现 protein-disjoint / random pair 两种数据集划分协议
依赖: pandas, scikit-learn
运行: python scripts/emb/split_dataset.py \
          --pairs  data/BIOGRID/pairs/all_pairs.csv \
          --out    data/BIOGRID/splits \
          --mode   protein_disjoint \
          --seed   42
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


VALID_MODES = ["random", "protein_disjoint"]


def parse_args():
    parser = argparse.ArgumentParser(description="数据集划分脚本")
    parser.add_argument("--pairs",  required=True, help="pairs CSV 路径")
    parser.add_argument("--out",    required=True, help="划分文件输出目录")
    parser.add_argument("--mode",   default="protein_disjoint",
                        choices=VALID_MODES)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--seed",   type=int, default=42)
    return parser.parse_args()


def random_split(df, train_ratio, valid_ratio, seed):
    """随机按行划分（仅用于 debug/附录）。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_train = int(len(df) * train_ratio)
    n_valid = int(len(df) * valid_ratio)
    train_idx = idx[:n_train]
    valid_idx = idx[n_train:n_train + n_valid]
    test_idx  = idx[n_train + n_valid:]
    return df.iloc[train_idx], df.iloc[valid_idx], df.iloc[test_idx]


def protein_disjoint_split(df, train_ratio, valid_ratio, seed):
    """
    Protein-disjoint 划分。
    先将所有唯一蛋白随机分为 train/valid/test 集合，
    再只保留两端蛋白均属于同一 split 的 pair。
    """
    all_proteins = list(
        set(df["protein_a"].tolist()) | set(df["protein_b"].tolist())
    )
    np.random.seed(seed)
    np.random.shuffle(all_proteins)

    n = len(all_proteins)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_proteins = set(all_proteins[:n_train])
    valid_proteins = set(all_proteins[n_train:n_train + n_valid])
    test_proteins  = set(all_proteins[n_train + n_valid:])

    def filter_pairs(protein_set):
        mask = (
            df["protein_a"].isin(protein_set) &
            df["protein_b"].isin(protein_set)
        )
        return df[mask].reset_index(drop=True)

    train_df = filter_pairs(train_proteins)
    valid_df = filter_pairs(valid_proteins)
    test_df  = filter_pairs(test_proteins)
    return train_df, valid_df, test_df


def run_leakage_check(train_df, valid_df, test_df):
    """基础泄漏检查：验证 split 间无蛋白重叠（protein-disjoint 模式）。"""
    def get_proteins(df):
        return set(df["protein_a"]) | set(df["protein_b"])

    train_p = get_proteins(train_df)
    valid_p = get_proteins(valid_df)
    test_p  = get_proteins(test_df)

    tv_overlap = train_p & valid_p
    tt_overlap = train_p & test_p
    vt_overlap = valid_p & test_p

    print(f"[泄漏检查] train-valid 蛋白重叠: {len(tv_overlap)}")
    print(f"[泄漏检查] train-test  蛋白重叠: {len(tt_overlap)}")
    print(f"[泄漏检查] valid-test  蛋白重叠: {len(vt_overlap)}")
    if tt_overlap:
        print("[警告] train 与 test 存在蛋白重叠，请检查划分逻辑！")


def main():
    args = parse_args()
    df = pd.read_csv(args.pairs)
    required_cols = {"protein_a", "protein_b", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    print(f"总样本数: {len(df)}, 正样本: {df['label'].sum()}, "
          f"负样本: {(df['label'] == 0).sum()}")

    if args.mode == "random":
        train_df, valid_df, test_df = random_split(
            df, args.train_ratio, args.valid_ratio, args.seed
        )
    elif args.mode == "protein_disjoint":
        train_df, valid_df, test_df = protein_disjoint_split(
            df, args.train_ratio, args.valid_ratio, args.seed
        )
    else:
        raise ValueError(f"未知 mode: {args.mode}")

    run_leakage_check(train_df, valid_df, test_df)

    os.makedirs(args.out, exist_ok=True)
    train_df.to_csv(os.path.join(args.out, "train.csv"), index=False)
    valid_df.to_csv(os.path.join(args.out, "valid.csv"), index=False)
    test_df.to_csv(os.path.join(args.out,  "test.csv"),  index=False)

    print(f"\n划分完成 [{args.mode}]:")
    print(f"  train: {len(train_df)}")
    print(f"  valid: {len(valid_df)}")
    print(f"  test:  {len(test_df)}")
    print(f"输出目录: {args.out}")


if __name__ == "__main__":
    main()
