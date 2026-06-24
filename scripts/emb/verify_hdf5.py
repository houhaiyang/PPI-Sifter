"""
脚本: verify_hdf5.py
功能: 验证 HDF5 嵌入文件完整性，输出统计信息
运行: python scripts/emb/verify_hdf5.py --h5 data/BIOGRID/embeddings/embeddings.h5
"""

import argparse
import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True, help="HDF5 文件路径")
    parser.add_argument("--sample_n", type=int, default=5, help="抽样展示数量")
    return parser.parse_args()


def main():
    args = parse_args()
    with h5py.File(args.h5, "r") as f:
        keys = list(f.keys())
        print(f"蛋白总数: {len(keys)}")
        print(f"抽样展示前 {args.sample_n} 条:")
        for k in keys[:args.sample_n]:
            arr = f[k][:]
            print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}, "
                  f"min={arr.min():.4f}, max={arr.max():.4f}")
        # 统计长度分布
        lengths = [f[k].shape[0] for k in keys]
        print(f"\n序列长度统计: min={min(lengths)}, max={max(lengths)}, "
              f"mean={np.mean(lengths):.1f}, median={np.median(lengths):.1f}")


if __name__ == "__main__":
    main()
