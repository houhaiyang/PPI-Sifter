"""
脚本: npz_to_hdf5.py  (修复版 v2)
功能: 将 faa_2_residue_level_embedding_optimized_v2.py 生成的 batch NPZ
      转换为 HDF5 格式，支持随机访问

NPZ 内部结构说明:
  文件名: batch_000000.npz
  内部 key: <seq_id>.npy  (例如: A0A001XXX.npy)
  读取方式: np.load(f)[seq_id + '.npy']  或  np.load(f)[seq_id]

运行:
  python scripts/emb/npz_to_hdf5.py \
      --npz_dir  data/BIOGRID/embeddings/residue \
      --out_h5   data/BIOGRID/embeddings/embeddings.h5
"""

import os
import argparse
import glob
import numpy as np
import h5py
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="batch NPZ -> HDF5 转换（兼容 optimized_v2 生成格式）")
    parser.add_argument("--npz_dir",        required=True,  help="包含 batch_*.npz 文件的目录")
    parser.add_argument("--out_h5",         required=True,  help="输出 HDF5 文件路径")
    parser.add_argument("--compress_level", type=int, default=4, help="gzip 压缩级别 0-9（默认 4）")
    parser.add_argument("--verbose_errors", action="store_true", help="打印每条失败详情")
    return parser.parse_args()


def convert_batch_npz_to_hdf5(
    npz_dir: str,
    out_h5: str,
    compress_level: int = 4,
    verbose_errors: bool = False,
) -> None:
    """
    遍历 npz_dir 下所有 batch_*.npz，提取每条蛋白 embedding 写入 HDF5。

    内部 key 兼容两种格式:
      - '<seq_id>.npy'  (optimized_v2 格式，writestr 写入)
      - '<seq_id>'      (标准 np.savez 格式)
    """
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "batch_*.npz")))
    if not npz_files:
        # 也尝试所有 .npz
        npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"在 {npz_dir} 下未找到任何 .npz 文件")

    os.makedirs(os.path.dirname(os.path.abspath(out_h5)), exist_ok=True)

    written = 0
    skipped = 0
    errors = 0

    with h5py.File(out_h5, "a") as h5f:
        for npz_path in tqdm(npz_files, desc="转换 batch NPZ -> HDF5"):
            try:
                # allow_pickle=False 兼容 optimized_v2 的 allow_pickle=False 写入
                data = np.load(npz_path, allow_pickle=False)
                raw_keys = data.files   # 例如 ['A0A001.npy', 'B0B002.npy', ...]
            except Exception as e:
                print(f"[错误] 无法读取 {npz_path}: {e}")
                errors += 1
                continue

            for raw_key in raw_keys:
                # 去除 .npy 后缀，得到蛋白 ID
                protein_id = raw_key[:-4] if raw_key.endswith(".npy") else raw_key

                if protein_id in h5f:
                    skipped += 1
                    continue

                try:
                    emb = data[raw_key]                          # (L, D) ndarray
                    emb = emb.astype(np.float32)                 # 统一 float32
                    if emb.ndim != 2:
                        raise ValueError(f"期望 2D (L, D)，实际 shape={emb.shape}")

                    h5f.create_dataset(
                        protein_id,
                        data=emb,
                        compression="gzip",
                        compression_opts=compress_level,
                        dtype=np.float32,
                    )
                    written += 1
                except Exception as e:
                    if verbose_errors:
                        print(f"  [错误] {protein_id}: {e}")
                    errors += 1

    print(f"\n转换完成: 写入={written}, 跳过(已存在)={skipped}, 错误={errors}")
    print(f"输出文件: {out_h5}")


def main():
    args = parse_args()
    convert_batch_npz_to_hdf5(
        npz_dir=args.npz_dir,
        out_h5=args.out_h5,
        compress_level=args.compress_level,
        verbose_errors=args.verbose_errors,
    )


if __name__ == "__main__":
    main()

    import numpy as np, glob

    f = sorted(glob.glob("data/BIOGRID/embeddings/residue/batch_*.npz"))[0]
    d = np.load(f, allow_pickle=False)
    print("文件名:", f)
    print("内部 keys（前5个）:", d.files[:5])
    print("第一条 shape:", d[d.files[0]].shape)