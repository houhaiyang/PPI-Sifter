"""
PPI-Sifter 全局常量定义
"""

# ESM-C 各版本输出维度
ESM_C_DIM = {
    "esm_c_300m": 960,
    "esm_c_600m": 1152,
}

# HDF5 嵌入文件中的数据集命名格式
HDF5_EMB_KEY = "{protein_id}"   # h5["<protein_id>"] -> np.ndarray (L, D)

# 数据集划分文件名约定
SPLIT_TRAIN = "train.csv"
SPLIT_VALID = "valid.csv"
SPLIT_TEST  = "test.csv"
