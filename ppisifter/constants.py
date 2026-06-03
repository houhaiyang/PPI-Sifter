"""Project-wide constants."""

DEFAULT_SEED = 42
EPS = 1e-8
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
DEFAULT_EMBEDDING_SUFFIX = ".residue_emb.npy"
DEFAULT_SPLITS = ("train", "valid", "test")
SUPPORTED_SPLIT_STRATEGIES = ("random", "protein_disjoint")
