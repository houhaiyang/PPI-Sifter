from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str) -> Dict[str, Any]:
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as handle:
        current = yaml.safe_load(handle) or {}
    base_config = current.pop('base_config', None)
    if base_config:
        base_path = (config_path.parent / base_config).resolve() if not Path(base_config).is_absolute() else Path(base_config)
        if not base_path.exists():
            base_path = Path(base_config)
        base = load_config(str(base_path))
        return deep_update(base, current)
    return current


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Dict[str, Any], path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def sigmoid_numpy(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def find_best_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * precision * recall / np.clip(precision + recall, 1e-8, None)
    if len(thresholds) == 0:
        return 0.5
    return float(thresholds[int(np.nanargmax(f1_scores[:-1]))])


def compute_classification_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    metrics = {
        'auprc': float(average_precision_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
        'auroc': float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'mcc': float(matthews_corrcoef(labels, preds)) if len(np.unique(preds)) > 1 else 0.0,
        'threshold': float(threshold),
    }
    return metrics
