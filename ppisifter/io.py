from pathlib import Path
import json
import torch
import pandas as pd


def load_embeddings_pt(path):
    payload = torch.load(path, map_location='cpu')
    return payload['data'] if 'data' in payload else payload


def read_pairs_csv(path):
    return pd.read_csv(path)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
