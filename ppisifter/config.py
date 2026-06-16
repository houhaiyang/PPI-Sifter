from pathlib import Path
import yaml


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    parent = cfg.pop('default_config', None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.exists():
            parent_path = path.parent / Path(parent).name
        base = load_config(parent_path)
        return _merge(base, cfg)
    return cfg
