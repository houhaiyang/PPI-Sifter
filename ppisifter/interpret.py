from pathlib import Path
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def topk_residue_pairs(attn_map, valid_mask, k=20):
    masked = attn_map.masked_fill(~valid_mask, -1)
    flat = masked.reshape(-1)
    valid_n = int(valid_mask.sum().item())
    k = min(k, valid_n)
    vals, idx = torch.topk(flat, k=k)
    n_cols = attn_map.shape[1]
    rows = (idx // n_cols).cpu().tolist()
    cols = (idx % n_cols).cpu().tolist()
    return [
        {'residue_a': int(r) + 1, 'residue_b': int(c) + 1, 'score': float(v)}
        for r, c, v in zip(rows, cols, vals.cpu().tolist())
    ]


def save_heatmap(attn_map, output_png, title='attention_map'):
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(attn_map, cmap='Purples')
    plt.title(title)
    plt.xlabel('Protein B residues')
    plt.ylabel('Protein A residues')
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def export_pair_interpretation(meta, attn_map, valid_mask, output_dir, topk=20):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_name = f"{meta['protein_a']}__{meta['protein_b']}"
    a_len = int(valid_mask.sum(dim=1).max().item())
    b_len = int(valid_mask.sum(dim=0).max().item())
    arr = attn_map[:a_len, :b_len].cpu().numpy()
    topk_items = topk_residue_pairs(attn_map, valid_mask, k=topk)
    csv_path = output_dir / f'{pair_name}.topk.csv'
    png_path = output_dir / f'{pair_name}.heatmap.png'
    json_path = output_dir / f'{pair_name}.meta.json'
    pd.DataFrame(topk_items).to_csv(csv_path, index=False)
    save_heatmap(arr, png_path, title=pair_name)
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'csv': str(csv_path), 'png': str(png_path), 'json': str(json_path)}
