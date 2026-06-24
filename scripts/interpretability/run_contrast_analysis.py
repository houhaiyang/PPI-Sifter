"""
PPI-Sifter 对比学习机制分析脚本
运行此脚本会自动完成：
  1. layer separability analysis
  2. partner shift analysis
  3. attention entropy gap profile
  4. seed stability analysis
  5. 结果保存到 outputs/analysis/

无命令行参数，所有路径从 configs/default.yaml 读取。
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ppisifter.config import load_config
from ppisifter.model import PPISifter
from ppisifter.data import PPIDataset, collate_fn
from ppisifter.contrast import LayerwiseContrastHead
from ppisifter.analysis import (
    layer_separability,
    partner_shift_analysis,
    attention_stats_profile,
    seed_stability,
    pca_layer_repr,
)
from ppisifter.utils import set_seed

CONFIG_PATH = "configs/default.yaml"


@torch.no_grad()
def collect_reprs(model, loader, cfg, device, n_samples=None):
    """
    从测试集收集 layer_reprs、attn_stats、labels、prob。
    n_samples: 最多收集多少个样本（None 表示全部）
    """
    model.eval()
    all_layer_reprs = {}
    all_attn_stats  = []
    all_labels      = []
    all_probs       = []
    count = 0

    for batch in loader:
        emb_a  = batch["emb_a"].to(device)
        emb_b  = batch["emb_b"].to(device)
        mask_a = batch["mask_a"].to(device)
        mask_b = batch["mask_b"].to(device)
        labels = batch["label"].numpy()

        out = model(emb_a, emb_b, mask_a, mask_b,
                    return_attention=True, return_layer_reprs=True)

        layer_reprs = out["layer_reprs"]  # {l: (B, 4d)}
        for l_idx, reprs in layer_reprs.items():
            if l_idx not in all_layer_reprs:
                all_layer_reprs[l_idx] = []
            all_layer_reprs[l_idx].append(reprs.cpu().numpy())

        for b_idx in range(emb_a.size(0)):
            stats_b = {
                l: {
                    "entropy_ab": out["attn_stats"][l]["entropy_ab"],
                    "sym_err":    out["attn_stats"][l]["sym_err"],
                }
                for l in out["attn_stats"]
            }
            all_attn_stats.append(stats_b)

        all_labels.extend(labels.tolist())
        all_probs.extend(out["prob"].cpu().tolist())
        count += emb_a.size(0)
        if n_samples and count >= n_samples:
            break

    for l_idx in all_layer_reprs:
        all_layer_reprs[l_idx] = np.concatenate(all_layer_reprs[l_idx], axis=0)

    return all_layer_reprs, all_attn_stats, np.array(all_labels), np.array(all_probs)


def main():
    cfg    = load_config(CONFIG_PATH)
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    set_seed(cfg["project"]["seed"])

    analysis_dir = Path(cfg["paths"]["analysis_dir"])
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    ckpt_path = Path(cfg["paths"]["checkpoint_dir"]) / "best_auprc.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = PPISifter(**cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"✓ 模型加载完成: {ckpt_path}")

    # 数据
    test_set    = PPIDataset(cfg, split="test")
    test_loader = DataLoader(
        test_set, batch_size=cfg["infer"]["batch_size"],
        shuffle=False, num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
    )

    # 1. 收集表示
    print("收集 layer representations 和 attention statistics ...")
    layer_reprs, attn_stats_list, labels, probs = collect_reprs(
        model, test_loader, cfg, device,
        n_samples=cfg.get("analysis", {}).get("n_samples", None),
    )

    pca_components = cfg.get("analysis", {}).get("pca_components", 50)

    # 2. Layer separability
    print("运行 layer separability analysis ...")
    lr_for_probe = {
        l: pca_layer_repr(reprs, pca_components) if pca_components > 0 else reprs
        for l, reprs in layer_reprs.items()
    }
    sep_results = layer_separability(lr_for_probe, labels)
    print(f"  Layer separability (linear probe AUROC): {sep_results}")
    with open(analysis_dir / "layer_separability.json", "w") as f:
        json.dump({str(k): v for k, v in sep_results.items()}, f, indent=2)

    # 3. Attention entropy gap
    print("运行 attention entropy gap profile ...")
    ent_results = attention_stats_profile(attn_stats_list, labels)
    print(f"  Entropy gap per layer: { {l: v['entropy_gap'] for l, v in ent_results.items()} }")
    with open(analysis_dir / "attention_stats_profile.json", "w") as f:
        json.dump({str(k): v for k, v in ent_results.items()}, f, indent=2)

    # 4. Partner shift analysis（需要正负样本行对齐，此处做 pos/neg 分桶）
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    min_n   = min(len(pos_idx), len(neg_idx), 1000)
    if min_n > 0:
        print("运行 partner shift analysis ...")
        pos_reprs = {l: layer_reprs[l][pos_idx[:min_n]] for l in layer_reprs}
        neg_reprs = {l: layer_reprs[l][neg_idx[:min_n]] for l in layer_reprs}
        shift_results = partner_shift_analysis(pos_reprs, neg_reprs)
        print(f"  Partner shift: {shift_results}")
        with open(analysis_dir / "partner_shift.json", "w") as f:
            json.dump({str(k): v for k, v in shift_results.items()}, f, indent=2)

    print(f"\n✓ 分析完成，结果保存至 {analysis_dir}")


if __name__ == "__main__":
    main()