"""
脚本: infer.py
功能: 批量推理，可选导出 attention map、top-k residue pairs、layer-level representations
依赖: torch, h5py, pandas, pyyaml, tqdm, matplotlib
运行: python scripts/train/infer.py
     （pairs CSV / checkpoint / 输出目录均在 configs/default.yaml 中配置）

[对比学习版改动]
  1. batch 从 tuple 改为 dict 解包
  2. model.forward() 增加 return_layer_reprs 开关（由 infer.export_layer_reprs 控制）
  3. 新增 layer_reprs 导出逻辑：保存为 .pt 文件供 run_contrast_analysis.py 使用
  4. checkpoint 加载兼容新版 {"model": state_dict} 格式
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from ppisifter.model import PPISifter
from ppisifter.data import PPIDataset, collate_fn
from ppisifter.utils import set_seed, get_logger
from ppisifter.interpret import AttentionInterpreter
from ppisifter.config import load_config

_CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "default.yaml",
)


def main() -> None:
    if not os.path.exists(_CFG_PATH):
        raise FileNotFoundError(f"配置文件不存在: {_CFG_PATH}")

    cfg = load_config(_CFG_PATH)

    set_seed(cfg["project"]["seed"])
    logger = get_logger("infer", log_dir=cfg["paths"]["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg  = cfg["data"]
    infer_cfg = cfg["infer"]

    split     = infer_cfg.get("split", "test")
    pairs_csv = os.path.join(data_cfg["splits_dir"], f"{split}.csv")
    if not os.path.isfile(pairs_csv):
        raise FileNotFoundError(f"Pairs CSV 不存在: {pairs_csv}")

    dataset = PPIDataset(
        csv_path=pairs_csv,
        hdf5_path=data_cfg["hdf5_path"],
        max_seq_len=data_cfg["max_seq_len"],
        inference=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=infer_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=data_cfg.get("num_workers", 0),
    )

    m = cfg["model"]
    model = PPISifter(
        d_in=m["d_in"],
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        ffn_expansion=m["ffn_expansion"],
        dropout=m["dropout"],
    ).to(device)

    ckpt_path = infer_cfg.get("checkpoint", "")
    if not ckpt_path:
        ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], "best_auprc.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")

    # [改动 1] 兼容新版 {"model": state_dict} 和旧版直接 state_dict 两种格式
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        logger.info(f"加载 checkpoint (model key): {ckpt_path}，epoch={ckpt.get('epoch', '?')}")
    else:
        model.load_state_dict(ckpt)
        logger.info(f"加载 checkpoint (直接 state_dict): {ckpt_path}")
    model.eval()
    logger.info(f"推理对数: {len(dataset)}")

    threshold = infer_cfg.get("threshold", 0.5)
    out_dir   = cfg["paths"]["pred_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # [改动 2] 读取对比学习版新配置项：是否导出 layer_reprs
    export_layer_reprs: bool = infer_cfg.get("export_layer_reprs", False)

    all_probs = []
    all_preds = []

    # [改动 3] 若需要导出 layer_reprs，则按层收集 list of Tensor
    #   结构: layer_reprs_all[layer_idx] = list of (B, 4*d_model) Tensor
    layer_reprs_all: dict = {}  # {int: list[Tensor]}

    with torch.no_grad():
        # [改动 4] batch 从 tuple 改为 dict 解包
        for batch in tqdm(loader, desc="推理中"):
            emb_a  = batch["emb_a"].to(device)
            emb_b  = batch["emb_b"].to(device)
            mask_a = batch["mask_a"].to(device)
            mask_b = batch["mask_b"].to(device)

            # [改动 5] return_layer_reprs 由配置控制
            out = model(
                emb_a, emb_b, mask_a, mask_b,
                return_attention=False,
                return_layer_reprs=export_layer_reprs,
            )

            probs = out["prob"].cpu().tolist()
            all_probs.extend(probs)
            all_preds.extend([int(p >= threshold) for p in probs])

            # [改动 6] 收集各层 pair repr（仅在 export_layer_reprs=True 时执行）
            if export_layer_reprs and "layer_reprs" in out:
                for layer_idx, repr_tensor in out["layer_reprs"].items():
                    if layer_idx not in layer_reprs_all:
                        layer_reprs_all[layer_idx] = []
                    layer_reprs_all[layer_idx].append(repr_tensor.cpu())

    # 保存 predictions.csv
    pairs_df = pd.read_csv(pairs_csv)
    pairs_df["prob"] = all_probs
    pairs_df["pred"] = all_preds
    out_csv = os.path.join(out_dir, "predictions.csv")
    pairs_df.to_csv(out_csv, index=False)
    logger.info(f"推理完成，结果保存至: {out_csv}")

    # [改动 7] 保存 layer_reprs 为 .pt 文件
    if export_layer_reprs and layer_reprs_all:
        layer_reprs_dir = os.path.join(cfg["paths"].get("analysis_dir", "outputs/analysis"))
        os.makedirs(layer_reprs_dir, exist_ok=True)
        for layer_idx, tensors in layer_reprs_all.items():
            # 拼接所有 batch → (N, 4*d_model)
            full_tensor = torch.cat(tensors, dim=0)
            save_path = os.path.join(layer_reprs_dir, f"layer_reprs_{split}_layer{layer_idx}.pt")
            torch.save({"reprs": full_tensor, "split": split, "layer": layer_idx}, save_path)
            logger.info(f"Layer {layer_idx} reprs 已保存: {save_path}，shape={full_tensor.shape}")

        # 同时保存对应 labels（供 run_contrast_analysis.py 的 linear probe 使用）
        if "label" in pairs_df.columns:
            labels_tensor = torch.tensor(pairs_df["label"].values, dtype=torch.long)
            labels_path = os.path.join(layer_reprs_dir, f"labels_{split}.pt")
            torch.save(labels_tensor, labels_path)
            logger.info(f"Labels 已保存: {labels_path}")
        else:
            logger.warning("CSV 无 label 列，跳过 labels.pt 保存（推理模式）")

    # export_attn: true 时逐样本导出 attention map
    if infer_cfg.get("export_attn", False):
        topk        = infer_cfg.get("topk_residue_pairs", 20)
        attn_dir    = cfg["paths"]["interpret_dir"]
        os.makedirs(attn_dir, exist_ok=True)
        interpreter = AttentionInterpreter(model, device=str(device))

        single_loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )
        for idx, batch in enumerate(tqdm(single_loader, desc="导出 attention")):
            emb_a  = batch["emb_a"]
            emb_b  = batch["emb_b"]
            mask_a = batch["mask_a"]
            mask_b = batch["mask_b"]
            row    = pairs_df.iloc[idx]
            pid_a  = str(row["protein_a"])
            pid_b  = str(row["protein_b"])
            result = interpreter.explain_pair(
                emb_a, emb_b, mask_a, mask_b, topk=topk
            )
            pair_id = f"{pid_a}__{pid_b}"
            interpreter.save_attn_heatmap(
                result["attn_map"],
                os.path.join(attn_dir, f"{pair_id}.png"),
                title=f"{pid_a} vs {pid_b} | prob={result['prob']:.4f}",
            )
            interpreter.save_topk_pairs(
                result["topk_pairs"],
                os.path.join(attn_dir, f"{pair_id}_topk.csv"),
                protein_a_id=pid_a,
                protein_b_id=pid_b,
            )
        logger.info(f"Attention 导出完成，保存至: {attn_dir}")


if __name__ == "__main__":
    main()
