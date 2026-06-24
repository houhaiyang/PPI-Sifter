# REFACTOR-TASKS.md
# PPI-Sifter 重构任务清单（Vibe Coding 专用）

> **使用方式**：将本文档提供给 AI，AI 按任务编号顺序逐步执行代码生成与修改。
> 每完成一个任务后在对应 `- [ ]` 打勾，再进行下一个。
> **不要跨任务跳跃**，确保每一步产物可独立运行再推进。

---

## Phase 0：清理冗余文件

- [ ] **T0-1** 删除 `scripts/interpretability/export_attention.py`
- [ ] **T0-2** 删除 `scripts/interpretability/feature_importance.py`
- [ ] **T0-3** 删除 `scripts/interpretability/visualize_attention.py`
- [ ] **T0-4** 删除 `scripts/interpretability/run_interpret.py`
- [ ] **T0-5** 删除 `scripts/interpretability/case_study__pairs.py`

验收：`scripts/interpretability/` 下仅剩 `plot_interpret.py`、`quantify_interpret.py`

---

## Phase 1：修改 ppisifter/losses.py

**目标**：`PPILoss.forward()` 改为返回 loss dict。

- [ ] **T1-1** `forward()` 返回值从 `scalar` 改为：
  ```
  {"total": scalar, "cls": scalar, "contrast": scalar, "reg": scalar}
  ```
- [ ] **T1-2** `contrast` 项：`contrast.enabled=false` 时返回 `torch.tensor(0.0)`
- [ ] **T1-3** 验证：`loss_dict['total'].backward()` 可正常执行

验收：单元测试 `scripts/tests/test_losses.py` 通过

---

## Phase 2：修改 ppisifter/model.py

**目标**：增加 `return_layer_reprs` 接口，导出各层中间表示。

- [ ] **T2-1** `PPISifter.forward()` 新增参数 `return_layer_reprs: bool = False`
- [ ] **T2-2** 每层 cross-attention 后，若 `return_layer_reprs=True`，构建并缓存：
  ```
  layer_reprs[l+1] = {
    "hidden_a":  H_a_l,       # [B, La, d]
    "hidden_b":  H_b_l,       # [B, Lb, d]
    "attn_ab":   attn_ab,     # [B, H, La, Lb]
    "attn_ba":   attn_ba,     # [B, H, Lb, La]
    "pair_repr": pair_repr_l, # [B, 2d]
  }
  ```
- [ ] **T2-3** `pair_repr_l = cat(mean_pool_a + mean_pool_b, abs(mean_pool_a - mean_pool_b))`
- [ ] **T2-4** 返回值 dict 增加 `'layer_reprs'` 键，仅在启用时填充，否则为 `{}`
- [ ] **T2-5** 原有 `return_attention` 接口保持兼容不变

验收：`scripts/tests/test_model.py` 中验证 `layer_reprs` 各键形状正确

---

## Phase 3：完善 ppisifter/contrast.py

**目标**：实现完整的 LayerwiseContrastHead（支持 supcon / triplet / infonce）。

- [ ] **T3-1** 实现投影头：2 层 MLP + L2 normalize，输入 `2*d_model`，输出 `d_proj`（默认 128）
- [ ] **T3-2** 实现 `SupConLoss`：基于 batch 内同标签正样本聚集，异标签分离
- [ ] **T3-3** 实现 `TripletLoss`：固定 anchor，真实 partner vs hard negative
- [ ] **T3-4** 实现 `InfoNCELoss`【待实验验证】
- [ ] **T3-5** `LayerwiseContrastHead.forward(layer_reprs, labels)` 对 active_layers 中各层加权求和
- [ ] **T3-6** `method` 由 cfg 控制，`enabled=false` 时 forward 直接返回 `tensor(0.0)`

验收：`scripts/tests/test_contrast.py` 覆盖三种 method

---

## Phase 4：完善 ppisifter/analysis.py

**目标**：实现三项对比机制分析函数。

- [ ] **T4-1** `compute_layer_separability(pair_reprs_dict, labels)`
  - 对每层训练 LogisticRegression probe
  - 返回：probe_auroc, probe_auprc, silhouette_score, davies_bouldin_score
- [ ] **T4-2** `compute_partner_shift(layer_reprs_dict, pairs_df)`
  - 固定 anchor，比较真实 partner vs 负 partner 导致的表示偏移
  - 返回：各层 mean_shift, std_shift
- [ ] **T4-3** `compute_entropy_gap(attn_stats_dict, labels)`
  - 计算正负样本各层 attention entropy 均值差
  - 返回：entropy_gap, head_var_gap, sym_error_gap per layer
- [ ] **T4-4** 所有函数输入均支持从 .pt 和 .json 文件加载

验收：三个函数可在 smoke test 数据上独立运行并输出正确格式 JSON

---

## Phase 5：修改 scripts/train/train.py

**目标**：接入 LayerwiseContrastHead，适配 loss_dict。

- [ ] **T5-1** 初始化时额外构建 `contrast_head = LayerwiseContrastHead(cfg)`
- [ ] **T5-2** 训练循环中：
  - `out = model(..., return_layer_reprs=cfg['contrast']['enabled'])`
  - `contrast_loss = contrast_head(out['layer_reprs'], labels)`
  - 将 `contrast_loss` 注入 `PPILoss.forward()`
- [ ] **T5-3** `loss_dict` 各分项（total/cls/contrast/reg）写入 TensorBoard
- [ ] **T5-4** Checkpoint 保存新增 `contrast_head_state_dict`
- [ ] **T5-5** 早停与最优保存基准仍为 val AUPRC

---

## Phase 6：修改 scripts/train/eval.py 和 infer.py

- [ ] **T6-1** `eval.py`：所有 `loss_fn(...)` 调用改为取 `['total']`
- [ ] **T6-2** `infer.py`：增加 `return_layer_reprs` 开关
- [ ] **T6-3** `infer.py`：`export_layer_reprs=True` 时，各层 `pair_repr` 保存为 `outputs/layer_reprs/pair_repr_l{N}.pt`
- [ ] **T6-4** `infer.py`：`attn_stats` 保存为 `outputs/layer_reprs/attn_stats_l{N}.json`

---

## Phase 7：新增 scripts/interpretability/run_contrast_analysis.py

**目标**：一键运行全部对比机制分析。

- [ ] **T7-1** 入口函数，读取 `configs/interpret.yaml`
- [ ] **T7-2** 加载 `outputs/layer_reprs/*.pt` + `predictions.csv`
- [ ] **T7-3** 调用 `analysis.compute_layer_separability()` → 保存 JSON
- [ ] **T7-4** 调用 `analysis.compute_partner_shift()` → 保存 JSON
- [ ] **T7-5** 调用 `analysis.compute_entropy_gap()` → 保存 JSON
- [ ] **T7-6** 打印汇总表（layer x metric）

---

## Phase 8：扩展 scripts/interpretability/plot_interpret.py

- [ ] **T8-1** 新增：Layer probe AUROC 柱状图 → `outputs/interpret/layer_probe_auroc.png`
- [ ] **T8-2** 新增：Partner shift 折线图（均值±标准差）→ `outputs/interpret/partner_shift_profile.png`
- [ ] **T8-3** 新增：Attention entropy gap 折线图 → `outputs/interpret/entropy_gap_profile.png`
- [ ] **T8-4** 新增：各层 UMAP 散点图（正/负样本着色）→ `outputs/interpret/umap_layer_*.png`
- [ ] **T8-5** 原有 attention map 热图功能保持兼容

---

## Phase 9：新增配置文件

- [ ] **T9-1** 新增 `configs/contrast.yaml`（消融配置，覆盖 4 个消融维度）
- [ ] **T9-2** 新增 `configs/interpret.yaml`（推理 + 解释导出专用配置）
- [ ] **T9-3** 更新 `configs/default.yaml`：追加 contrast / loss / infer / analysis 配置块

---

## Phase 10：验收与回归测试

- [ ] **T10-1** 端到端 smoke test：训练 3 epoch（小数据集），确认无报错
- [ ] **T10-2** `eval.py` 运行，输出 JSON 指标无 NaN
- [ ] **T10-3** `infer.py + export_layer_reprs=True`，验证 pt 文件形状正确
- [ ] **T10-4** `run_contrast_analysis.py` 运行，输出三个 JSON 文件格式正确
- [ ] **T10-5** `plot_interpret.py` 运行，全部 PNG 正常生成
- [ ] **T10-6** `contrast.enabled=false` 退化测试：确认行为与原版主干一致
- [ ] **T10-7** 跨平台测试：Win11 本地路径兼容无报错

---

## 附：任务依赖关系

```
T0 → T1 → T2 → T3
              ↓
         T4 (独立)
              ↓
    T5 (依赖 T1 T2 T3)
              ↓
         T6 (依赖 T2)
              ↓
    T7 (依赖 T4 T6)
              ↓
         T8 (依赖 T7)
              ↓
         T9 (任意阶段可并行)
              ↓
        T10 (所有完成后)
```