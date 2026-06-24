# CODEBASE-CONTEXT.md
# PPI-Sifter 代码仓库上下文速查（供 AI 辅助编码使用）

> **用途**：将此文档与代码仓库一同提供给 AI，让其在编写/修改代码时获得完整架构上下文，避免幻觉。
> **仓库**：https://github.com/houhaiyang/PPI-Sifter
> **当前状态**：旧版已有基础主干代码，需按《项目设计书（对比学习版）》进行全量重构。

---

## 一、仓库现状（改造前文件清单）

### ppisifter/ 包

| 文件 | 现状 | 改造动作 |
|---|---|---|
| model.py | 主模型，含双向 cross-attention 主干 | **修改**：增加 return_layer_reprs 接口 |
| attention.py | MultiHeadCrossAttention | 保留不改 |
| modules.py | GatedFFN, AttentionPooling | 保留不改 |
| contrast.py | 雏形 | **修改**：完善 LayerwiseContrastHead |
| losses.py | 返回标量 | **修改**：改为返回 loss_dict |
| analysis.py | 雏形 | **修改**：完善三项分析函数 |
| data.py | PPIDataset，基于 HDF5 | 保留，微调 negative_type 字段支持 |
| metrics.py | AUROC, AUPRC, F1 | 保留，补充 MCC |
| interpret.py | 旧版 attention map 解释 | 保留，兼容旧接口 |
| io.py | HDF5 读写 | 保留不改 |
| config.py | 配置加载 | 保留，补充 contrast/analysis 字段校验 |
| constants.py | 全局常量 | 保留不改 |
| train_utils.py | 训练辅助 | 保留，补充 loss_dict 日志支持 |
| utils.py | 通用工具 | 保留不改 |

### scripts/ 目录

| 文件 | 现状 | 改造动作 |
|---|---|---|
| scripts/train/train.py | 训练主入口 | **修改**：接入对比头，适配 loss_dict |
| scripts/train/eval.py | 标准评估 | **小改**：loss_dict['total'] 对接 |
| scripts/train/infer.py | 推理脚本 | **小改**：增加 return_layer_reprs 开关 |
| scripts/emb/ | ESM embedding 提取 | 保留 |
| scripts/biogrid/ | BioGRID 数据预处理 | 保留 |
| scripts/interpretability/run_contrast_analysis.py | 不存在 | **新增** |
| scripts/interpretability/plot_interpret.py | 已有 | **扩展**：支持新分析结果绘图 |
| scripts/interpretability/quantify_interpret.py | 已有 | 保留备用 |
| scripts/interpretability/export_attention.py | 已有 | **删除** |
| scripts/interpretability/feature_importance.py | 已有 | **删除** |
| scripts/interpretability/visualize_attention.py | 已有 | **删除** |
| scripts/interpretability/run_interpret.py | 已有 | **删除** |
| scripts/interpretability/case_study__pairs.py | 已有 | **删除** |

---

## 二、模型前向流（目标架构）

```
输入: emb_a [B, La, 1152], emb_b [B, Lb, 1152]
      mask_a [B, La],  mask_b [B, Lb]

Step 1 - 共享投影:
  H_a_0 = emb_a @ W_proj  ->  [B, La, d_model]
  H_b_0 = emb_b @ W_proj  ->  [B, Lb, d_model]

Step 2 - N 层双向 Cross-Attention + GatedFFN:
  for l in 0..N-1:
    H_a_l, attn_ab = CrossAttn(Q=H_a_{l-1}, K=H_b_{l-1}, V=H_b_{l-1})
    H_b_l, attn_ba = CrossAttn(Q=H_b_{l-1}, K=H_a_{l-1}, V=H_a_{l-1})
    H_a_l = GatedFFN(H_a_l) + residual
    H_b_l = GatedFFN(H_b_l) + residual

    if return_layer_reprs:
      pool_a_l = mean_pool(H_a_l)              # [B, d]
      pool_b_l = mean_pool(H_b_l)              # [B, d]
      pair_repr_l = cat(pool_a_l + pool_b_l,
                        abs(pool_a_l - pool_b_l))  # [B, 2d]
      layer_reprs[l+1] = {
          "hidden_a":  H_a_l,
          "hidden_b":  H_b_l,
          "attn_ab":   attn_ab,
          "attn_ba":   attn_ba,
          "pair_repr": pair_repr_l,
      }

Step 3 - Attention Pooling:
  s_a = AttnPool(H_a_N)  ->  [B, d]
  s_b = AttnPool(H_b_N)  ->  [B, d]

Step 4 - 对称共享表示:
  V = cat(s_a + s_b, abs(s_a - s_b))  ->  [B, 2d]

Step 5 - MLP 分类头:
  logit = MLP(V)   ->  [B, 1]
  prob  = sigmoid(logit)

返回值 dict:
  "logit":       [B, 1]
  "prob":        [B, 1]
  "layer_reprs": {1: {...}, 2: {...}}  # 仅 return_layer_reprs=True
  "attn_ab_last": [B, H, La, Lb]       # 仅 return_attention=True
  "attn_ba_last": [B, H, Lb, La]       # 仅 return_attention=True
```

---

## 三、PPILoss 返回格式（必须返回 dict）

```python
# losses.py  PPILoss.forward() 必须返回:
loss_dict = {
    "total":    tensor_scalar,  # 唯一用于 .backward()
    "cls":      tensor_scalar,  # weighted BCE + focal
    "contrast": tensor_scalar,  # SupCon/Triplet/InfoNCE（disabled 时为 0.0）
    "reg":      tensor_scalar,  # sparse + sym + entropy reg 之和
}
```

---

## 四、LayerwiseContrastHead 接口规范

```python
class LayerwiseContrastHead(nn.Module):
    """
    对各层 pair_repr 施加监督式对比损失。
    输入:
        layer_reprs: dict {layer_idx: {'pair_repr': Tensor [B, 2d]}}
        labels:      Tensor [B], binary 0/1
    输出:
        contrast_loss: scalar Tensor
    """
    def forward(self, layer_reprs: dict, labels: torch.Tensor) -> torch.Tensor:
        ...
```

---

## 五、run_contrast_analysis.py 输出格式

```
outputs/analysis/layer_separability.json
{
  "layer_1": {"probe_auroc": 0.xx, "probe_auprc": 0.xx, "silhouette": 0.xx, "db_index": 0.xx},
  "layer_2": {...}
}

outputs/analysis/partner_shift.json
{
  "layer_1": {"mean_shift": 0.xx, "std_shift": 0.xx},
  "layer_2": {...}
}

outputs/analysis/attention_stats_profile.json
{
  "layer_1": {
    "entropy_pos": 0.xx, "entropy_neg": 0.xx, "entropy_gap": 0.xx,
    "head_var_pos": 0.xx, "head_var_neg": 0.xx,
    "sym_error_pos": 0.xx, "sym_error_neg": 0.xx
  },
  "layer_2": {...}
}
```

---

## 六、三处必须适配的代码改动

### ① scripts/train/eval.py

```python
# 改前:
loss = loss_fn(logits, targets, attn_ab, attn_ba)

# 改后:
loss_dict = loss_fn(logits, targets, attn_ab, attn_ba)
loss = loss_dict["total"]
```

### ② scripts/train/infer.py

```python
# 改前:
out = model(emb_a, emb_b, mask_a, mask_b, return_attention=True)

# 改后:
export_reprs = cfg["infer"].get("export_layer_reprs", False)
out = model(emb_a, emb_b, mask_a, mask_b,
            return_attention=True,
            return_layer_reprs=export_reprs)
```

### ③ ppisifter/interpret.py（若有 loss_fn 调用）

```python
loss = loss_fn(...)['total']
```

---

## 七、新增 YAML 配置块

```yaml
contrast:
  enabled: true
  active_layers: [1, 2]
  method: supcon          # supcon | triplet | infonce
  d_proj: 128
  temperature: 0.07
  margin: 0.5
  lambda_contrast: 0.3

loss:
  beta_contrast: 0.3
  beta_reg: 0.1
  lambda_focal: 0.5
  lambda_sparse: 0.1
  lambda_sym: 0.1
  lambda_entropy: 0.05

infer:
  export_layer_reprs: true
  output_dir: outputs/

analysis:
  active_layers: [1, 2]
  probe_max_iter: 200
  probe_c: 1.0
  partner_shift_n_anchors: 100
  seed: 42
```

---

## 八、新增文件（需从零编写）

| 文件路径 | 说明 |
|---|---|
| scripts/interpretability/run_contrast_analysis.py | 三项分析主入口，调用 ppisifter/analysis.py |
| configs/contrast.yaml | 消融实验配置 |
| configs/interpret.yaml | 推理 + 解释导出专用配置 |

---

## 九、数据格式规范

### pairs_*.csv

```
protein_a, protein_b, label, split, negative_type
P12345,    Q67890,    1,     train, ''
P11111,    Q22222,    0,     train, random
```

### embeddings.h5

```
/{protein_id}/embedding    float16  shape=[L, 1152]
/{protein_id}/length       int64    scalar
```

---

## 十、Checkpoint 格式

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "contrast_head_state_dict": contrast_head.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "metrics": {"auprc": val_auprc, "auroc": val_auroc},
    "config": cfg,
}, ckpt_path)
```

---

## 十一、代码规范速查

1. Python 3.11 · PyTorch 2.1.0 · CUDA 12.1
2. 函数/变量 snake_case，类 PascalCase，常量 UPPER_CASE
3. 文件头部注释：功能 / 依赖 / 执行命令
4. 维度变换必须注释形状，如 # [B, L, d] -> [B, d]
5. 所有超参从 cfg dict 读取，禁止硬编码
6. 路径统一 pathlib.Path，兼容 Win11 + Linux
7. 外部输入做校验，关键流程有 try-except + 友好报错
8. 未验证优化点标注 # 【待实验验证】