# AI-PROMPT-TEMPLATE.md
# 给 GPT-5.4 Thinking 的 Vibe Coding 提示词模板

> 每次向 AI 发起编码任务时，按此模板组织 prompt，确保 AI 有足够上下文、不产生幻觉。

---

## 标准 Prompt 结构

```
你是一名资深 Python 工程师，正在参与 PPI-Sifter 项目的重构开发。

【项目背景】
PPI-Sifter 是一个蛋白质相互作用预测模型，使用冻结 ESM-C + B-PPI 风格双向
cross-attention 主干 + 对比学习式机制解释。仓库地址：
https://github.com/houhaiyang/PPI-Sifter

【参考文档】（已附上）
- CODEBASE-CONTEXT.md：仓库现状、目标架构、接口规范、数据格式
- 项目设计书（对比学习）.md：完整设计方案
- 对比学习解释性分析流水线.md：分析链路详细说明

【当前任务】
[在此填写具体任务，例如：完成 Phase 2 T2-1 到 T2-5，修改 ppisifter/model.py]

【需要修改的文件】
[在此列出需要修改或新建的文件路径]

【约束条件】
1. Python 3.11，PyTorch 2.1.0，CUDA 12.1
2. 所有超参从 cfg dict 读取，不硬编码
3. 路径使用 pathlib.Path，兼容 Win11 + Linux
4. 函数/变量 snake_case，类 PascalCase
5. 维度变换处必须注释形状，如 # [B, L, d] -> [B, d]
6. 文件头部注释：功能 / 依赖 / 执行命令
7. 输出完整可运行代码，不截断
8. 未验证的优化点标注 # 【待实验验证】

【请按以下格式输出】
1. 修改思路（3-5句话概述）
2. 完整代码（每个文件单独一个代码块，文件路径作为注释写在第一行）
3. 运行测试命令
4. 注意事项
```

---

## 各 Phase 专用补充说明

### Phase 1（losses.py）补充

```
【特别要求】
- PPILoss.forward() 原来接收什么参数，改造后保持签名不变
- contrast_loss 参数在 forward 中作为可选 kwarg 传入：contrast_loss=None
- 若 contrast_loss 为 None，对应 dict 项为 tensor(0.0, device=...)
```

### Phase 2（model.py）补充

```
【特别要求】
- 不要改变 forward() 中原有任何计算逻辑，只在末尾增加 layer_reprs 收集逻辑
- pair_repr_l 使用 mean pooling（不是 attention pooling），保持轻量
- layer_reprs 字典 key 从 1 开始（1-indexed）
```

### Phase 3（contrast.py）补充

```
【特别要求】
- SupConLoss 参考 Khosla et al. 2020 实现，temperature 从 cfg 读取
- 投影头权重不参与主干 optimizer，建议单独参数组
- forward 输入的 layer_reprs 中只处理 active_layers 指定的层
```

### Phase 7（run_contrast_analysis.py）补充

```
【特别要求】
- 支持 --config 命令行参数指定 YAML 路径
- 支持 --layer_reprs_dir 覆盖默认路径
- 运行完毕后在终端打印汇总表：layer x {probe_auroc, entropy_gap, mean_shift}
```

---

## 调试优先级建议

| 优先级 | 任务 | 建议在哪运行 |
|---|---|---|
| P0 | Phase 0~2（losses + model 修改）| 本地 RTX4070S debug |
| P1 | Phase 3~5（contrast + train）| 本地验证 forward，云端 A100 完整训练 |
| P2 | Phase 6~8（infer + analysis + plot）| 本地 |
| P3 | Phase 9~10（配置 + 回归测试）| 本地 |