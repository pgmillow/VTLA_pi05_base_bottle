# 当前全量微调实验

- 服务器：hit100，项目目录 `/home/xudi_ge/pi05_fastslow`。
- tmux：会话 `train`，窗口 `pi05_fastslow_full`；启动时训练 PID 为 807147。
- 实验：`full_from_10000_30k_min2e6_20261002`。
- 配置：`pi05_righthand_ee_fastslow_full`，`freeze_filter=Nothing()`，视觉语言主干、动作专家和触觉模块全部训练。
- 初始化：`/data/final_double_hands_openpi_atvla/pi05_righthand_ee_atvla/contact_adapter_v1/10000/params`，新优化器。
- 30,000 步；前 200 步从 2e-6 预热到 5e-5，随后余弦衰减至 2e-6。
- 全局 batch 8，4 张 A100 80 GB，FSDP 4；每 1000 步保存，保留间隔 2000。
- 输出：`/data/final_double_hands_openpi_fastslow/pi05_righthand_ee_fastslow_full/full_from_10000_30k_min2e6_20261002`。
- 日志和配置快照：项目 `logs/full_from_10000_30k_min2e6_20261002.*`。

## 已验证的启动状态

2026-10-02 04:22:59（Asia/Shanghai）读取 TensorBoard：Step 10 总 loss 4.684726、slow loss 0.503327、fast loss 4.169185；梯度范数 17.284552（裁剪前）、更新范数 0.085087、参数范数 1804.828。没有 NaN/Inf 或跳步。此为启动快照，不代表后续实时状态或收敛结论。详见 `artifacts/training_full_20261002.startup.json`。

## 初始化变更原因

最初从旧全量实验的 29999 检查点启动，发现 `param_norm=inf`，且更新范数的非有限值被旧日志代码显示成 0。直接读取原检查点确认动作专家 `PaliGemma/llm/layers/attn/kv_einsum_1/w` 中 768 个数值超过 1e6，最大约 1.0134e35。该旧全量实验保存的 2000～29999 检查点均有同一处异常，具体产生原因尚未定位。

本次从 29999 启动的进程已经停止，日志保留在 `logs/full_from_29999_30k_min2e6_20261002.*`，旧窗口命名为 `fastslow_bad29999`；没有将其作为有效训练结果。原检查点未修改。

触觉适配阶段 10000 的全部 140 个权重数组通过检查：没有非有限值或超大值，最大绝对值 284；归一化文件与 29999 完全相同。数值诊断见 `artifacts/source_kv_checkpoint_audit.json` 和 `artifacts/source_10000_full_parameter_audit.json`。

训练入口新增初始权重检查，以及训练后的范数溢出检查，在保存检查点前终止异常运行。新增的 5 项数值回归测试全部通过。这些检查会拒绝本次发现的异常；不替代后续 loss 和实机效果评估。

## 查看训练

```bash
ssh -t hit100 'tmux attach -t train'
```

选择 `pi05_fastslow_full` 窗口。TensorBoard 数据在上述输出目录的 `tensorboard` 子目录。
