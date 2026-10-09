# VTLA Pi05 Base Bottle

本仓库管理 `/home/xudi_ge/VTLA_pi05_base_bottle` 的训练代码，基于 Physical Intelligence 的 openpi，使用 π0.5 base、O10 268 通道触觉、接触门控和动作 Adapter 学习拧瓶盖。

## 本次完成的训练

| 项目 | 值 |
| --- | --- |
| 配置 | `vtla_bottle_direct_full` |
| 实验 | `base_touch_joint_46k_seed43` |
| 训练 seed / episode 划分 seed | 43 / 42 |
| batch / 梯度累积 / FSDP | 64 / 1 / 4 张 GPU |
| 续训 | checkpoint 19000 后保留优化器，再执行 5000 次更新 |
| 学习率尾部 | 余弦衰减，约 `3.26594e-5` → `1e-5`，不重新 warmup |
| 最终 checkpoint | `24000`，内部 step、Adam、schedule 计数均为 24001 |
| 完成时间 | 2026-10-09 03:10:55，北京时间 |
| 结果 | 恢复校验通过，正常退出，退出码 0 |

最终 checkpoint 保存在服务器：

```text
/data/checkpoints_vtla_bottle/vtla_bottle_direct_full/base_touch_joint_46k_seed43/24000
```

原始配置计划训练 46000 步；本次在 checkpoint 19000 改为额外 5000 步的收尾计划，实际完成至 24000。训练状态与源码哈希快照见 [docs/training/](docs/training/)。原始运行清单的 85 个条目在入库时全部 SHA-256 一致；其中 76 个实际训练源码纳入版本管理，9 个 macOS 元数据文件按忽略规则排除。

## 代码入口

| 路径 | 用途 |
| --- | --- |
| `scripts/train_vtla_finish.py` | 精确恢复 checkpoint 与优化器，执行余弦收尾续训 |
| `scripts/train.py` | 主训练循环 |
| `scripts/train_vtla_direct.sh` | 从 π0.5 base 启动直接联合训练及预检 |
| `scripts/watch_vtla_finish.py` | 本次等待 checkpoint、交接并启动续训的控制脚本 |
| `src/openpi/training/config.py` | `vtla_bottle_direct_full` 与其他训练配置 |
| `src/openpi/vtla/` | VTLA 模型、O10 触觉、数据缓存及分阶段方案 |
| `data_pipeline/` | MCAP 解析、质量检查及 LeRobot 转换 |
| `scripts/prepare_vtla_bottle.py` | 缓存、训练集归一化和数据审查 |
| `tests/vtla/`、`tests/fastslow/` | 已有 VTLA 和 fast/slow 测试 |

## 数据与模型

状态为 23 维（arm 7 + hand 10 + measured EE 6），动作监督为 17 维绝对关节目标（arm 7 + O10 hand 10），action horizon 为 30。触觉保留原生 268 通道，使用 16 帧因果窗口；训练采用 paired slow/fast 目标与接触弱监督。

数据、原始 MCAP、模型权重、归一化资产、缓存、运行日志与导出包保留在服务器，由 `.gitignore` 排除。上游 Apache 2.0 许可证及 Gemma 许可证随源码保留。

## 环境和路径

依赖声明在 `pyproject.toml`，锁定版本在 `uv.lock`。本次实际运行复用了 `/home/xudi_ge/openpi/.venv/bin/python`。本次同步未重新安装或验证全新的依赖环境。

迁移机器时，需要在配置和启动脚本中调整以下服务器路径：

| 内容 | 当前路径 |
| --- | --- |
| 数据 | `/data/open_bottle_cap_vtla_20261006/lerobot` |
| π0.5 base 参数 | `/home/xudi_ge/openpi-assets/checkpoints/pi05_base/params` |
| 归一化资产根目录 | `/home/xudi_ge/VTLA_pi05_base_bottle/assets/bottle` |
| 输出根目录 | `/data/checkpoints_vtla_bottle` |
| Python | `/home/xudi_ge/openpi/.venv/bin/python` |

克隆时使用 `git clone --recurse-submodules`；已有克隆执行 `git submodule update --init --recursive`。`third_party/aloha` 和 `third_party/libero` 按服务器现有 openpi 仓库中的上游版本固定。

## 本次执行的命令

以下记录用于复现已完成的训练。重新实验时先选择独立实验与输出目录，并准备对应 checkpoint、数据和归一化资产。

```bash
cd /home/xudi_ge/VTLA_pi05_base_bottle
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export JAX_TRANSFER_GUARD_DEVICE_TO_DEVICE=disallow
export PYTHONUNBUFFERED=1
unset XLA_FLAGS JAX_PLATFORMS JAX_PLATFORM_NAME

/home/xudi_ge/openpi/.venv/bin/python scripts/train_vtla_finish.py \
  --checkpoint-step 19000 --additional-steps 5000 --end-lr 1e-5
```

原始直接训练的入口如下；正式训练要求相同配置的 100 次更新预检通过：

```bash
bash scripts/train_vtla_direct.sh preflight
bash scripts/train_vtla_direct.sh train
```

## Git 管理和检查

`main` 保存正式源码；更改算法时建议在独立分支提交。普通更新使用 `git add <文件>`、`git commit`、`git push origin main`。

GitHub 自动检查 Python 和 shell 语法。完整模型及 GPU 测试需要训练环境；上游格式检查保留为手动 workflow，不会为同步源码自动改写训练代码。
