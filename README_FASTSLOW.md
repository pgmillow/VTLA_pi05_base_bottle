# 基于现有 π0.5 的触觉快慢系统

服务器目录：`/home/xudi_ge/pi05_fastslow`。本地代码副本：`/Users/rosepenguin/pi05_fastslow`。
由 `/home/xudi_ge/pi05_atvla` 独立复制；保留原模型、训练和部署目录。
这个版本直接使用现有 **JAX π0.5、视觉语言主干、动作专家、触觉编码器及门控适配器**，不依赖 T-Rex/Qwen 的模型或权重。

## 已实现的推理流程

1. 每个 episode 调用 `reset`，用起始无接触数据标定，持续发送原始触觉帧。
2. `slow` 接收三路 RGB、26 维原始状态和任务指令，运行 π0.5 前缀，保存 KV 缓存；用当时的触觉完成前 6/10 步去噪，保存中间动作 `x_split`。
3. `fast` 使用最新的 16 帧触觉历史，重新计算触觉 token 和 gate，复用同一份 KV 和 `x_split`，完成剩余 4/10 步去噪。
4. 每个快分支修订都从同一个 `x_split` 出发，不把上一次修订的结果反复反馈。已执行部分在快分支中冻结，只交付未来动作。
5. 客户端 `ActionBuffer` 丢弃传输/推理期间已经过期的动作、旧会话和旧版本，并对替换边界及每步动作变化进行限制。

参数的名称、形状与原来的 π0.5 完全一致，没有新增随机初始化网络。因此原检查点可以直接加载，并保留标准 `sample_actions`。触觉相同、执行偏移为 0 时，6+4 分段与原来的 10 步使用同一采样公式。FP32 使用严格数值对照；bfloat16 的两个独立编译图可能产生额外舍入误差，实际差异单独记录在验证报告中，不宣称逐位相同。

这是一版共享动作专家的快慢系统，未实现 T-Rex 的三个独立专家。快分支仍运行 π0.5 动作专家，并非只运行门控 MLP。π0.5 的离散状态位于前缀，**状态、图像、任务指令都通过 slow 刷新；fast 只刷新触觉**。不接受一个看似更新、实际上没有被模型使用的 fast state 参数。

模型计算由同一把锁串行执行，传感器输入有独立锁，客户端动作执行循环应独立运行。快请求可能等待正在执行的慢请求；频率必须结合真实设备和慢分支占用实测。

## 检查点与输入输出

本次全量训练使用经过全部参数检查的触觉适配阶段检查点：

```text
/data/final_double_hands_openpi_atvla/pi05_righthand_ee_atvla/contact_adapter_v1/10000
```

2026-10-02 启动检查发现，旧全量实验的 2000～29999 检查点均包含动作专家 K/V 异常大权重，29999 的最大绝对值约 1e35。该次启动已停止，不再从这些权重继续训练。历史兼容性/延迟报告仍保留原测试条件，不能作为权重健康证明。当前服务和冻结配置的历史默认值尚未迁移，使用时必须显式指定健康检查点；下面的服务命令已指定 10000。

输入和输出的归一化使用检查点 `assets/local/V1show_merged` 内的原统计量；保持原来的 13 维状态选择、32 维模型动作空间和 30 帧动作块。HTTP 输出只提取已训练的右手动作维度 `[6,7,8,9,10,11,13]`，形状为 `[30,7]`，顺序为原动作列对应的右手末端 6 维和手部 1 维。输出沿用原数据定义的物理单位，不把另外未监督的维度作为控制指令。

触觉顺序：`12,13,22,23,32,33,42,43,52,54,61`，每帧 `[11,14,14]`，另带 `[11]` 有效掩码。服务按训练速率约 29.994 Hz 因果重采样，维持约半秒历史，不能把高频采集的最近 16 个帧直接当作训练窗口。

依照此前约定，开始 16 个训练时间网格帧视为无接触：固定逐像素中位数基线、99.5% 残差噪声尺度（下限 1），仍由原编码器做 `asinh` 预处理。减去标定基线后，无接触输入接近 0。会话内不自动把持续接触重新归零。这里不使用曲线显示中的 `0.25 → 100%` 映射，gate 保持模型原始 0～1 值。

默认缓存允许最多 8 个动作帧的延迟（约 267 ms），触觉样本最大年龄 120 ms。超出时拒绝本次修订并要求新 slow 输入。客户端仍要按结果到达时的本地时间丢弃过期动作。

## 启动服务

```bash
cd /home/xudi_ge/pi05_fastslow
bash scripts/serve_pi05_fastslow.sh --checkpoint /data/final_double_hands_openpi_atvla/pi05_righthand_ee_atvla/contact_adapter_v1/10000
```

默认单卡 `CUDA_VISIBLE_DEVICES=0`，端口 `127.0.0.1:18771`。启动时先用合成数据预编译两个分支，然后重置会话，避免把第一次 JIT 编译延迟放到机器人实时循环。可用 `--checkpoint /path/to/checkpoint` 切换相同结构检查点。

服务自身没有机器人控制调用。通过 SSH 转发访问：

```bash
ssh -N -L 18771:127.0.0.1:18771 hit100
```

接口：`GET /health`；`POST /reset`、`/pressure`、`/slow`、`/fast`。图片为 RGB PNG/JPEG 的 base64；状态为原始 26 维。各时间戳全部来自客户端同一个单调时钟。

Python 客户端位于 `src/openpi/fastslow/client.py`，只需要 NumPy、Pillow 和 Python 标准库；模块不导入 JAX。基本调用顺序：

```python
from openpi.fastslow.client import Pi05FastSlowClient
from openpi.fastslow.buffer import ActionBuffer

client = Pi05FastSlowClient("http://127.0.0.1:18771")
session = client.reset()
# 传感器采集循环持续调用，包括最初无接触标定阶段：
client.feed_pressure(raw_11x14x14, valid_11, sensor_timestamp)
# 标定完成后，慢循环调用，images 的键为 top/left_wrist/right_wrist：
context = client.slow(images, prompt, raw_state_26, observation_timestamp)
# 快循环读取当前触觉时刻及已执行动作数；无需重复发送图像和状态：
update, metrics = client.fast(context["chunk_id"], latest_timestamp, executed_steps)
print(metrics["gate"], metrics["fast_ms"])
# 执行循环维护实际控制状态，调用 buffer.apply(update, arrival_time)。
# ActionBuffer 初始化必须提供真实上一条命令和各维合理的 max_step。
# buffer.next_action(now) 返回 None 时，交给现有控制器处理，不重复旧动作。
```

代码块展示接口，不是可直接启动硬件的控制脚本。项目保留原有部署入口；没有接入机器人 SDK。

## 延迟触觉训练

`FastSlowDataset` 按 episode 划分 train/validation（seed 42，约 5% 验证）；每个样本从标定完成后开始，随机选择同轨迹内 0～8 帧延迟：

- 慢输入：t0 的图像、状态、指令和截至 t0 的触觉。
- 快输入：截至 t1 的新触觉；不读取 t1 的图像、状态或未来触觉。
- 动作目标仍从 t0 起算，快损失只覆盖 t1 之后且在 episode 内的动作，尾部填充不参与损失。
- 慢损失保持原动作 flow matching 和接触 BCE。快损失以实际 6 步慢推理结果为起点，构造剩余 flow 时间内到示范动作的桥接目标；慢缓存和起点停止梯度。
- 总损失为 slow loss + fast loss，接触监督权重延续 0.01。日志新增 `slow_loss/fast_loss/fast_contact_loss/fast_gate/fast_start/future_fraction`。

沿用旧检查点的归一化是为了保持权重兼容；旧检查点曾在完整数据上训练，因此这里的验证划分只能评估后续适配过程，**不是从未见过的数据集**。不能把它报告成严格独立的最终泛化评测。

```bash
# 当前实验：视觉语言主干、共享动作专家和触觉模块全部训练。
bash scripts/train_pi05_fastslow_full.sh --exp-name full_from_10000_30k_min2e6_20261002
```

本次全量微调实验使用 `full_from_10000_30k_min2e6_20261002`，在 `tmux train` 的 `pi05_fastslow_full` 窗口运行。启动配置快照和日志位于 `logs/full_from_10000_30k_min2e6_20261002.config.json` 与同名 `.log` 文件。完整模型权重从上述健康 `10000` 检查点初始化。加载时拒绝非有限值或绝对值超过 1e6 的权重；训练时遇到参数范数或已接受更新的逐叶范数溢出会立即退出，避免继续保存异常权重。

默认 30,000 步，前 200 步从 2e-6 预热到 5e-5，随后余弦衰减至 2e-6，batch 8，4 卡 FSDP。相比原单分支增加慢采样计算，先采用较小 batch，未假定原 batch 64 一定能放下。新优化器从零开始；`--resume` 用于续训新实验。输出放在 `/data/final_double_hands_openpi_fastslow`，默认拒绝覆盖同名实验。新检查点会保存 `assets/fastslow.json`，推理自动恢复去噪切分和训练速率，续训检查配置一致性。当前实现没有在原通用训练循环中自动插入定期验证；可通过下述回放脚本检查验证接口及样本链路。

直接加载旧检查点可验证功能和时延；新触觉更新能否提升倒水任务效果，需要延迟训练和实机评测。

## 验证

```bash
bash scripts/test_pi05_fastslow.sh
source scripts/fastslow_env.sh
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/validate_pi05_fastslow.py
# 可额外运行一次完整检查点的内存内优化器更新，不保存覆盖原检查点：
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/validate_pi05_fastslow.py --train-step
```

检查内容包含参数兼容、分段与标准采样一致、触觉变化响应、缓存不变、前缀填充屏蔽、快分支不运行视觉主干、延迟配对/尾部/episode 边界、训练梯度与冻结参数、标定、过期动作、会话切换及 HTTP 协议。实际执行结果见 `VALIDATION_FASTSLOW.md`。
