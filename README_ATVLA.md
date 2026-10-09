# π0.5 接触门控适配器（AT-VLA 思路，阶段 A+B）

远程目录：`/home/xudi_ge/pi05_atvla`。
来源：`/home/xudi_ge/Ttest_V1/openpi` 的当前代码副本，包含原项目未提交的改动。
不包含原项目的 `.git`、虚拟环境或旧检查点。运行时复用 `/home/xudi_ge/openpi/.venv/bin/python`，新目录优先加入 PYTHONPATH。

## 已实现

- 保留 CNN → 时间注意力 → 传感器间注意力 → 池化，输出 11×256 触觉 token。
- 根据用户确认，每条轨迹前 16 帧无接触：以这 16 帧中有效数据的逐像素中位数作为固定基线，99.5% 绝对残差分位数（下限 1 原始计数）作为各传感器固定噪声尺度。
- 校准完成前不使用触觉；每个传感器至少需要 4 个有效标定帧。传感器、时间帧、空间填充分别屏蔽；当前帧缺失时关闭对应传感器。
- 新编码模式使用 `asinh((raw-baseline)/scale)`，不会减去当前滑动窗口的中位数，因此保留持续压力。尺度整个 episode 固定，未标定为牛顿等物理单位。
- 用当前有效区域的平均绝对压力 / 噪声尺度构造弱标签：任一传感器 >=3 为接触；全部传感器有效且均 <=1 为无接触；中间区域及不完整负样本不监督。阈值可在 data.contact_on/contact_off 中修改，需按真实数据核验。
- 接触头输入为有效触觉 token 的池化特征 + 当前平均幅度、峰值、超阈值面积比例和最近变化量。连续 sigmoid 门控，不把接触概率当作传感器有效性。
- 将动作 token 作为 Query、11 个触觉 token 作为 Key/Value，添加动作侧门控残差；不增加视觉语言前缀 token，不改变原动作长度和位置编码。
- 仅残差输出投影零初始化；接触头可从第一步获得 BCE 梯度。关闭门控 / 全部无效时，适配器为恒等映射。
- 训练目标：原右手 7 维动作 flow-matching 损失 + 0.01×有效标签的接触 BCE。
- 独立配置默认只训练触觉编码器、接触头和适配器，冻结预训练策略。仍记录并拦截非有限梯度及无效候选参数更新。
- 日志分别记录 flow_loss、contact_loss、contact_gate、contact_accuracy、contact_label_fraction 和 contact_positive_fraction。

这不是 AT-VLA 的 GO-1 Query 硬切换实现。跨控制时刻的视觉 KV 缓存复用、最新连续状态支路及快慢流调度（阶段 C）未在本次加入。现有 sample_actions 在一次采样内复用前缀缓存，并且每次采样只编码一次触觉。

## 检查数据

```bash
cd /home/xudi_ge/pi05_atvla
PYTHONPATH="$PWD/src" /home/xudi_ge/openpi/.venv/bin/python scripts/check_atvla_data.py --episodes 8
```

此命令只读取触觉列，统计弱标签，不启动训练。先核验正负比例和初始无接触假设；阈值不是人工真值。

## 启动新的训练

```bash
cd /home/xudi_ge/pi05_atvla
bash scripts/train_pi05_atvla.sh --exp-name contact_adapter_v1
```

配置名 `pi05_righthand_ee_atvla`；基础权重从 `pi05_base` 加载，新模块随机初始化。不会自动复用旧的 15999 触觉权重，因为预处理与融合位置已变化。

默认输出：
`/data/final_double_hands_openpi_atvla/pi05_righthand_ee_atvla/contact_adapter_v1`

`overwrite=False`、`resume=False`；同名输出存在时拒绝覆盖。继续同一个新实验才使用 `--resume`；新实验改 `--exp-name`。用户尚未要求正式训练，本次不启动 16000 步训练。

数据仍使用原始完整训练集；本次不宣称有独立验证集表现。正式比较时应先按 episode 或采集场次划分训练/验证，并保持各方法一致。

## 在线推理输入

```python
from openpi.models.tactile_contact import TactileContactPreprocessor
preprocessor = TactileContactPreprocessor(num_frames=16, baseline_frames=16)
# 每条轨迹开始保持无接触，连续送入原始传感器帧；换轨迹必须 reset。
extra_fields = preprocessor.update(raw_taxels_11x14x14, sensor_valid_11)
policy_input.update(extra_fields)
# 再交给与训练配置相同的 Octopus policy 数据变换。
```

在线标定只使用已经到达的帧。若没有触觉可完全省略 tactile；若提供触觉则必须同时提供本预处理器产生的基线、尺度和逐帧有效性。全无效时自动回到无触觉分支。

## 测试

```bash
cd /home/xudi_ge/pi05_atvla
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH="$PWD/src" \
  /home/xudi_ge/openpi/.venv/bin/python -m pytest tests/test_atvla.py -q
```

已通过 7 项测试（分批执行）；预热标定修正后，相关 2 项回归再次通过。测试包括固定基线、持续接触、无效与非有限数据、适配器关闭等价性、初始梯度、小型真实 Gemma 的损失和采样，以及一次优化器更新时主干保持冻结。小模型测试不等于完整 π0.5 GPU 训练或机器人评测。

真实数据抽查：episode 0、80、160、240、320、400、480、561 共 7142 帧，接触弱标签 4685、无接触弱标签 1502、不监督 955；每条均完成 11 个传感器标定。这仅验证标签链路能运行，不证明标签准确率。

## 第二阶段：全量微调 30,000 步

配置 `pi05_righthand_ee_atvla_full`，实验名 `full_from_10000_30k`。
加载第一阶段 `contact_adapter_v1/10000/params` 的全部权重，解冻全部模型参数并新建优化器；阶段内步数从 0 开始，额外训练 30,000 步。
学习率前 200 步预热到 5e-5，随后余弦衰减，在 30,000 步结束时达到 5e-6。
其余 batch 64、4 卡 FSDP、接触损失权重 0.01、梯度裁剪 1.0 保持不变。

启动命令：`bash scripts/train_pi05_atvla_full.sh`。
输出：`/data/final_double_hands_openpi_atvla/pi05_righthand_ee_atvla_full/full_from_10000_30k`。
需要第 10,000 步检查点保存完整，并结束占用相同 GPU 的第一阶段后再启动。
本次只新增第二阶段配置和脚本，未切换正在运行的第一阶段进程。
