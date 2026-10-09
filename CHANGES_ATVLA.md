# 修改文件与交付记录

远程新目录：`/home/xudi_ge/pi05_atvla`

原目录：`/home/xudi_ge/Ttest_V1/openpi`。复制后对原目录 77 个源文件逐一校验 SHA-256，均未改变。未启动正式训练。

共修改 8 个文件，新增 6 个实现/测试/说明文件；另附本清单 `CHANGES_ATVLA.md` 和 `changed-files.txt`。

| 类型 | 文件（相对于新目录） | 内容 |
|---|---|---|
| 修改 | `src/openpi/models/model.py` | 新增标定基线、尺度、逐帧有效性和接触监督字段 |
| 修改 | `src/openpi/models/tactile_encoder.py` | 保留现有编码器结构；新增固定基线预处理与逐传感器时序掩码 |
| 修改 | `src/openpi/models/pi0.py` | 动作侧接触门控残差融合；接触辅助损失与推理接入 |
| 修改 | `src/openpi/models/pi0_config.py` | 新增融合模式、接触损失权重和输入规格 |
| 修改 | `src/openpi/policies/octopus_policy.py` | 传递新增触觉字段 |
| 修改 | `src/openpi/training/data_loader.py` | 前16帧标定、接触弱标签、无效与预热帧屏蔽 |
| 修改 | `src/openpi/training/config.py` | 独立 pi05_righthand_ee_atvla 配置、主干冻结、独立检查点目录 |
| 修改 | `scripts/train.py` | 辅助损失梯度与训练日志 |
| 新增 | `src/openpi/models/tactile_contact.py` | 训练/在线共用标定与弱标签，含逐轨迹 reset |
| 新增 | `src/openpi/models/tactile_adapter.py` | 接触头和动作到触觉的交叉注意力残差 |
| 新增 | `scripts/train_pi05_atvla.sh` | 新实验启动脚本，优先加载新目录代码 |
| 新增 | `scripts/check_atvla_data.py` | 只读检查真实数据的弱标签分布 |
| 新增 | `tests/test_atvla.py` | 7项测试：预处理、掩码、门控、模型、训练更新、配置隔离 |
| 新增 | `README_ATVLA.md` | 设计、启动、输入约定、验证结果和限制 |

## 验证

- CPU 测试：首批 6 项通过；单步优化器集成测试 1 项通过；预热修正后相关 2 项回归通过。
- 单步更新：适配器有更新、主干参数逐元素不变、更新未跳过。
- 真实 episode 0 前 31 帧在线/离线所有触觉字段逐元素一致。
- 真实 parquet → repack → OctopusInputs → 批次拼接 → Observation 字段通过。
- 抽查 8 条轨迹、7142 帧：接触弱标签 4685，无接触 1502，不监督 955。
- 关键 Ruff 静态检查通过（排除 jaxtyping 形状字符串的 F722 误报）；启动脚本 bash -n 通过。
- 未进行完整 GPU 训练或机器人评测；阶段 C 的跨控制时刻快慢流调度未实现。

启动方式和详细参数见 `README_ATVLA.md`。默认每条轨迹前16帧无接触，在线也需要相同的初始标定段。
