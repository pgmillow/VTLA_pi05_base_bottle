# 修改文件清单

来源 `/home/xudi_ge/pi05_atvla`，新目录 `/home/xudi_ge/pi05_fastslow`；本地副本 `/Users/rosepenguin/pi05_fastslow`。

## 新增核心模块

| 文件 | 作用 |
|---|---|
| `src/openpi/fastslow/model.py` | π0.5 配置与子类、不可变慢缓存、6+4 分段去噪、未来动作掩码和延迟训练损失；没有新增参数 |
| `src/openpi/fastslow/data.py` | episode 划分、同轨迹 t0/t1 配对、因果触觉历史、动作尾部屏蔽 |
| `src/openpi/fastslow/pressure.py` | 约 30 Hz 时间网格重采样、固定无接触标定、传感器缺失处理 |
| `src/openpi/fastslow/policy.py` | 检查点加载、慢/快接口、会话与缓存寿命、右手 7 维输出 |
| `src/openpi/fastslow/buffer.py` | 客户端动作版本、过期丢弃、未来动作替换和变化幅度限制 |
| `src/openpi/fastslow/serve.py` | HTTP 服务、模型预编译、健康检查和传感器/推理接口 |
| `src/openpi/fastslow/client.py` | 无 JAX 依赖的客户端 |
| `src/openpi/fastslow/__init__.py` | 包入口 |

## 修改已有模块

| 文件 | 改动 |
|---|---|
| `src/openpi/models/model.py` | 修复 Orbax 丢弃无偏置卷积 `None` 叶子导致的加载失败；非空权重仍严格检查 |
| `src/openpi/training/config.py` | 新增快慢训练配置和独立输出目录；全量版本改从数值健康的触觉适配阶段 10000 初始化，旧 29999 检查点已发现异常 |
| `src/openpi/training/data_loader.py` | 新配置使用配对数据和 `FastSlowObservation`，携带元数据；原配置继续原路径 |
| `src/openpi/training/checkpoints.py` | 新检查点保存 `fastslow.json`，续训核验时间/切分/动作/数据划分配置 |
| `src/openpi/training/optimizer.py` | 支持显式预热初始学习率，新快慢实验从 2e-6 预热并衰减回 2e-6 |
| `scripts/train.py` | 加载时检查异常权重；训练时检测范数溢出并在保存检查点前停止 |
| `.gitignore` | 忽略本地验证产物、日志和 macOS 元数据 |

原 `pi0.py`、触觉编码器和触觉适配器无需修改。所有迁移逻辑都在新子类与新接口中。

## 新增入口、测试与文档

- `scripts/fastslow_env.sh`
- `scripts/serve_pi05_fastslow.sh`
- `scripts/train_pi05_fastslow.sh`
- `scripts/train_pi05_fastslow_full.sh`（固定全量解冻入口）
- `scripts/test_pi05_fastslow.sh`
- `scripts/validate_pi05_fastslow.py`
- `tests/fastslow/test_fastslow.py`
- `tests/fastslow/test_training_numerics.py`（异常检查点和范数溢出回归测试）
- `README_FASTSLOW.md`
- `VALIDATION_FASTSLOW.md`
- 本文件 `CHANGES_FASTSLOW.md`

验证产物放在 `artifacts/`，不作为模型权重或训练结果发布。
