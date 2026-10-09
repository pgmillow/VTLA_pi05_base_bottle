# 新采集 MCAP 数据处理流程

本文档适用于每次完成一批 MCAP 采集后，将其检查、筛选并转换为一个多 episode 的 LeRobot v2.1 数据集。

当前统一输出规格：

- 每个 MCAP 对应一个 episode
- 整个采集文件夹对应一个 LeRobot 数据集
- `observation.state`：26D
- `action`：14D
- 数值对齐模式：`continuous`
- RPY 在插值前执行 unwrap
- 以 top camera 时间戳作为帧锚点

## 1. 设置本批次路径

以下变量需要针对新批次修改：

```bash
SRC_ROOT=/data/0727right
FOLDER=right_0727
SRC="$SRC_ROOT/$FOLDER"

AUDIT_ROOT="$SRC_ROOT/${FOLDER}_audit"
CLASS_DIR="$AUDIT_ROOT/classification"

OUT_ROOT=/data/lerobot_v21_0727right
DATASET="$OUT_ROOT/$FOLDER"

TASK="Pour juice into the glass"
PYTHON=/home/xudi_ge/openpi/.venv/bin/python
```

首先确认 MCAP 数量：

```bash
python - "$SRC" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
files = sorted(root.glob("*.mcap"))
print(f"MCAP count: {len(files)}")
PY
```

## 2. 执行只读质量分类

分类过程不会修改原始 MCAP：

```bash
"$PYTHON" /data/classify_mcap_quality.py \
  "$SRC" \
  --out-dir "$CLASS_DIR" \
  --converter /data/mcap_to_lerobot_v21.py
```

主要输出：

```text
classification/
├── classification_report.json
├── pass.txt
├── hold_review.txt
├── split_required.txt
├── reject_review.txt
└── records/
```

分类含义：

- `PASS`：未发现明显的数据完整性问题。
- `HOLD_REVIEW`：主要是稳定状态下的手部缺口；当前流程允许保留。
- `SPLIT_REQUIRED`：存在动态手部、arm/EE 或边界缺口；当前策略是人工确认后保留并使用 continuous。
- `REJECT_REVIEW`：存在严重动态缺口、边界缺失或其他高风险问题；当前流程删除。
- `rpy_wrap_requires_unwrap`：EE orientation 跨越 `-π/π`。转换器会在插值前 unwrap，**保留**这些文件，不删除。

分类器同时检查：

- arm joint state 和 command
- 左右 EE position、orientation state
- 左右 hand state 和 command
- top、left wrist、right wrist 图像流
- 时间戳缺口、边界缺失、孤立尖峰
- EE RPY 跨 `-π/π`
- 归位 command 跳变

归位操作引起的 command 跳变，如果 state 保持物理连续，不单独作为删除条件。

## 3. 查看异常证据

对需要人工确认的单个文件生成异常图：

```bash
"$PYTHON" /data/visualize_mcap_anomalies.py \
  --mcap "$SRC/文件名.mcap" \
  --out-dir "$AUDIT_ROOT/viz" \
  --converter /data/mcap_to_lerobot_v21.py
```

重点检查：

- 超过 1 秒的动态数据缺口
- EE position/orientation 是否出现真实突变
- hand command/state 缺口两端是否发生动作
- command 跳变是否只是归位操作
- state 是否连续响应

## 4. 生成删除清单

当前统一删除规则：

1. **仅**删除 `REJECT_REVIEW`
2. **保留** `rpy_wrap_requires_unwrap` 文件（转换器会在插值前 unwrap）

生成删除清单：

```bash
REMOVE_LIST="$CLASS_DIR/remove_files.txt"
cp "$CLASS_DIR/reject_review.txt" "$REMOVE_LIST"
```

删除前必须检查清单和数量：

```bash
wc -l "$REMOVE_LIST"
less "$REMOVE_LIST"
```

建议先移动到隔离目录，确认转换完成后再删除：

```bash
QUARANTINE="$AUDIT_ROOT/quarantine"
mkdir -p "$QUARANTINE"

while IFS= read -r file; do
  [ -z "$file" ] || sudo mv -- "$SRC/$file" "$QUARANTINE/"
done < "$REMOVE_LIST"
```

如果已经备份并确认永久删除：

```bash
while IFS= read -r file; do
  [ -z "$file" ] || printf '%s\0' "$SRC/$file"
done < "$REMOVE_LIST" | sudo xargs -0 --no-run-if-empty rm --
```

再次统计保留数量：

```bash
"$PYTHON" - "$SRC" <<'PY'
from pathlib import Path
import sys
print(len(list(Path(sys.argv[1]).glob("*.mcap"))))
PY
```

## 5. 转换为多 episode LeRobot v2.1 数据集

必须在筛选完成后使用一个新的输出目录，避免混入筛选前生成的 episode。

```bash
"$PYTHON" \
  /home/xudi_ge/data/pipeline/08_batch_folder_to_lerobot_v21.py \
  --src-root "$SRC_ROOT" \
  --folders "$FOLDER" \
  --out-root "$OUT_ROOT" \
  --task "$TASK" \
  --numeric-interp-mode continuous \
  --generate-cmd-state-viz
```

输出结构：

```text
<OUT_ROOT>/<FOLDER>/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── tasks.jsonl
│   └── episode_sources.jsonl
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── viz/
│   ├── episode_000000_all_cmd.png
│   ├── episode_000000_all_state.png
│   └── ...
├── convert_reports/
└── batch_summary.json
```

`--generate-cmd-state-viz` 会在整个文件夹转换完成后，为每个 episode
生成 command 和 state 两张图。

批处理中断后，可以使用完全相同的命令继续；脚本默认根据成功报告恢复。

### Continuous 模式行为

- 保留 top camera 的时间锚点。
- arm、EE、hand command/state 使用连续线性插值。
- RPY 先 unwrap，再进行线性插值。
- wrist image 使用无时间门限的最近帧。
- 短时 hand 离开后立即返回的已知尖峰会先进行清理。

Continuous 会在长缺口内生成平滑的人工中间值，因此不能省略转换前的质量筛选。

## 6. 验证转换结果

检查 MCAP、episode、parquet 数量及 26D/14D schema：

```bash
"$PYTHON" - "$SRC" "$DATASET" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dataset = Path(sys.argv[2])
info = json.loads((dataset / "meta/info.json").read_text())

n_mcap = len(list(src.glob("*.mcap")))
n_parquet = len(list((dataset / "data").glob("chunk-*/episode_*.parquet")))
n_episode = int(info["total_episodes"])
state_dim = info["features"]["observation.state"]["shape"]
action_dim = info["features"]["action"]["shape"]

print("MCAP:", n_mcap)
print("episodes in meta:", n_episode)
print("parquet:", n_parquet)
print("state shape:", state_dim)
print("action shape:", action_dim)

assert n_mcap == n_episode == n_parquet
assert state_dim == [26]
assert action_dim == [14]
print("VERIFY OK")
PY
```

查看批处理失败记录：

```bash
less "$DATASET/batch_summary.json"
```

要求：

- `n_failed` 为 0
- `n_episodes_in_meta` 等于保留 MCAP 数量
- `total_frames` 大于 0

## 7. 转换后异常审计和可视化

再次检查 parquet 中的跳变：

```bash
"$PYTHON" \
  /home/xudi_ge/data/pipeline/10_audit_lerobot_v21_jumps.py \
  --dataset "$DATASET" \
  --out-dir "$DATASET/audit"
```

批处理命令已通过 `--generate-cmd-state-viz` 自动生成 command 和 state
可视化。如果需要单独补生成或重新生成，可执行：

这里的最终可视化直接读取转换后的 parquet，只展示经过 continuous 插值和
对齐后的 action/state 曲线，不展示原始 MCAP 的 gap、缺口标记或流间时间差。
原始 gap 只在转换前的质量审计阶段使用。

```bash
"$PYTHON" \
  /home/xudi_ge/data/pipeline/09_viz_lerobot_v21_from_parquet.py \
  --dataset "$DATASET" \
  --out-dir "$DATASET/viz" \
  --overwrite
```

每个 episode 会生成两张图：

```text
viz/episode_XXXXXX_all_cmd.png
viz/episode_XXXXXX_all_state.png
```

`all_state.png` 包含：

- 左右机械臂 joint state
- 左右手部 state
- 左右 EE position（xyz）
- 左右 EE orientation（roll、pitch、yaw）

可视化脚本同时兼容旧版 27D/15D 和当前新版 26D/14D schema。

最终抽查：

- EE orientation 不再出现错误插值造成的大幅跨圆跳变
- state 没有孤立尖峰
- hand command/state 插值符合实际动作
- 归位 command 可以跳变，但 state 应保持连续
- episode 编号从 0 连续递增
- 全局 `index` 连续且无重复

## 8. 每批数据的完成条件

只有同时满足以下条件，数据才可进入训练：

1. 原始 MCAP 分类扫描完成。
2. 删除或隔离 `REJECT_REVIEW`（RPY wrap 文件保留）。
3. 保留文件数量已记录。
4. continuous 批量转换无失败。
5. episode 数量与保留 MCAP 数量一致。
6. schema 确认为 state 26D、action 14D。
7. 转换后跳变审计和可视化抽查通过。
8. 保存 `classification_report.json`、`remove_files.txt` 和 `batch_summary.json`。
