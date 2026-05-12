运行说明 — 使用 conda `midi_dit` 环境

目的
- 确保每次运行训练或绘图之前，都会在 PowerShell 中激活 `midi_dit` 环境，避免依赖/版本不一致。

推荐脚本
- `scripts/run_train.ps1`  — 激活 `midi_dit` 并运行 `python train_midi.py`（可传递参数）。
- `scripts/run_plot.ps1`   — 激活 `midi_dit` 并运行绘图脚本 `utils/plot_training_curves.py`（支持转发参数）。

在 PowerShell 中运行（示例）
```powershell
# 进入项目根目录（含 train_midi.py）
cd D:\University\course_materials\dda4220\DIT_MIDI

# 运行训练（如果想传参数可在后面加）
.\scripts\run_train.ps1

# 生成所有图表（默认生成 latest + comparison）
.\scripts\run_plot.ps1

# 生成每个 run 的单独 loss 图
.\scripts\run_plot.ps1 --per-run
```

注意与替代方案
- 如果 PowerShell 未初始化 conda（`conda activate` 报错），请运行一次：
```powershell
conda init powershell
# 然后关闭并重新打开 PowerShell 窗口以使更改生效
```
- 更可靠的无交互方式（不依赖 shell 激活）可以使用 `conda run`：
```powershell
conda run -n midi_dit --no-capture-output python .\train_midi.py
conda run -n midi_dit --no-capture-output python .\utils\plot_training_curves.py --logs-dir logs --per-run --output-dir plots
```
- 如果你使用的是 Anaconda Prompt 或已正确配置的 PowerShell，`conda activate midi_dit` 就可以直接使用。

问题排查
- 若 `conda activate` 无效：检查 Conda 是否在 PATH 或已执行 `conda init powershell`。
- 若脚本找不到 `python`：在激活后运行 `where python`（或 `Get-Command python`）确认可执行路径。

如需我来：
- 我可以现在在终端执行一次 `scripts\run_plot.ps1`（生成 `plots/`），或运行 `scripts\run_train.ps1 --test` 做快速过拟合测试。你想我现在运行哪一个？
