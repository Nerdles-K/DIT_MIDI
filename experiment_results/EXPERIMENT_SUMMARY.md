# 实验汇总 — Improved Text-to-MIDI Evaluation

生成时间: 2025-11-30

概览:
- 本次评估基于模型检查点 `logs/ckpts/midi_dit_2025-11-29-00-54-13/epoch=25-val_loss=67.8794.ckpt`。
- 样本数量: 90（30 个 prompts × 3 次重复）。
- 所有生成的 MIDI 文件已整理到 `experiment_results/generated_midis/`。
- 所有可视化与聚合报告已放到 `experiment_results/figures/`，主要文件：
  - `evaluation_report.md`（自动生成的评估报告）
  # 实验汇总 — Attachment Prompts (精简保存)

  保存时间: 2025-11-30

  本文件已精简，仅保留“attachment prompts”（来自用户附件的 prompts）本次推理实验结果。

  概览:
  - Prompt 集: `exp/attachment_prompts.yaml`（来自用户上传的 prompt 列表）
  - 生成样本: 48（16 条 prompts × 3 次重复）
  - 成功生成: 48 / 48

  关键统计（摘自 `evaluation_outputs/attachment_prompts/evaluation_summary.json`）：
  - 样本数: `total_samples = 48`, `successful_samples = 48`
  - Note Density (notes/sec, 全部样本): mean = `1.8677`, std = `0.2996`, min = `1.0544`, max = `2.4265`
  - Pitch Entropy (全样本): mean = `4.7474`, std = `0.1874`
  - Duration Entropy (全样本): mean = `0.0609`, std = `0.2033`
  - Empty Bar Rate (全样本): mean = `0.1363`, std = `0.1007`
  - Pitch KL Divergence (全样本): mean = `1.5980`, std = `0.1930`
  - Avg Velocity (全样本均值): mean ≈ `67.83`, std ≈ `1.07`

  主要结论（简要）:
  - Note Density 偏低（约 1.9 notes/s），生成整体较为稀疏，但稳定；
  - Pitch Entropy 良好（≈4.75），表明音高分布多样性接近预期；
  - Duration Entropy 很低（≈0.06），表示节奏/时值多样性不足；
  - Empty Bar Rate 约 13.6%，有部分小节为空或非常稀疏；
  - Pitch KL Divergence ≈1.60，说明与训练数据音高分布仍有差距。

**Pitch KL Divergence（详细说明）**

- 全体样本（按 sample 级别计算并取平均）:
  - mean = `1.59795`，std = `0.19296`，min = `1.19010`，max = `2.30434`（来源：`evaluation_outputs/attachment_prompts/evaluation_summary.json`）
- 说明：评估流水线对每个生成样本计算其音高直方图与训练集参考音高分布之间的 KL 散度（使用自然对数 -> 单位为 nats），然后对所有样本取均值并报告上述统计量。因此上面的值反映的是“样本级 KL 的平均值”，而不是把所有生成样本合并后计算的全局 KL。
- 含义与建议：数值约 1.6 nats 表示生成分布与训练分布存在可观差距（越接近 0 越好）。若需要更直观对比，建议：
  1. 计算“合并所有生成样本的音高分布”并与训练集做一次全局 KL（我可以为你计算）；
  2. 绘制生成分布与训练分布的直方图/核密度图以视觉化差异（我可以生成并保存到 `experiment_results/attachment_figures/pitch_distribution.png`）。

**Baseline 对照（Quick Comparison）**

下面给出两个你指定用于对比的指标：`Empty Bar Rate`（空小节率）与 `Pitch KL`，表中列出 Baseline 值（来源见附件）以及本次 `attachment_prompts` 的实际测量值：

| Metric | Baseline 1 (Diff) | Baseline 3 (GetMusic) | This Run (attachment_prompts) |
|---|---:|---:|---:|
| Empty Bar Rate | 0.81 | 0.00 | 0.1363 |
| Pitch KL Divergence (nats) | 5.16 | 12.28 | 1.59795 |

- 说明：Baseline 值取自你提供的对照表（见附件）。“This Run” 列使用 `evaluation_outputs/attachment_prompts/evaluation_summary.json` 中对应的均值。表明本次模型在 Pitch KL 上明显优于所列 baselines（更接近训练分布），但 Empty Bar Rate 相比 Baseline 3 仍偏高。


  文件与路径（保留项）:
  - 原始评估输出（JSON）: `evaluation_outputs/attachment_prompts/evaluation_results.json`
  - 评估汇总: `evaluation_outputs/attachment_prompts/evaluation_summary.json`
  - 可视化报告（图与 Markdown）: `experiment_results/attachment_figures/`（含 `metrics_by_difficulty.png`, `radar_chart.png`, `metrics_distribution.png`, `evaluation_report.md`）
  - 可直接下载的 MIDI（便捷副本）: `experiment_results/generated_midis/attachment_prompts/`（共 48 个 `.mid` 文件）

  建议的后续动作（优先级排序）:
  1. 提高时值多样性：尝试在后处理或采样上增加随机性，例如提高温度（1.1、1.3）或增加采样步数（75、100），再比较 Duration Entropy。
  2. 控制稀疏小节：在后处理引入每小节最小音数策略或调整阈值百分位（当前使用 92%），以降低 empty bar rate。
  3. 若需对比：我可以把此前的 ablation/旧 baseline 数据（若需要恢复）打包归档后与此结果并列展示。

  复现实验（Windows PowerShell，conda 环境 `midi_dit`）:
  ```powershell
  conda run -n midi_dit python scripts/run_evaluation.py --output-dir evaluation_outputs/attachment_prompts --prompts exp/attachment_prompts.yaml
  conda run -n midi_dit python utils/generate_evaluation_report.py --results evaluation_outputs/attachment_prompts/evaluation_results.json --output-dir experiment_results/attachment_figures --name "Attachment Prompts Evaluation"
  ```

  我已将其他与本次实验无关的聚合文件移除，仅保留上述路径中的文件与报告。如需我现在：
  - 1) 重新打包 `experiment_results/` 为 `experiment_results.zip`（只含保留项），或
  - 2) 把 `EXPERIMENT_SUMMARY.md` 同步到 `docs/`，
  请回复要执行的操作。

  _结束（精简版）_

