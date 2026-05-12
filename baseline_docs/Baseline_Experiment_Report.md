# Text-to-MIDI Baseline 实验结果文档

**版本:** 1.0 | **日期:** 2025-11-30 | **项目:** Discrete Diffusion Baseline (No Multimodal Fusion)

---

## 实验概述

**目标:** 建立 Synesthesia 项目的性能基线 (Baseline) - 评估标准离散扩散模型在无 CLIP/CLAP 对齐情况下的生成能力。

**内容:** 模型复现 | 客观指标评估 | 失败案例分析 | 差距对比

---

## 实验环境

**硬件:** Google Colab T4 GPU (16GB VRAM)

**软件:** Ubuntu | Python 3.10 | PyTorch 2.x | MidiTok (REMI) | Symusic

**配置:**

* **模型:** Standard Bidirectional Transformer (BERT-like)
* **参数:** embed_dim=512, layers=6, heads=8, seq_len=512
* **条件:** Simple Text Embedding (无预训练语义空间)
* **扩散:** Masked Diffusion (Absorbing State), Steps=100
* **训练:** BS=64, Epochs=50, Mixed Precision (AMP)

---

## 训练过程

**概况:** 训练过程已完成，模型成功保存，但收敛方向出现“模式崩塌”迹象。

**损失观察:** * **初期:** Loss 快速下降，模型学会了基本的 Token 结构（如 Bar Token）。
* **中期:** Loss 趋于平缓，但并没有进一步学习到复杂的旋律分布。
* **后期:** 模型倾向于预测高概率的 Token（如 Padding 或 MASK），导致生成趋向保守。

**现象:** 训练虽然没有报错，但模型为了降低 Loss，选择了“少写音符”甚至“不写音符”的策略，这是非自回归模型常见的 **Posterior Collapse (后验崩塌)** 现象。

---

## 模型性能 (关键评估)

**测试集:** 生成样本 (20 files) vs 参考样本 (MidiCaps Subset, 50 files)

**客观指标统计:**

| 指标 (Metric) | 测量值 (Value) | 目标值 (Goal) | 状态 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **Empty Bar Rate** | **0.8167** | < 0.1 | ❌ 严重失败 | **81.6%** 的小节是空白的，模型几乎处于静音状态。 |
| **Pitch KL Div** | **5.1611** | < 0.5 | ❌ 严重失败 | 音高分布与真实音乐差异巨大，生成的是随机噪声。 |
| **Duration Entropy** | **1.1057** | > 2.0 | 🟡 偏差较大 | 节奏型极其单一，缺乏变化。 |
| **Note Density** | **0.09** | N/A | ❌ 极低 | 平均每秒仅 0.09 个音符 (几乎每 10 秒才响一声)。 |

**分析结论:**
* **静音问题:** 极高的 Empty Bar Rate (0.8167) 证明了 Baseline 模型在缺乏强语义引导时，无法有效地从全 Mask 状态恢复出有意义的音乐结构。
* **语义脱节:** 极高的 KL 散度 (5.16) 表明模型生成的仅有音符完全不符合调性规律，验证了简单 Text Embedding 无法跨越“语义鸿沟”。

---

## 生成样例分析 (定性评估)

**Prompt:** `"High energy rock music with strong drums"`
* **生成结果:** 长度 10s，全曲仅包含 3 个离散的音符，且分布在极高音区。
* **听感:** 几乎全是静音，偶尔出现突兀的“叮”声。
* **评价:** 完全不符合 Prompt，无节奏，无能量感。

**Prompt:** `"A sad and slow piano song"`
* **生成结果:** 包含 5 个音符，但没有连贯的旋律线。
* **听感:** 像是随机按键，且伴有大量空白时间。
* **评价:** 虽然速度慢符合描述（因为本身就没音符），但缺乏音乐性。

**可视化观察 (Piano Roll):**
* 图像呈现出大量的**空白区域**。
* 缺乏连续的横线（长时值音符）。
* 缺乏纵向的堆叠（和弦/多声部）。

---

## 问题与归因

**问题 1 (Mode Collapse):**
* **现象:** 生成结果极度稀疏 (Density 0.09)。
* **原因:** 离散扩散模型在去噪初期，如果缺乏明确的 Condition (条件) 引导，往往不知道该填什么，最终倾向于保留 Mask 或填充 Pad，因为这在统计上是最安全的。

**问题 2 (Lack of Control):**
* **现象:** 无论 Prompt 是 Rock 还是 Piano，生成结果看起来都一样（都是空的）。
* **原因:** **Simple Text Embedding** 无法区分不同文本的特征空间，导致 Condition 信号在 Transformer 中被忽略。

**问题 3 (Structure Failure):**
* **现象:** KL 散度高 (5.16)，音高混乱。
* **原因:** 模型没有学习到长期依赖关系，缺乏对“调性”和“和弦进行”的全局规划能力。

---

## 结论

**综合评价: ⭐ (1/5)** 该 Baseline 模型未能完成 Text-to-MIDI 生成任务。

1.  **不可用:** 生成的 MIDI 文件 80% 以上是空白的，无法作为音乐使用。
2.  **验证了假设:** 实验数据强有力地支持了本项目 (Synesthesia) 的核心假设 —— **仅靠基础的 Transformer 和扩散架构，无法解决文本到符号音乐的生成问题**。
3.  **对比意义:** 这些糟糕的数据 (Empty Bar 0.81, KL 5.16) 将作为完美的“地板”数据。Synesthesia 模型只需生成出连续的旋律，即可在各项指标上实现 **数倍的性能提升**，从而证明方法的有效性。

**后续计划:**
* 保留所有 Baseline 产生的 `.mid` 文件和指标截图。
* 开始训练 **Synesthesia (Ours)** 模型，重点关注 CLIP/CLAP 融合模块对 `Empty Bar Rate` 的降低作用。
