# Text-to-MIDI Baseline 实验结果文档 (II)

**版本:** 1.0 | **日期:** 2025-11-30 | **项目:** LLM-based Baseline (ChatMusician / ABC Notation)

---

## 实验概述

**目标:** 评估基于大语言模型 (LLM) 和 ABC 记谱法的生成方案在 Text-to-MIDI 任务上的表现，并与 Discrete Diffusion Baseline 形成对比。

**方法:** 使用预训练的 **ChatMusician (Base on LLaMA-2 7B)** 模型进行推理，将生成的 ABC 文本转换为 MIDI 进行指标评测。

---

## 实验环境

**硬件:** Google Colab T4 GPU (16GB VRAM)

**软件:** Transformers 4.36+ | BitsAndBytes (4-bit Quantization) | Music21

**配置:**

* **模型:** m-a-p/ChatMusician (7B Parameters)
* **输入:** Text Prompt -> ABC Notation
* **解码:** Greedy/Sampling (Temp=0.7) -> ABC to MIDI Conversion
* **训练状态:** Inference Only (Pre-trained)

---

## 模型性能 (关键评估)

**测试集:** 生成样本 (5 files) vs 参考样本 (MidiCaps Subset, 50 files)

**客观指标统计:**

| 指标 (Metric) | 测量值 (Value) | 目标值 (Goal) | 状态 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **Empty Bar Rate** | **0.0000** | < 0.1 | ✅ 完美 | **0%** 的空白小节。LLM 完美掌握了乐谱的语法结构，彻底解决了扩散模型的“静音”问题。 |
| **Note Density** | **3.53** | N/A | ✅ 优秀 | 每秒 3.53 个音符，密度适中，内容丰富，远优于扩散模型的 0.09。 |
| **Duration Entropy** | **1.0296** | > 2.0 | 🟡 偏差较大 | 节奏型较为单一（主要由 ABC 记谱法的量化特性导致）。 |
| **Pitch KL Div** | **5.9659** | < 0.5 | ❌ 严重偏差 | 音高分布与参考集（MidiCaps/LMD）差异极大，表明风格严重不匹配。 |

**分析结论:**
1.  **结构性完胜:** Empty Bar Rate 为 0，证明了 **Autoregressive LLM** 在生成强结构化序列（Syntax）方面的绝对统治力。它绝不会像扩散模型那样“不知道该填什么”。
2.  **风格/领域偏差 (Domain Mismatch):** KL 散度极高 (5.96) 甚至高于随机噪声。这并非因为生成的不好听，而是因为 **ChatMusician 主要在爱尔兰民歌/单声部旋律 (ABC数据)** 上训练，而参考集 **MidiCaps/LMD** 多为复杂的现代流行/摇滚多轨音乐。分布完全不重叠。
3.  **表现力受限:** Duration Entropy 较低 (1.03)，说明节奏过于规整（机械化），缺乏真实演奏的微小变化 (Micro-timing) 和复杂切分。

---

## 生成样例分析 (定性评估)

**Prompt:** `"A sad and slow piano song"`
* **生成结果:** 一段完整的、结构工整的单声部旋律。
* **听感:** 听起来像老式的手机铃声或八音盒。旋律是连贯的，调性是正确的，但非常机械，缺乏情感力度变化 (Velocity)。
* **评价:** **有“形”无“神”**。结构完美，但缺乏表现力。

**对比 Discrete Diffusion Baseline:**

| 特性 | Discrete Diffusion (前次实验) | ChatMusician LLM (本次实验) |
| :--- | :--- | :--- |
| **结构完整性** | ❌ 极差 (80% 空白) | ✅ **完美 (0% 空白)** |
| **内容丰富度** | ❌ 极低 (0.09 notes/sec) | ✅ **正常 (3.53 notes/sec)** |
| **风格匹配度** | ❌ 随机噪声 | 🟡 **单一风格 (Folk/Traditional)** |
| **表现力** | ❌ 无 | 🟡 机械化/量化严重 |

---

## 问题与归因

**问题 1 (KL Divergence 异常高):**
* **原因:** **数据域偏差 (Data Domain Shift)**。
    * ChatMusician 的核心能力来自 ABC 文本数据，这类数据大多是单声部民歌。
    * 评测用的 MidiCaps 数据集包含复杂的和弦、多乐器织体。
    * **结论:** LLM 虽然生成了“正确的音乐”，但不是用户想要的“现代音乐风格”。

**问题 2 (Rhythmic Monotony):**
* **原因:** **中间表示 (Intermediate Representation) 的限制**。
    * ABC 记谱法本质上是高度量化的（Quantized）。它无法记录 REMI 格式中那种精细的 `Time Shift`。
    * **结论:** 文本符号（ABC）丢失了音乐的“演奏感”。

---

## 结论

**综合评价: ⭐⭐⭐ (3/5)** 这是一个**结构可用但风格受限**的基线。

1.  **强基线 (Strong Baseline):** 对于“生成可播放的音乐”这一任务，ChatMusician 是一个非常强的基线，它证明了自回归模型解决结构问题的能力。
2.  **Synesthesia 的机会:**
    * ChatMusician 的弱点（风格单一、缺乏表现力、单声部限制）正是 **Synesthesia (REMI + Diffusion + CLIP)** 的机会。
    * 你的 Synesthesia 模型可以使用 REMI 格式（支持多轨、力度、微时值）来解决表现力问题。
    * 利用 CLIP/CLAP 对齐来解决风格匹配问题（降低 KL 散度）。

**后续计划:**
* 将此结果记录在案，作为 **"LLM-based Approach"** 的代表性基线。
* 在论文中，ChatMusician 代表“结构正确但缺乏表现力的一端”，Discrete Diffusion 代表“缺乏结构的一端”。**Synesthesia 将结合二者之长。**
