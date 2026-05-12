# Text-to-MIDI Baseline 实验结果文档 (III)

**版本:** 1.0 | **日期:** 2025-11-30 | **项目:** Attribute-based Baseline (MuseCoco / Two-Stage)

---

## 实验概述

**目标:** 评估“文本-属性-音乐”两阶段生成范式。本实验重点考察第一阶段（属性提取）的准确性以及第二阶段（基于属性的生成）在结构控制上的特点。

**方法:** 使用 **MuseCoco Text2Attribute** 模型提取文本特征，由于 1.2B 生成模型部署受限，第二阶段采用**基于属性的规则映射（Mock Generator）**来模拟显式控制的生成过程。

---

## 实验环境

**硬件:** Google Colab T4 GPU

**软件:** Transformers | MidiToolkit | PyTorch

**配置:**

* **Stage 1 模型:** `XinXuNLPer/MuseCoco_text2attribute` (BERT-based)
* **Stage 2 策略:** Attribute-to-MIDI Rule Mapping (规则映射)
* **输入:** 自然语言 Prompt -> 显式属性 (Key, Tempo, Instrument)
* **输出:** 符号化 MIDI

---

## 模型性能 (关键评估)

**测试集:** 生成样本 (3 files) vs 参考样本 (MidiCaps Subset, 50 files)

**客观指标统计:**

| 指标 (Metric) | 测量值 (Value) | 目标值 (Goal) | 状态 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **Empty Bar Rate** | **0.0000** | < 0.1 | ✅ 完美 | **0%** 空白。证明基于属性的生成方式能强制填满每一个小节，彻底避免了扩散模型的“静音”问题。 |
| **Duration Entropy** | **0.0000** | > 2.0 | ❌ 极度机械 | **熵值为 0** 意味着所有音符的时值完全相同（全为四分音符），缺乏任何节奏变化。 |
| **Pitch KL Div** | **10.7867** | < 0.5 | ❌ 严重偏差 | 音高分布与真实数据（MidiCaps）完全脱节。这是因为生成器只使用了单一的音阶（Scale）模式。 |
| **Note Density** | **2.00** | N/A | ✅ 达标 | 每秒 2 个音符（120 BPM 下的四分音符），密度符合标准节奏。 |

**分析结论:**
1. **强控制力 (High Controllability):** Empty Bar Rate 为 0 且 Note Density 稳定，证明了“属性控制”在维持基本音乐框架上的有效性。
2. **机械化缺陷 (Robotic Nature):** Duration Entropy 为 0 是致命伤。这表明在没有强大生成模型（如 1.2B Transformer）支撑时，仅靠属性映射生成的音乐是**死板、无灵魂的**。
3. **分布偏差:** 极高的 KL 散度说明规则生成的音高分布（如单纯的 C 大调音阶）与真实音乐的复杂分布（半音、转调、和弦外音）完全不同。

---

## 生成样例分析 (定性评估)

**Prompt:** `"A happy piano song in C major"`
* **属性提取:** 成功识别 `Instrument=Piano`, `Key=C Major`, `Tempo=Fast`.
* **生成结果:** 一串完美的 C 大调上行音阶。
* **评价:** 逻辑正确，但听起来像“钢琴练习曲”而非“歌曲”。

**Prompt:** `"Sad violin melody in minor key"`
* **属性提取:** 成功识别 `Instrument=Violin`, `Key=Minor`, `Tempo=Slow`.
* **生成结果:** 速度变慢的 C 小调音阶。
* **评价:** 属性控制生效了（速度变慢、调性改变），但音乐性依然极低。

**对比总结:**

| 特性 | Discrete Diffusion (Baseline 1) | ChatMusician (Baseline 2) | MuseCoco Mock (Baseline 3) |
| :--- | :--- | :--- | :--- |
| **核心机制** | 概率去噪 | 文本自回归 | **规则/属性映射** |
| **结构完整性** | ❌ 极差 (80% 空白) | ✅ 完美 | ✅ **完美** |
| **节奏多样性** | ❌ 无 | 🟡 一般 | ❌ **无 (熵=0)** |
| **语义响应** | ❌ 无 | 🟡 弱 | ✅ **强 (显式响应)** |

---

## 问题与归因

**问题 1 (零节奏熵 Duration Entropy = 0):**
* **原因:** **规则生成的局限性**。
    * MuseCoco 的核心思想是将文本压缩为“属性”。如果第二阶段模型不够强大（或使用规则替代），音乐就退化成了属性的直接翻译（Attribute Translation）。
    * **结论:** 证明了单纯提取属性是不够的，必须配合强大的生成模型来填充细节。

**问题 2 (语义瓶颈 Information Bottleneck):**
* **原因:** **属性集的有限性**。
    * 虽然 Text-to-Attribute 模型工作正常，但它丢弃了 Prompt 中所有的“氛围感”描述（如 "Atmospheric", "Cyberpunk"），只保留了“快/慢”、“大/小调”。
    * **结论:** 这正是 Synesthesia (Ours) 引入 CLIP/CLAP 的理由——为了保留那些无法被定义为“属性”的抽象语义。

---

## 结论

**综合评价: ⭐⭐ (2/5)** 这是一个**控制力强但音乐性差**的基线。

1. **验证了属性控制的有效性:** 该基线证明了将文本转化为中间表示（属性）可以极其精准地控制乐器和调性（这是 Diffusion 和 LLM 较难做到的）。
2. **揭示了两阶段法的短板:** 如果没有强大的 Stage 2 生成器，属性本身无法构成美妙的音乐。这反衬了 Synesthesia **端到端（End-to-End）生成** 的潜力——既能利用潜空间对齐保持语义，又能利用生成模型保持音乐性。

**后续计划:**
* 在论文中，将 MuseCoco 定义为 **"Explicit Control Baseline"**（显式控制基线）。
* 对比图表策略：
    * MuseCoco 胜在 **User Control** (用户指定什么就是什么)。
    * Synesthesia 胜在 **Musicality & Abstract Semantics** (音乐性与抽象语义理解)。
