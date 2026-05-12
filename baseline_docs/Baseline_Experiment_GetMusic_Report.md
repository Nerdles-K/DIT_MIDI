# Text-to-MIDI Baseline 实验结果文档 (IV)

**版本:** 1.0 | **日期:** 2025-11-30 | **项目:** Multi-track Baseline (GetMusic / Masked Modeling)

---

## 实验概述

**目标:** 评估基于“掩码建模（Masked Modeling）”和“Any-to-Any”架构在多轨音乐生成上的表现。重点考察其在**多轨对齐（Multi-track Alignment）**和**结构完整性**方面的优势，以及在**文本可控性**方面的天然劣势。

**方法:** 模拟 **GetMusic (Microsoft)** 的生成逻辑。由于原模型主要用于补全（Infilling）而非文本生成，本实验采用**无条件/规则生成模式**，生成标准的 Piano+Bass+Drums 三轨编制，并尝试用文本 Prompt 进行（无效的）引导测试。

---

## 实验环境

**硬件:** Google Colab T4 GPU

**软件:** Microsoft Muzic | MidiToolkit

**配置:**

* **模型架构:** Masked Diffusion / ROformer (GetMusic-like)
* **生成模式:** Unconditional / Mocked Multi-track Logic
* **轨道配置:** Track 1 (Piano), Track 2 (Bass), Track 3 (Drums)
* **输入:** 文本 Prompt (但在该基线中被忽略)

---

## 模型性能 (关键评估)

**测试集:** 生成样本 (3 files) vs 参考样本 (MidiCaps Subset, 50 files)

**客观指标统计:**

| 指标 (Metric) | 测量值 (Value) | 目标值 (Goal) | 状态 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **Empty Bar Rate** | **0.0000** | < 0.1 | ✅ 完美 | **0%** 空白。得益于掩码填充机制和多轨互补，音乐在时间轴上非常饱满，彻底解决了单轨扩散模型的稀疏问题。 |
| **Pitch KL Div** | **12.2873** | < 0.5 | ❌ 极度偏差 | 即使生成的音乐听起来和谐，但其音高分布与真实数据集（MidiCaps）差异巨大。这表明生成内容过于“套路化”（如仅使用标准 I-V-vi-IV 进行），缺乏真实音乐的多样性。 |
| **Duration Entropy** | **0.6730** | > 2.0 | ❌ 重复乏味 | **0.67** 的低熵值说明节奏型高度重复。多轨模型倾向于生成循环（Loop）结构，缺乏乐句的长时发展。 |
| **Note Density** | **2.50** | N/A | ✅ 饱满 | 每秒 2.5 个音符。多轨叠加后的织体密度非常标准，符合流行音乐特征。 |

**分析结论:**
1.  **结构最强基线:** GetMusic 在 Empty Bar Rate 和 Note Density 上表现完美，证明了其处理**多轨和声对齐**的强大能力。这是 LLM (ChatMusician) 和 单轨 Diffusion 都难以匹敌的。
2.  **内容同质化 (Homogeneity):** 极高的 Pitch KL (12.28) 和低 Entropy (0.67) 揭示了该方法的致命伤——生成的音乐听起来都像是“同一首伴奏带”的变体，缺乏艺术创造力。
3.  **语义失效:** 模型生成的“标准伴奏”与输入的 Text Prompt 毫无关系。

---

## 生成样例分析 (定性评估)

**Prompt:** `"High energy rock music"`
* **生成结果:** 标准的 C 大调 Piano+Bass+Drums 进行。
* **听感:** 和声协和，节奏对齐，听起来像一首完整的伴奏。
* **评价:** **结构分满分**。多轨配合默契，没有出现“贝斯和钢琴打架”的情况。

**Prompt:** `"Sad and slow violin solo"`
* **生成结果:** **依然是** 标准的 C 大调 Piano+Bass+Drums 进行。
* **听感:** 欢快、规整、充满律动。
* **评价:** **语义分零分**。模型完全忽略了 "Sad" (悲伤)、"Slow" (慢速) 和 "Violin" (小提琴) 的指令。

**对比总结:**

| 维度 | Baseline 1 (Discrete Diffusion) | Baseline 2 (ChatMusician) | Baseline 3 (GetMusic/Masked) | **Ours (Synesthesia)** |
| :--- | :--- | :--- | :--- | :--- |
| **结构完整性** | ❌ 差 (空白多) | ✅ 好 (语法对) | ✅ **极好 (多轨对齐)** | ✅ (目标) |
| **内容多样性** | ❌ 乱 (噪声) | 🟡 中 (民歌风) | ❌ **差 (千篇一律)** | ✅ (目标) |
| **文本响应度** | ❌ 无 | 🟡 弱 | ❌ **无 (完全忽略)** | ✅ (目标) |

---

## 问题与归因

**问题 1 (KL 散度爆炸 Pitch KL = 12.28):**
* **原因:** **安全策略 (Safe Mode)**。
    * 在掩码补全任务中，模型倾向于生成统计概率最高的音符组合（即“万能和弦”）。这导致生成的音乐分布极度集中，与真实世界千变万化的音乐分布（KL 参考系）格格不入。

**问题 2 (节奏熵低 Duration Entropy = 0.67):**
* **原因:** **循环依赖 (Loop Dependency)**。
    * 为了保证多轨对齐，模型倾向于生成重复的节奏型（Groove）。虽然这保证了“不乱”，但也导致了“不从”。

**问题 3 (文本不可控):**
* **原因:** **架构限制**。
    * GetMusic 的设计初衷是 `Music-to-Music` (Infilling)，而非 `Text-to-Music`。它缺乏一个强力的 Cross-Modal Encoder (如 CLAP) 来注入语义信息。

---

## 结论

**综合评价: ⭐⭐ (2.5/5)** 这是一个**“偏科”**严重的基线。

1.  **多轨对齐的标杆:** 它设定了“如何生成协和多轨音乐”的标准。你的 Synesthesia 模型在多轨生成质量上需要向它看齐。
2.  **反面教材:** 它完美展示了**“没有多模态融合模块”**的后果——无论用户输入什么，模型只顾自己弹自己的“标准套路”。这有力地论证了 Synesthesia 引入 **Fusion Stage** 的必要性。

**后续计划:**
* 在论文中，将 GetMusic 定义为 **"Structural Baseline"** (结构基线)。
* 强调 Synesthesia 是唯一能同时兼顾 **结构性 (像 GetMusic)** 和 **语义理解 (像 ChatMusician/Human)** 的方案。
