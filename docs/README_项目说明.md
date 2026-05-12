# Text-to-MIDI DiT 项目说明文档

**项目名称:** Text-to-MIDI 音乐生成系统  
**版本:** v5.1  
**最后更新:** 2025-11-30  
**课程:** DDA4220 深度学习应用

---

## 📋 目录

- [项目简介](#项目简介)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [实验记录](#实验记录)
- [使用指南](#使用指南)
- [配置说明](#配置说明)
- [技术架构](#技术架构)
- [数据增强](#数据增强)
- [常见问题](#常见问题)
- [故障排除](#故障排除)
- [性能优化](#性能优化)

---

## 项目简介

基于 **DiT (Diffusion Transformer)** + **AdaLN-Zero** 的文本驱动 MIDI 音乐生成系统。通过自然语言描述生成对应风格的MIDI音乐。

### 核心特性

✅ **DiT架构**: 基于Transformer的扩散模型,生成质量优异  
✅ **AdaLN-Zero**: 零初始化自适应层归一化,训练稳定  
✅ **文本条件生成**: 支持自然语言描述控制音乐风格  
✅ **数据增强**: 6种MIDI增强技术,提升泛化能力  
✅ **混合精度训练**: FP16加速,8GB显存可训练  
✅ **模块化设计**: 清晰的代码结构,易于扩展

### 应用示例

**输入文本:** `"happy piano melody in C major"`  
**输出:** C大调钢琴旋律 MIDI 文件

**输入文本:** `"gentle guitar melody in G major"`  
**输出:** G大调吉他柔和旋律 MIDI 文件

---

## 项目结构

```
DIT_MIDI/
├── main/                           # 核心代码模块
│   ├── midi_dit.py                # DiT 模型架构定义
│   ├── text_encoder.py            # 文本编码器实现
│   ├── module_midi_dit.py         # PyTorch Lightning 训练模块
│   ├── muon_optimizer.py          # Muon 优化器 (实验性)
│   └── data/
│       ├── __init__.py
│       ├── dataset_midi.py        # MIDI 数据集类
│       └── data_augmentation.py   # 数据增强模块 (v5.0新增)
│
├── exp/                            # 实验配置
│   └── train_babyslakh_midi_dit.yaml  # 训练超参数配置
│
├── scripts/                        # 工具脚本
│   ├── batch_generate_midi.py     # 批量生成MIDI
│   └── inference_text_to_midi.py  # 单个推理生成
│
├── tests/                          # 测试脚本
│   ├── test_text_to_midi_dataset.py    # 数据集测试
│   ├── test_text_to_midi_model.py      # 模型测试
│   ├── test_data_augmentation.py       # 数据增强测试
│   └── test_muon_optimizer.py          # 优化器测试
│
├── utils/                          # 分析工具
│   ├── analyze_model_data_ratio.py      # 模型-数据比分析
│   ├── check_loss_difference.py         # 损失对比
│   └── performance_optimization_analysis.py  # 性能分析
│
├── docs/                           # 项目文档
│   ├── 01_实验报告_v3.0_初始版本_2025-11-28.md
│   ├── 02_实验报告_v3.1_AdamW优化_2025-11-29.md
│   ├── 03_实验报告_v4.2_Muon优化器_2025-11-30.md
│   ├── 04_实验报告_v5.0_小模型与数据增强_2025-11-30.md
│   ├── 05_模型改进建议与优化方案.md
│   ├── 附录_Muon优化器说明.md
│   ├── 附录_Muon优化器改进记录.md
│   ├── README_项目说明.md (本文件)
│   ├── 项目说明文档.md (旧版)
│   └── 项目需求文档.md
│
├── archived/                       # 历史版本文件
│   ├── 实验报告_2025-11-29.md
│   └── 实验报告_改进版_2025-11-29.md
│
├── logs/                           # 训练日志
│   ├── ckpts/                      # 模型检查点
│   │   ├── midi_dit_2025-11-29-*/  # v3.1检查点
│   │   ├── midi_dit_2025-11-30-*/  # v4.2/v5.0检查点
│   │   └── ...
│   ├── runs/                       # TensorBoard日志
│   └── wandb/                      # WandB离线日志
│
├── midi_data/                      # 数据集目录
│   └── lmd_full/                   # Lakh MIDI Dataset (2000文件)
│
├── train_midi.py                   # 主训练脚本
└── README.md                       # 项目根README (待创建)
```

---

## 环境配置

### 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | GTX 1660 (6GB) | RTX 4060 (8GB+) |
| 内存 | 16GB | 32GB |
| 存储 | 20GB | 50GB+ |
| CPU | 4核 | 8核+ |

**当前开发环境:** RTX 4060 Laptop 8GB + 32GB RAM

### 软件依赖

**Python版本:** 3.10+

**核心依赖:**
- PyTorch 2.0+ (CUDA 11.8)
- PyTorch Lightning 2.1+
- pretty-midi 0.2.10+
- Hydra-core 1.3+
- NumPy, SciPy, Matplotlib

### 安装步骤

```powershell
# 1. 创建Conda环境
conda create -n midi_dit python=3.10
conda activate midi_dit

# 2. 安装PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安装其他依赖
pip install pytorch-lightning==2.1.0
pip install pretty-midi hydra-core omegaconf
pip install numpy scipy matplotlib wandb

# 4. 验证CUDA可用
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**预期输出:**
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

---

## 快速开始

### 1. 准备数据集

确保数据集路径正确:

```powershell
# 检查数据集
ls midi_data/lmd_full/*.mid | Measure-Object | Select-Object Count

# 预期: 至少10个MIDI文件
```

**数据集说明:**
- **当前使用:** Lakh MIDI Dataset (lmd_full, 2000文件)
- **其他选择:** Slakh2100, MAESTRO, 自定义MIDI集合

### 2. 验证环境

运行测试脚本确认环境正常:

```powershell
# 测试数据集加载
python tests/test_text_to_midi_dataset.py

# 测试模型初始化
python tests/test_text_to_midi_model.py

# 测试数据增强 (v5.0+)
python tests/test_data_augmentation.py
```

**预期输出:** 所有测试通过,无报错

### 3. 开始训练

**使用默认配置 (v5.1 - AdamW + 小模型 + 数据增强):**

```powershell
python train_midi.py +exp=train_babyslakh_midi_dit
```

**自定义配置:**

```powershell
# 调整批大小和学习率
python train_midi.py +exp=train_babyslakh_midi_dit \
  datamodule.batch_size=8 \
  model.lr=3e-5

# 减小模型规模 (内存不足时)
python train_midi.py +exp=train_babyslakh_midi_dit \
  model.embed_dim=384 \
  model.depth=8 \
  model.num_heads=6 \
  datamodule.batch_size=12

# 快速测试 (过拟合小批数据)
python train_midi.py +exp=train_babyslakh_midi_dit \
  trainer.fast_dev_run=10
```

**训练监控:**
- 终端输出: 实时loss和进度
- TensorBoard: `tensorboard --logdir logs/runs`
- WandB: 离线模式,可同步到云端

### 4. 生成MIDI

**单个生成:**

```powershell
# 使用最佳检查点
python scripts/inference_text_to_midi.py \
  --text "happy piano melody in C major" \
  --checkpoint logs/ckpts/midi_dit_2025-11-30-*/epoch=*-val_loss=*.ckpt \
  --output output.mid \
  --num-steps 50 \
  --temperature 1.0

# 调整参数
python scripts/inference_text_to_midi.py \
  --text "gentle guitar melody in G major" \
  --checkpoint logs/ckpts/*/best.ckpt \
  --output gentle_guitar.mid \
  --num-steps 100 \
  --temperature 0.8 \
  --duration 15 \
  --tempo 90
```

**批量生成:**

```powershell
# 使用内置示例
python scripts/batch_generate_midi.py \
  --checkpoint logs/ckpts/*/best.ckpt \
  --use-examples \
  --output-dir generated_midi

# 使用自定义文本文件
python scripts/batch_generate_midi.py \
  --checkpoint logs/ckpts/*/best.ckpt \
  --text-file my_prompts.txt \
  --output-dir my_outputs \
  --num-steps 75
```

---

## 实验记录

### 版本历史

| 版本 | 日期 | 优化器 | 模型规模 | 数据增强 | Best Val Loss | 状态 |
|------|------|--------|----------|----------|---------------|------|
| v3.0 | 2025-11-28 | AdamW | 125M | ❌ | ~90 | 基线 |
| v3.1 | 2025-11-29 | AdamW | 125M | ❌ | **70.95** | ✅ 最佳 |
| v4.0 | 2025-11-30 | Muon | 125M | ❌ | ~120 | ❌ 失败 |
| v4.1 | 2025-11-30 | Muon | 125M | ❌ | ~110 | ❌ 失败 |
| v4.2 | 2025-11-30 | Muon | 125M | ❌ | 101.19 | ❌ 失败 |
| v5.0 | 2025-11-30 | Muon | 75M | ✅ medium | 171.54 | ❌ 失败 |
| **v5.1** | 2025-11-30 | **AdamW** | **75M** | **✅ light** | **进行中** | 🔄 当前 |

### 关键发现

1. **AdamW > Muon**: 所有Muon实验 (v4.0-v5.0) 均未超越AdamW基线
2. **数据-参数匹配**: v5.0参数量减少40%,数据量翻倍,改善匹配度
3. **数据增强有效**: 集成6种增强技术,训练稳定
4. **温和增强更优**: v5.1降低增强强度为light,避免过度扰动

### 推荐配置

**当前最优 (v5.1):**
```yaml
优化器: AdamW (lr=6e-5, weight_decay=0.03)
模型: embed_dim=512, depth=10, num_heads=8 (~75M参数)
数据: 2000文件, batch_size=16, 数据增强light
训练: FP16, Early Stopping (patience=12)
```

**未来方向:**
- v5.2: 优化学习率调度
- v5.3: 扩展数据至5000文件
- v5.4: 探索更大模型 (embed_dim=576-640)

详见: `docs/05_模型改进建议与优化方案.md`

---

## 使用指南

### 文本描述格式

**推荐格式:** `{情绪} {乐器} melody in {调性}`

**示例:**
```
happy piano melody in C major
gentle guitar melody in G major
energetic drums pattern in D minor
calm violin melody in F major
powerful bass line in A minor
```

### 支持的词汇

**乐器 (Instrument):**
- 键盘: piano, keyboard, organ
- 弦乐: guitar, violin, cello, strings
- 贝斯: bass, double bass
- 打击: drums, percussion

**情绪 (Mood):**
- calm (平静)
- happy (快乐)
- sad (悲伤)
- energetic (充满活力)
- gentle (温柔)
- powerful (有力)
- expressive (富有表现力)

**强度 (Intensity):**
- soft (柔和)
- moderate (中等)
- loud (响亮)

**调性 (Key):**
- 大调: C, D, E, F, G, A, B major
- 小调: C, D, E, F, G, A, B minor
- 变调: C#, Db, F#, Gb, etc.

### 生成参数

| 参数 | 默认值 | 范围 | 说明 |
|------|-------|------|------|
| `--num-steps` | 50 | 25-250 | 去噪步数,越大质量越好但越慢 |
| `--temperature` | 1.0 | 0.5-2.0 | 采样温度,越高越随机 |
| `--duration` | 10 | 5-30 | 生成时长(秒) |
| `--tempo` | 120 | 60-180 | 速度(BPM) |
| `--velocity` | 80 | 40-127 | 力度 |

**推荐设置:**
- 高质量: `--num-steps 100 --temperature 0.8`
- 快速预览: `--num-steps 25 --temperature 1.0`
- 多样性: `--num-steps 50 --temperature 1.5`

---

## 配置说明

### 配置文件

主配置文件: `exp/train_babyslakh_midi_dit.yaml`

### 关键参数

**模型配置 (model):**
```yaml
embed_dim: 512        # 嵌入维度 (384/512/640/768)
depth: 10             # Transformer层数 (8/10/12/18)
num_heads: 8          # 注意力头数 (需能整除embed_dim)
dropout: 0.15         # Dropout率 (0.1-0.2)
lr: 6e-5              # 学习率 (3e-5到1e-4)
lr_weight_decay: 0.03 # 权重衰减 (0.01-0.05)
```

**数据配置 (datamodule):**
```yaml
batch_size: 16               # 批大小 (4/8/12/16)
max_files: 2000              # 最大文件数
use_augmentation: True       # 启用数据增强
augmentation_strength: light # 增强强度 (light/medium/heavy)
val_split: 0.15              # 验证集比例
```

**训练配置 (trainer):**
```yaml
max_epochs: 100              # 最大轮数
precision: 16                # 混合精度 (16/32)
accumulate_grad_batches: 1   # 梯度累积
gradient_clip_val: 0.5       # 梯度裁剪
```

### 命令行覆盖

```powershell
# 修改单个参数
python train_midi.py +exp=train_babyslakh_midi_dit model.lr=3e-5

# 修改多个参数
python train_midi.py +exp=train_babyslakh_midi_dit \
  model.embed_dim=384 \
  model.depth=8 \
  datamodule.batch_size=12 \
  trainer.max_epochs=50
```

---

## 技术架构

### 整体流程

```
训练阶段:
MIDI文件 → PianoRoll表示 → Patchify → 特征提取 → 文本描述 → 配对训练

生成阶段:
文本描述 → 文本编码 → DiT去噪 → Unpatchify → PianoRoll → MIDI文件
```

### 核心组件

**1. 文本编码器 (Text Encoder)**
- 架构: 4层Transformer
- 词汇量: 56个词
- 输出维度: 768维
- 参数量: ~29M

**2. DiT模型 (Diffusion Transformer)**
- 架构: 10层 Transformer + AdaLN-Zero
- 嵌入维度: 512
- 注意力头: 8个
- 参数量: ~75M (v5.1) 或 125M (v3.1)

**3. AdaLN-Zero**
- 条件注入: 通过零初始化门控
- 稳定训练: 避免早期训练不稳定
- 灵活控制: 支持文本和类别条件

**4. 数据增强 (v5.0+)**
- Pitch Shift: 音高平移 ±6半音
- Time Stretch: 时间拉伸 0.9-1.1x
- Velocity Scaling: 力度缩放 0.8-1.2x
- Note Dropout: 音符丢弃 5%
- Random Crop: 随机裁剪
- Gaussian Noise: 高斯噪声

### 数据处理

**MIDI → 特征:**
1. 加载MIDI文件 (pretty-midi)
2. 转换为PianoRoll表示 (16x16量化)
3. Patchify (16×16 patches)
4. 归一化到 [-1, 1]

**特征 → MIDI:**
1. 去归一化
2. Unpatchify重建PianoRoll
3. 转换为MIDI事件
4. 保存为.mid文件

---

## 数据增强

### 增强技术 (v5.0引入)

| 技术 | 参数范围 | 应用概率 | 效果 |
|------|---------|----------|------|
| Pitch Shift | ±6半音 | 50% | 变调不变风格 |
| Time Stretch | 0.9-1.1x | 30% | 速度变化 |
| Velocity Scaling | 0.8-1.2x | 40% | 力度变化 |
| Note Dropout | 5% | 20% | 增强鲁棒性 |
| Random Crop | 0.8-1.0x | 10% | 长度变化 |
| Gaussian Noise | σ=0.02 | 总是 | 轻微扰动 |

### 增强强度

**light (轻度):** 适合高质量数据
```python
pitch_shift_range = (-3, 3)
time_stretch_range = (0.95, 1.05)
velocity_scale_range = (0.9, 1.1)
note_dropout_prob = 0.02
apply_prob = 0.5
```

**medium (中度):** 平衡质量和多样性
```python
pitch_shift_range = (-6, 6)
time_stretch_range = (0.9, 1.1)
velocity_scale_range = (0.8, 1.2)
note_dropout_prob = 0.05
apply_prob = 0.7
```

**heavy (重度):** 数据稀缺时
```python
pitch_shift_range = (-12, 12)
time_stretch_range = (0.8, 1.2)
velocity_scale_range = (0.7, 1.3)
note_dropout_prob = 0.1
apply_prob = 0.8
```

### 使用建议

- **数据充足 (>5000文件)**: 使用light或关闭增强
- **数据中等 (1000-5000文件)**: 使用light (推荐)
- **数据稀缺 (<1000文件)**: 使用medium

**当前配置 (v5.1):** light强度,2000文件

---

## 常见问题

### Q1: 数据集从哪里获取?

**A:** 
- **Lakh MIDI Dataset**: https://colinraffel.com/projects/lmd/
- **Slakh2100**: https://zenodo.org/record/4599666
- **MAESTRO**: https://magenta.tensorflow.org/datasets/maestro
- **自定义**: 任意10+个MIDI文件

### Q2: 训练需要多长时间?

**A:**
- 小数据集 (100文件): 2-4小时
- 中等数据集 (1000文件): 12-24小时
- 大数据集 (2000文件): 24-48小时

*基于 RTX 4060 Laptop 8GB*

### Q3: 如何处理CUDA内存不足?

**A:** 按优先级尝试:
1. 减小batch_size: `datamodule.batch_size=8`
2. 减小模型: `model.embed_dim=384 model.depth=8`
3. 关闭数据增强: `datamodule.use_augmentation=False`
4. 使用FP16: `trainer.precision=16` (默认)

### Q4: 生成的MIDI质量不好怎么办?

**A:**
1. 增加去噪步数: `--num-steps 100`
2. 调整温度: `--temperature 0.8` (降低随机性)
3. 使用更好的检查点 (更低val_loss)
4. 延长训练时间
5. 扩展训练数据集

### Q5: 如何查看训练进度?

**A:**
- **终端输出**: 实时显示loss和进度条
- **TensorBoard**: `tensorboard --logdir logs/runs`
- **WandB**: 离线日志在 `logs/wandb/`
- **检查点**: 自动保存最佳2个模型

### Q6: 可以用CPU训练吗?

**A:** 理论可以,但极慢(100倍+),强烈不推荐。建议使用Google Colab免费GPU。

### Q7: 支持多GPU训练吗?

**A:** 支持!修改配置:
```powershell
python train_midi.py +exp=train_babyslakh_midi_dit trainer.devices=2
```

### Q8: 为什么Muon优化器失败了?

**A:** 经过v4.0-v5.0四次实验,Muon在此任务上表现不佳。可能原因:
- 音乐生成任务特性
- 模型架构不匹配
- 超参数未充分调优

推荐使用AdamW (v3.1/v5.1)。

---

## 故障排除

### 问题1: CUDA内存不足

**错误信息:**
```
RuntimeError: CUDA out of memory
```

**解决方案:**
```powershell
# 方案A: 减小batch size
python train_midi.py +exp=train_babyslakh_midi_dit datamodule.batch_size=8

# 方案B: 减小模型
python train_midi.py +exp=train_babyslakh_midi_dit \
  model.embed_dim=384 \
  model.depth=8 \
  model.num_heads=6

# 方案C: 启用梯度累积
python train_midi.py +exp=train_babyslakh_midi_dit \
  datamodule.batch_size=8 \
  trainer.accumulate_grad_batches=2
```

### 问题2: 找不到数据集

**错误信息:**
```
FileNotFoundError: midi_data/lmd_full not found
```

**解决方案:**
```powershell
# 检查路径
ls midi_data/

# 修改配置文件中的data_path
# 或使用命令行覆盖
python train_midi.py +exp=train_babyslakh_midi_dit \
  datamodule.data_path="your/path/to/midi"
```

### 问题3: 损失不下降

**症状:** Loss在初始值附近震荡,不收敛

**解决方案:**

1. 检查学习率:
```powershell
python train_midi.py +exp=train_babyslakh_midi_dit model.lr=3e-5
```

2. 增加warmup步数:
```powershell
python train_midi.py +exp=train_babyslakh_midi_dit model.lr_warmup_steps=3000
```

3. 验证数据:
```powershell
python tests/test_text_to_midi_dataset.py
```

4. 检查梯度:
```powershell
# 在module_midi_dit.py的training_step中添加:
print(f"Grad norm: {torch.nn.utils.clip_grad_norm_(self.parameters(), float('inf'))}")
```

### 问题4: 生成无音符

**症状:** 生成的MIDI文件为空或只有噪音

**解决方案:**

1. 增加去噪步数:
```powershell
python scripts/inference_text_to_midi.py \
  --num-steps 100 \
  --temperature 1.5
```

2. 尝试其他检查点:
```powershell
# 使用不同epoch的模型
python scripts/inference_text_to_midi.py \
  --checkpoint logs/ckpts/*/epoch=XX-*.ckpt
```

3. 检查文本描述:
```powershell
# 使用简单明确的描述
python scripts/inference_text_to_midi.py \
  --text "piano melody"
```

### 问题5: Windows Unicode错误

**错误信息:**
```
UnicodeEncodeError: 'charmap' codec can't encode character
```

**解决方案:**
```powershell
$env:PYTHONIOENCODING="utf-8"
python train_midi.py +exp=train_babyslakh_midi_dit
```

### 问题6: WandB连接失败

**错误信息:**
```
wandb: ERROR Unable to connect
```

**解决方案:**
配置文件已设置offline模式,无需联网。如需同步:
```powershell
wandb online
wandb sync logs/wandb/offline-run-*
```

---

## 性能优化

### 训练加速

**1. 混合精度训练 (已启用)**
```yaml
trainer:
  precision: 16  # FP16, 约2x加速
```

**2. 多GPU训练**
```powershell
python train_midi.py +exp=train_babyslakh_midi_dit trainer.devices=2
```

**3. 增加DataLoader线程**
```yaml
datamodule:
  num_workers: 8  # 根据CPU核心数调整
```

**4. 启用CuDNN benchmark**
```yaml
trainer:
  benchmark: True  # 已启用
```

**5. 梯度累积 (内存不足时)**
```yaml
trainer:
  accumulate_grad_batches: 2  # 模拟2x batch size
```

### 生成加速

**1. 减少去噪步数**
```powershell
# 从50减到25,速度提升2x,质量轻微下降
python scripts/inference_text_to_midi.py --num-steps 25
```

**2. 批量生成**
```powershell
# 一次生成多个样本
python scripts/batch_generate_midi.py \
  --checkpoint logs/ckpts/*/best.ckpt \
  --text-file prompts.txt \
  --batch-size 4
```

**3. 使用编译模型 (PyTorch 2.0+)**
```python
# 在module_midi_dit.py中添加:
self.model = torch.compile(self.model)
```

### 质量提升

**1. 扩展训练数据**
```yaml
datamodule:
  max_files: 5000  # 更多数据,更好泛化
```

**2. 延长训练时间**
```yaml
trainer:
  max_epochs: 200  # 充分训练
```

**3. 增加模型容量**
```yaml
model:
  embed_dim: 640  # 从512增加
  depth: 12       # 从10增加
```

**4. 优化数据增强**
```yaml
datamodule:
  use_augmentation: True
  augmentation_strength: light  # 温和增强更优
```

**5. 增加生成步数**
```powershell
python scripts/inference_text_to_midi.py --num-steps 100
```

### 内存优化

**1. 梯度检查点 (Gradient Checkpointing)**
```python
# 牺牲速度换内存,可节省30-50%
# 在midi_dit.py的DiTBlock中启用
```

**2. 减小序列长度**
```yaml
model:
  max_seq_len: 512  # 从1024减少
```

**3. 减小patch size**
```yaml
patch_size: [8, 8]  # 从[16, 16]减小,但会增加序列长度
```

---

## 相关资源

### 项目文档

- **实验报告**: `docs/01-04_实验报告_v*.md` - 详细实验记录
- **改进建议**: `docs/05_模型改进建议与优化方案.md`
- **需求文档**: `docs/项目需求文档.md`
- **Muon说明**: `docs/附录_Muon优化器*.md`

### 学术论文

- **DiT**: Peebles & Xie (2023) - "Scalable Diffusion Models with Transformers"
- **DDPM**: Ho et al. (2020) - "Denoising Diffusion Probabilistic Models"
- **AdaLN**: Dhariwal & Nichol (2021) - "Diffusion Models Beat GANs"

### 数据集

- **Lakh MIDI**: https://colinraffel.com/projects/lmd/
- **Slakh2100**: https://zenodo.org/record/4599666
- **MAESTRO**: https://magenta.tensorflow.org/datasets/maestro

### 工具与框架

- **PyTorch**: https://pytorch.org/
- **Lightning**: https://lightning.ai/
- **pretty-midi**: https://github.com/craffel/pretty-midi
- **Hydra**: https://hydra.cc/

### 相关项目

- **Magenta**: https://magenta.tensorflow.org/
- **MuseGAN**: https://salu133445.github.io/musegan/
- **Music Transformer**: https://magenta.tensorflow.org/music-transformer

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v1.0 | 2025-11-26 | 初始版本,基础功能 |
| v2.0 | 2025-11-28 | 完善文档,添加示例 |
| v3.0 | 2025-11-28 | 实验v3.0,建立基线 |
| v3.1 | 2025-11-29 | AdamW优化,val_loss=70.95 |
| v4.2 | 2025-11-30 | Muon优化器实验 (失败) |
| v5.0 | 2025-11-30 | 小模型+数据增强 (失败) |
| **v5.1** | 2025-11-30 | **AdamW+小模型+light增强 (当前)** |

---

## 贡献者

**开发团队:** DDA4220 课程组  
**技术支持:** GitHub Copilot  
**课程:** 深度学习应用

---

## 许可证

本项目仅用于学术研究和教学目的。

---

**最后更新:** 2025年11月30日  
**文档版本:** v5.1  
**维护者:** DDA4220课程组

---

## 联系方式

如有问题或建议,请通过以下方式联系:
- 课程邮箱: (待补充)
- 项目Issues: (待补充)

---

**祝您训练顺利! 🎵🎹🎸**
