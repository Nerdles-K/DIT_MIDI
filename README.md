# Text-to-MIDI DiT 🎵

基于 DiT (Diffusion Transformer) + AdaLN-Zero 的文本驱动 MIDI 音乐生成系统

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.1+-blueviolet.svg)](https://lightning.ai/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## 🚀 快速开始

### 安装

```bash
conda create -n midi_dit python=3.10
conda activate midi_dit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning pretty-midi hydra-core omegaconf numpy scipy matplotlib wandb
```

### 训练

```bash
python train_midi.py +exp=train_babyslakh_midi_dit
```

### 生成

```bash
python scripts/inference_text_to_midi.py \
  --text "happy piano melody in C major" \
  --checkpoint logs/ckpts/*/best.ckpt \
  --output generated.mid
```

---

## ✨ 特性

- ✅ **DiT架构** - Transformer-based扩散模型
- ✅ **AdaLN-Zero** - 稳定的条件注入机制
- ✅ **文本控制** - 自然语言描述生成音乐
- ✅ **数据增强** - 6种MIDI增强技术
- ✅ **混合精度** - FP16训练,8GB显存可用
- ✅ **模块化** - 清晰的代码结构

---

## 📊 实验结果

| 版本 | 优化器 | 模型 | 数据增强 | Val Loss | 状态 |
|------|--------|------|----------|----------|------|
| v3.1 | AdamW | 125M | ❌ | **70.95** | ✅ 最佳 |
| v4.2 | Muon | 125M | ❌ | 101.19 | ❌ |
| v5.0 | Muon | 75M | ✅ medium | 171.54 | ❌ |
| **v5.1** | **AdamW** | **75M** | **✅ light** | **进行中** | 🔄 当前 |

详见: [`docs/`](docs/) 目录下的实验报告

---

## 📁 项目结构

```
DIT_MIDI/
├── main/              # 核心代码
│   ├── midi_dit.py           # DiT模型
│   ├── module_midi_dit.py    # Lightning模块
│   └── data/                 # 数据处理
├── scripts/           # 工具脚本
├── tests/             # 测试脚本
├── utils/             # 分析工具
├── docs/              # 文档
├── exp/               # 配置文件
└── train_midi.py      # 训练入口
```

---

## 📖 文档

- **快速入门**: [`docs/README_项目说明.md`](docs/README_项目说明.md) - 完整使用指南
- **项目需求**: [`docs/项目需求文档.md`](docs/项目需求文档.md)
- **实验报告**: [`docs/01-04_实验报告_*.md`](docs/)
- **改进建议**: [`docs/05_模型改进建议与优化方案.md`](docs/05_模型改进建议与优化方案.md)

---

## 🎯 示例

**输入文本:**
```
happy piano melody in C major
gentle guitar melody in G major
energetic drums pattern in D minor
```

**生成MIDI:** 对应风格的MIDI音乐文件

---

## 🛠️ 技术栈

- **PyTorch** 2.0+ - 深度学习框架
- **PyTorch Lightning** 2.1+ - 训练框架
- **Hydra** - 配置管理
- **pretty-midi** - MIDI处理
- **WandB** - 实验跟踪

---

## 📈 性能

**训练环境:** RTX 4060 Laptop 8GB + 32GB RAM

**训练速度:**
- 小数据集 (100文件): 2-4小时
- 中等数据集 (1000文件): 12-24小时
- 大数据集 (2000文件): 24-48小时

**生成速度:**
- 单个样本 (50步): ~5-10秒
- 批量生成: 可并行加速

---

## 🔧 系统要求

**最低配置:**
- GPU: GTX 1660 (6GB)
- 内存: 16GB
- 存储: 20GB

**推荐配置:**
- GPU: RTX 4060 (8GB+)
- 内存: 32GB
- 存储: 50GB+

---

## 📝 引用

如果本项目对您有帮助,请引用:

```bibtex
@misc{text2midi_dit_2025,
  title={Text-to-MIDI Generation with Diffusion Transformers},
  author={DDA4220 Course Group},
  year={2025}
}
```

---

## 📄 许可

本项目仅用于学术研究和教学目的。

---

## 🙏 致谢

- **DiT论文**: Peebles & Xie (2023)
- **DDPM论文**: Ho et al. (2020)
- **数据集**: Lakh MIDI Dataset
- **技术支持**: GitHub Copilot

---

## 📞 联系

**课程:** DDA4220 深度学习应用  
**更新:** 2025年11月30日

**详细文档**: 请查看 [`docs/README_项目说明.md`](docs/README_项目说明.md)

---

**祝您训练顺利! 🎵🎹🎸**
