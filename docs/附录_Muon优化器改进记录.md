# Muon优化器改进说明 (v4.1)

## 问题诊断 (v4.0失败原因)

### 1. 参数遗漏Bug (最严重)
```
v4.0代码问题:
- 定义了 muon_params (89个) 和 adamw_params (116个)
- 但最终只返回了 muon_optimizer
- 导致 116个参数 (56.6%) 从未被优化!
```

### 2. 学习率过高
```
v3.1 (AdamW): 实际lr = 2.448e-05
v4.0 (Muon):  实际lr = 8.208e-05 (3.36倍)
→ Loss = 219 vs 71 (3.09倍差距)
```

### 3. 训练不稳定
- Muon lr = base_lr × 3.0 太激进
- 大部分参数未更新导致模型退化

---

## v4.1 改进方案

### ✅ 修复1: 确保所有参数被优化
```python
# 所有参数统一用Muon优化器
all_params = muon_2d_params + other_params  # 205个参数全部包含
optimizer = MuonWithDecoupledWeightDecay(all_params, ...)
```

### ✅ 修复2: 降低学习率倍数
```python
# 从 3.0x 降低到 1.5x
muon_lr = base_lr × 1.5  # 6e-5 × 1.5 = 9e-5
```

**预期效果:**
- warmup后实际lr ≈ 4.5e-5 (vs v3.1的2.4e-5)
- 1.87倍提升,更温和的加速
- 预计loss在 60-75 范围内

### ✅ 修复3: 保持参数分组(仅用于监控)
```python
# 分组不影响优化,只是区分参数类型
muon_2d_params: 89个 (2D权重,受益于Newton-Schulz正交化)
other_params: 116个 (1D/embeddings,使用标准momentum)
```

---

## 预期结果对比

| 版本 | 优化器 | 参数数 | Base LR | 实际LR | 预期Val Loss | 状态 |
|------|--------|--------|---------|--------|--------------|------|
| v3.1 | AdamW  | 205    | 6e-5    | ~2.4e-5| 70.95        | ✅ 成功 |
| v4.0 | Muon   | 89 ❌  | 6e-5    | ~8.2e-5| 219.03       | ❌ 失败 |
| v4.1 | Muon   | 205 ✅ | 6e-5    | ~4.5e-5| 60-75 (预期) | 🔄 测试中 |

---

## 关键改进点

1. **参数完整性**: 205个参数全部优化 ✅
2. **学习率调整**: 1.5x倍数 (原3.0x) ✅
3. **稳定性增强**: 避免激进的lr scaling ✅
4. **监控透明**: 打印参数分组信息 ✅

---

## 训练建议

### 如果v4.1效果好 (val_loss < 71):
- 继续使用Muon,可尝试2.0x lr multiplier
- 加入Cosine Annealing LR
- 扩展到2000个训练文件

### 如果v4.1效果一般 (val_loss 71-80):
- Muon优势不明显,考虑回退AdamW
- 或尝试1.0x lr (不做scaling)

### 如果v4.1失败 (val_loss > 80):
- Muon与当前架构不兼容
- 彻底回退到AdamW
- 专注于数据增强和架构改进

---

## 下一步

**立即执行:**
```bash
conda activate midi_dit
python train_midi.py +exp=train_babyslakh_midi_dit
```

**监控指标:**
1. 参数数量是否为205个 ✅
2. 实际lr是否在4-5e-5范围
3. 前10个epoch的loss下降曲线
4. 是否出现NaN或梯度爆炸

**训练时长:** 预计2.5-3小时 (38-45 epochs)
