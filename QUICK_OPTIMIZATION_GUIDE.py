"""
性能优化建议 - 立即可用的改进
==============================
"""

# 建议1: 在train_midi.py的第8行之后添加 (Tensor Core优化)
"""
import torch
torch.set_float32_matmul_precision('medium')  # 利用Tensor Cores,加速15-20%
"""

# 建议2: 减少num_workers (当前8太多,CPU竞争)
"""
在 exp/train_babyslakh_midi_dit.yaml 中:
datamodule:
  num_workers: 4  # 从8改为4
  # 可选添加:
  # persistent_workers: true
  # pin_memory: true
"""

# 建议3: 降低Muon学习率倍数 (解决Loss异常高问题)
"""
在 main/module_midi_dit.py configure_optimizers中:
muon_lr = self.hparams.lr * 1.0  # 从1.5改为1.0 (不做scaling)
或者
muon_lr = self.hparams.lr * 1.2  # 温和的1.2倍
"""

# 建议4: 减少Muon Newton-Schulz迭代次数
"""
在 main/module_midi_dit.py configure_optimizers中:
optimizer = MuonWithDecoupledWeightDecay(
    all_params,
    lr=muon_lr,
    momentum=0.95,
    nesterov=True,
    ns_steps=3,  # 从5改为3,每步快10-15%
    weight_decay=self.hparams.lr_weight_decay
)
"""

# 建议5: 减少checkpoint保存次数
"""
在 exp/train_babyslakh_midi_dit.yaml 中:
callbacks:
  model_checkpoint:
    save_top_k: 2  # 从3改为2
    every_n_epochs: 2  # 每2个epoch才考虑保存
"""

# 建议6: 降低验证频率 (训练期间专注训练)
"""
在 exp/train_babyslakh_midi_dit.yaml 中:
trainer:
  check_val_every_n_epoch: 2  # 从1改为2,节省20%时间
"""

print("=" * 60)
print("快速优化方案 (按优先级)")
print("=" * 60)
print("\n🔥 最高优先: 修复Loss异常高问题")
print("   当前v4.1 Loss=255→244 (应该在60-80)")
print("   建议: Muon lr倍数 1.5x → 1.0x")
print("   位置: main/module_midi_dit.py:239")
print("   改动: muon_lr = self.hparams.lr * 1.0")
print("\n⚡ 性能提升: 添加Tensor Core优化")
print("   建议: torch.set_float32_matmul_precision('medium')")
print("   位置: train_midi.py:第8行之后")
print("   收益: +15-20% 速度")
print("\n📊 效率提升: 减少num_workers")
print("   建议: num_workers: 8 → 4")
print("   位置: exp/train_babyslakh_midi_dit.yaml:66")
print("   收益: 减少CPU竞争")
print("\n💾 减少IO: 降低checkpoint频率")
print("   建议: save_top_k=2, every_n_epochs=2")
print("   位置: exp/train_babyslakh_midi_dit.yaml:80-81")
print("   收益: 减少磁盘写入")
print("\n⏱️ 训练专注: 降低验证频率")
print("   建议: check_val_every_n_epoch: 2")
print("   位置: exp/train_babyslakh_midi_dit.yaml:121")
print("   收益: +20% 训练速度")
print("=" * 60)
print("\n注意: 当前训练不要中断!")
print("等本次训练完成后再应用这些优化。")
print("=" * 60)
