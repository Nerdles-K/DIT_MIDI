"""
检查v3.1和v4.0的loss差异
"""
import torch
import sys

# v3.1 checkpoints (AdamW)
v31_ckpts = [
    "logs/ckpts/midi_dit_2025-11-29-16-22-50/epoch=33-val_loss=70.9454.ckpt",
    "logs/ckpts/midi_dit_2025-11-29-16-22-50/epoch=25-val_loss=71.6976.ckpt",
]

# v4.0 checkpoints (Muon)
v40_ckpts = [
    "logs/ckpts/midi_dit_2025-11-29-19-52-04/epoch=37-val_loss=219.0278.ckpt",
    "logs/ckpts/midi_dit_2025-11-29-19-52-04/epoch=36-val_loss=220.8702.ckpt",
]

print("=" * 60)
print("检查Checkpoint Loss值")
print("=" * 60)

print("\n[v3.1 - AdamW Optimizer]")
for ckpt in v31_ckpts:
    try:
        data = torch.load(ckpt, map_location='cpu')
        if 'hyper_parameters' in data:
            hp = data['hyper_parameters']
            print(f"\n{ckpt.split('/')[-1]}")
            print(f"  patch_dim: {hp.get('patch_dim', 'N/A')}")
            print(f"  embed_dim: {hp.get('embed_dim', 'N/A')}")
            print(f"  sigma_min: {hp.get('sigma_min', 'N/A')}")
            print(f"  sigma_max: {hp.get('sigma_max', 'N/A')}")
            print(f"  lr: {hp.get('lr', 'N/A')}")
        
        # 检查callbacks中的best_model_score
        if 'callbacks' in data:
            for key, val in data['callbacks'].items():
                if hasattr(val, 'best_model_score'):
                    print(f"  best_model_score: {val.best_model_score}")
                    
    except Exception as e:
        print(f"  Error loading {ckpt}: {e}")

print("\n" + "=" * 60)
print("\n[v4.0 - Muon Optimizer]")
for ckpt in v40_ckpts:
    try:
        data = torch.load(ckpt, map_location='cpu')
        if 'hyper_parameters' in data:
            hp = data['hyper_parameters']
            print(f"\n{ckpt.split('/')[-1]}")
            print(f"  patch_dim: {hp.get('patch_dim', 'N/A')}")
            print(f"  embed_dim: {hp.get('embed_dim', 'N/A')}")
            print(f"  sigma_min: {hp.get('sigma_min', 'N/A')}")
            print(f"  sigma_max: {hp.get('sigma_max', 'N/A')}")
            print(f"  lr: {hp.get('lr', 'N/A')}")
        
        # 检查callbacks中的best_model_score
        if 'callbacks' in data:
            for key, val in data['callbacks'].items():
                if hasattr(val, 'best_model_score'):
                    print(f"  best_model_score: {val.best_model_score}")
                    
    except Exception as e:
        print(f"  Error loading {ckpt}: {e}")

print("\n" + "=" * 60)
print("\n[分析]")
print("Loss值差异过大 (70 vs 219),可能原因:")
print("1. Muon优化器学习率scaling (lr × 3.0 = 1.8e-4) 导致训练不稳定")
print("2. 参数分组逻辑问题 (89个Muon参数 vs 116个AdamW参数)")
print("3. Diffusion loss scale计算错误")
print("4. Mixed precision (FP16) 与Muon交互问题")
print("\n建议:")
print("- 降低Muon学习率倍数 (3.0 → 1.5 或 2.0)")
print("- 或者直接使用AdamW学习率 (不做scaling)")
print("- 检查Newton-Schulz正交化在FP16下的数值稳定性")
print("=" * 60)
