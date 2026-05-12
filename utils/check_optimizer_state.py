"""
检查优化器状态和实际学习率
"""
import torch

print("=" * 60)
print("检查优化器配置")
print("=" * 60)

# v3.1 checkpoint
print("\n[v3.1 - AdamW]")
ckpt_v31 = torch.load("logs/ckpts/midi_dit_2025-11-29-16-22-50/epoch=33-val_loss=70.9454.ckpt", 
                      map_location='cpu', weights_only=False)

if 'optimizer_states' in ckpt_v31:
    opt_states = ckpt_v31['optimizer_states']
    print(f"Optimizer states: {len(opt_states)} optimizers")
    for i, opt_state in enumerate(opt_states):
        print(f"\nOptimizer {i}:")
        if 'param_groups' in opt_state:
            for j, pg in enumerate(opt_state['param_groups']):
                print(f"  Param group {j}:")
                print(f"    lr: {pg.get('lr', 'N/A')}")
                print(f"    weight_decay: {pg.get('weight_decay', 'N/A')}")
                print(f"    betas: {pg.get('betas', 'N/A')}")
                print(f"    params count: {len(pg.get('params', []))}")

# v4.0 checkpoint  
print("\n" + "=" * 60)
print("\n[v4.0 - Muon]")
ckpt_v40 = torch.load("logs/ckpts/midi_dit_2025-11-29-19-52-04/epoch=37-val_loss=219.0278.ckpt",
                      map_location='cpu', weights_only=False)

if 'optimizer_states' in ckpt_v40:
    opt_states = ckpt_v40['optimizer_states']
    print(f"Optimizer states: {len(opt_states)} optimizers")
    for i, opt_state in enumerate(opt_states):
        print(f"\nOptimizer {i}:")
        if 'param_groups' in opt_state:
            for j, pg in enumerate(opt_state['param_groups']):
                print(f"  Param group {j}:")
                print(f"    lr: {pg.get('lr', 'N/A')}")
                print(f"    weight_decay: {pg.get('weight_decay', 'N/A')}")
                print(f"    momentum: {pg.get('momentum', 'N/A')}")
                print(f"    nesterov: {pg.get('nesterov', 'N/A')}")
                print(f"    params count: {len(pg.get('params', []))}")

print("\n" + "=" * 60)
print("\n[关键发现]")
print(f"v3.1实际训练lr: {ckpt_v31['optimizer_states'][0]['param_groups'][0].get('lr', 'N/A')}")
print(f"v4.0实际训练lr: {ckpt_v40['optimizer_states'][0]['param_groups'][0].get('lr', 'N/A')}")
print(f"\nLoss比值: {219.0 / 70.9:.2f}x")
print("=" * 60)
