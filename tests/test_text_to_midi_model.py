"""
测试 Text-to-MIDI 模型
====================

测试完整的 Text-to-MIDI 流程:
1. 文本编码器
2. DiT 模型与文本条件
3. 训练一个 step
"""

import torch
from main.midi_dit import create_midi_dit
from main.text_encoder import MUSIC_VOCAB


def test_text_encoder():
    """测试文本编码器"""
    print("\n=== 1. 测试文本编码器 ===")
    
    test_texts = [
        "piano happy melody in C",
        "guitar and violin gentle melody in G"
    ]
    
    print(f"测试文本: {test_texts}")
    
    encoded = MUSIC_VOCAB.encode_batch(test_texts, max_length=32)
    print(f"✓ Token IDs shape: {encoded['token_ids'].shape}")
    print(f"✓ Attention Mask shape: {encoded['attention_mask'].shape}")
    

def test_dit_with_text():
    """测试 DiT 模型与文本条件"""
    print("\n=== 2. 测试 DiT 模型 (带文本条件) ===")
    
    # 创建模型
    model = create_midi_dit(
        patch_dim=256,
        embed_dim=768,
        depth=4,  # 使用较小的模型测试
        num_heads=8,
        use_text_conditioning=True
    )
    
    print(f"✓ 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 准备输入
    B, N, D = 2, 128, 256
    x = torch.randn(B, N, D)
    t = torch.rand(B)
    class_labels = torch.randint(0, 10, (B,))
    text_descriptions = [
        "piano happy melody in C",
        "guitar gentle melody in G"
    ]
    
    # 前向传播
    with torch.no_grad():
        output = model(
            x, 
            t, 
            class_labels=class_labels,
            text_descriptions=text_descriptions
        )
    
    print(f"✓ 输入 shape: {x.shape}")
    print(f"✓ 输出 shape: {output.shape}")
    print(f"✓ 输出范围: [{output.min():.3f}, {output.max():.3f}]")
    assert output.shape == x.shape, "输出形状不匹配!"
    

def test_training_step():
    """测试训练 step"""
    print("\n=== 3. 测试训练 Step ===")
    
    from main.module_midi_dit import MIDIDiTLightningModule, DiffusionLoss
    
    # 创建模型
    model = MIDIDiTLightningModule(
        patch_dim=256,
        embed_dim=512,  # 更小的模型
        depth=4,
        num_heads=8,
        use_text_conditioning=True
    )
    
    print(f"✓ Lightning 模型创建成功")
    
    # 准备 batch
    B, N, D = 2, 128, 256
    batch = {
        'patches': torch.randn(B, N, D),
        'masks': torch.ones(B, N),
        'text_descriptions': [
            "piano happy melody in C",
            "drums energetic in A"
        ]
    }
    
    # 运行一个 training step
    loss = model.training_step(batch, 0)
    
    print(f"✓ Training loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss 应该为正数!"
    
    # 测试反向传播
    loss.backward()
    print(f"✓ 反向传播成功")
    

def test_without_text():
    """测试不使用文本条件的情况"""
    print("\n=== 4. 测试无文本条件模式 ===")
    
    # 创建不使用文本的模型
    model = create_midi_dit(
        patch_dim=256,
        embed_dim=512,
        depth=4,
        num_heads=8,
        use_text_conditioning=False
    )
    
    print(f"✓ 无文本模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 前向传播 (不提供文本)
    B, N, D = 2, 128, 256
    x = torch.randn(B, N, D)
    t = torch.rand(B)
    
    with torch.no_grad():
        output = model(x, t)
    
    print(f"✓ 无文本输出 shape: {output.shape}")
    

if __name__ == "__main__":
    print("=" * 60)
    print("Text-to-MIDI 模型测试")
    print("=" * 60)
    
    try:
        test_text_encoder()
        test_dit_with_text()
        test_training_step()
        test_without_text()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过!")
        print("=" * 60)
        print("\n下一步:")
        print("1. 准备 BabySlakh 数据集")
        print("2. 运行: python train_midi.py exp=train_babyslakh_midi_dit")
        print("3. 训练完成后测试生成")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
