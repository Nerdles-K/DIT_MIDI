"""Text-to-MIDI 数据集测试脚本"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from main.data.dataset_midi import (
    MusicFeatureExtractor,
    create_text_to_midi_dataset,
    create_midi_dataloader
)


def test_feature_extraction():
    """测试音乐特征提取"""
    
    print("=" * 60)
    print("测试 1: 音乐特征提取")
    print("=" * 60)
    
    data_path = "midi_data/babyslakh/babyslakh_16k"
    
    if not os.path.exists(data_path):
        print(f"\n⚠ 数据集路径不存在: {data_path}")
        print("请先解压数据集:")
        print("  tar -xzf babyslakh_16k.tar.gz -C midi_data/babyslakh/")
        return False
    
    extractor = MusicFeatureExtractor()
    
    import glob
    midi_files = glob.glob(os.path.join(data_path, "**", "*.mid"), recursive=True)
    
    if not midi_files:
        print("\n⚠ 未找到 MIDI 文件")
        return False
    
    print(f"\n找到 {len(midi_files)} 个 MIDI 文件")
    print("\n提取前 3 个文件的特征:\n")
    
    for i, midi_path in enumerate(midi_files[:3]):
        print(f"[{i+1}] {os.path.basename(midi_path)}")
        features = extractor.extract_features(midi_path)
        
        if features:
            print(f"  [OK] 文本: \"{features['text_description']}\"")
            print(f"    乐器: {', '.join(features['instruments'])}")
            print(f"    速度: {features['tempo']:.1f} BPM ({features['tempo_category']})")
            print(f"    调性: {features['key']}")
        else:
            print(f"  [FAIL] 提取失败")
    
    return True


def test_dataset_creation():
    """Test Text-to-MIDI dataset creation."""
    
    print("\n" + "=" * 60)
    print("Testing Text-to-MIDI Dataset Creation")
    print("=" * 60)
    
    data_path = "midi_data/babyslakh/babyslakh_16k"
    
def test_dataset_creation():
    """测试数据集创建"""
    
    print("\n" + "=" * 60)
    print("测试 2: 数据集创建")
    print("=" * 60)
    
    data_path = "midi_data/babyslakh/babyslakh_16k"
    
    if not os.path.exists(data_path):
        print(f"\n⚠ 数据集路径不存在")
        return False
    
    try:
        print("\n创建数据集 (max_files=5)...")
        dataset = create_text_to_midi_dataset(
            path=data_path,
            max_files=5,
            use_cache=True
        )
        
        print(f"\n✓ 数据集创建成功! 共 {len(dataset)} 个样本")
        
        # 测试加载样本
        sample = dataset[0]
        print(f"\n样本示例:")
        print(f"  Patches 形状: {sample['patches'].shape}")
        print(f"  文本描述: \"{sample['text_description']}\"")
        print(f"  乐器: {sample['instruments']}")
        print(f"  速度: {sample['tempo']:.1f} BPM")
        print(f"  调性: {sample['key']}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    data_path = "midi_data/babyslakh/babyslakh_16k"
    
def test_dataloader():
    """测试 DataLoader"""
    
    print("\n" + "=" * 60)
    print("测试 3: DataLoader 批处理")
    print("=" * 60)
    
    data_path = "midi_data/babyslakh/babyslakh_16k"
    
    if not os.path.exists(data_path):
        print(f"\n⚠ 数据集路径不存在")
        return False
    
    try:
        dataset = create_text_to_midi_dataset(
            path=data_path,
            max_files=5,
            use_cache=True
        )
        
        dataloader = create_midi_dataloader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )
        
        print(f"\n✓ DataLoader 创建成功")
        
        batch = next(iter(dataloader))
        
        print(f"\n批次信息:")
        print(f"  Patches: {batch['patches'].shape}")
        print(f"  Masks: {batch['masks'].shape}")
        print(f"  Tempos: {batch['tempos']}")
        
        print(f"\n文本描述:")
        for i, text in enumerate(batch['text_descriptions']):
            print(f"  [{i}] \"{text}\"")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
def main():
    """运行所有测试"""
    
    print("\n" + "=" * 60)
    print("TEXT-TO-MIDI 数据集测试")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("特征提取", test_feature_extraction()))
    results.append(("数据集创建", test_dataset_creation()))
    results.append(("DataLoader", test_dataloader()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\n共 {total} 个测试，通过 {passed} 个")
    
    if passed == total:
        print("\n🎉 所有测试通过!")
    else:
        print("\n⚠ 部分测试失败，请检查上面的输出")


if __name__ == "__main__":
    main()