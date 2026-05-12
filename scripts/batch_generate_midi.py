"""
批量生成 MIDI 脚本
================

从文本列表批量生成 MIDI 文件。
"""

import os
import sys
import argparse
from typing import List, Dict
from pathlib import Path

# ensure project root is on PYTHONPATH when script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm

from scripts.inference_text_to_midi import TextToMIDIGenerator


# 预定义的测试文本
EXAMPLE_TEXTS = [
    # Piano
    "piano happy melody in C",
    "piano gentle melody in G",
    "piano sad melody in A minor",
    "piano powerful melody in D",
    
    # Guitar
    "guitar calm melody in E",
    "guitar energetic in A",
    "guitar and violin gentle melody in C",
    
    # Drums
    "drums powerful energetic in C",
    "drums fast in A",
    
    # Multi-instrument
    "piano and guitar happy melody in C",
    "violin expressive melody in G",
    "strings calm melody in F",
]


def batch_generate(
    checkpoint_path: str,
    texts: List[str],
    output_dir: str = "generated_midi",
    duration: float = 10.0,
    num_steps: int = 50,
    temperature: float = 1.0,
    tempo: int = 120,
    device: str = 'auto'
):
    """
    批量生成 MIDI 文件
    
    Args:
        checkpoint_path: 模型检查点路径
        texts: 文本描述列表
        output_dir: 输出目录
        duration: 生成时长
        num_steps: 采样步数
        temperature: 采样温度
        tempo: BPM
        device: 设备
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定设备
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建生成器
    print(f"初始化生成器...")
    generator = TextToMIDIGenerator(
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    # 批量生成
    print(f"\n开始批量生成 {len(texts)} 个 MIDI 文件...")
    print(f"输出目录: {output_dir}\n")
    
    results = []
    
    for i, text in enumerate(tqdm(texts, desc="生成进度")):
        try:
            # 生成文件名
            filename = f"generated_{i:03d}.mid"
            output_path = os.path.join(output_dir, filename)
            
            # 生成 MIDI
            generator.generate_midi(
                text=text,
                output_path=output_path,
                duration=duration,
                num_steps=num_steps,
                temperature=temperature,
                tempo=tempo
            )
            
            results.append({
                'index': i,
                'text': text,
                'output': output_path,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"\n❌ 生成失败 (索引 {i}): {text}")
            print(f"   错误: {e}")
            results.append({
                'index': i,
                'text': text,
                'output': None,
                'status': 'failed',
                'error': str(e)
            })
    
    # 保存结果摘要
    summary_path = os.path.join(output_dir, "generation_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("批量生成摘要\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"总数: {len(texts)}\n")
        f.write(f"成功: {sum(1 for r in results if r['status'] == 'success')}\n")
        f.write(f"失败: {sum(1 for r in results if r['status'] == 'failed')}\n\n")
        
        f.write("详细列表:\n")
        f.write("-" * 60 + "\n")
        
        for result in results:
            f.write(f"\n[{result['index']:03d}] {result['text']}\n")
            if result['status'] == 'success':
                f.write(f"  ✓ {result['output']}\n")
            else:
                f.write(f"  ✗ 失败: {result.get('error', 'Unknown error')}\n")
    
    print(f"\n{'='*60}")
    print(f"✅ 批量生成完成!")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}")
    print(f"成功: {sum(1 for r in results if r['status'] == 'success')}/{len(texts)}")
    print(f"摘要文件: {summary_path}")


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="批量 Text-to-MIDI 生成")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--texts-file', type=str, default=None,
                        help='包含文本列表的文件 (每行一个文本)')
    parser.add_argument('--use-examples', action='store_true',
                        help='使用预定义的示例文本')
    parser.add_argument('--output-dir', type=str, default='generated_midi',
                        help='输出目录, 默认 generated_midi')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='生成时长 (秒), 默认 10.0')
    parser.add_argument('--num-steps', type=int, default=50,
                        help='采样步数, 默认 50')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='采样温度, 默认 1.0')
    parser.add_argument('--tempo', type=int, default=120,
                        help='BPM, 默认 120')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备 (cuda/cpu/auto), 默认 auto')
    
    args = parser.parse_args()
    
    # 获取文本列表
    if args.use_examples:
        texts = EXAMPLE_TEXTS
        print(f"使用 {len(texts)} 个预定义示例文本")
    elif args.texts_file:
        with open(args.texts_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"从 {args.texts_file} 读取 {len(texts)} 个文本")
    else:
        print("错误: 必须指定 --texts-file 或 --use-examples")
        return
    
    # 批量生成
    batch_generate(
        checkpoint_path=args.checkpoint,
        texts=texts,
        output_dir=args.output_dir,
        duration=args.duration,
        num_steps=args.num_steps,
        temperature=args.temperature,
        tempo=args.tempo,
        device=args.device
    )


if __name__ == "__main__":
    main()
