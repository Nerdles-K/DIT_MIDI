
"""
Text-to-MIDI 推理脚本
===================

允许用户输入文本描述来生成 MIDI 文件。

使用示例:
    python inference_text_to_midi.py \
        --text "piano happy melody in C" \
        --checkpoint "checkpoints/best.ckpt" \
        --output "generated.mid"
"""

# ==== 添加 main 目录到 sys.path，解决导入问题 ====
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import argparse
from typing import Optional, Tuple, List

import torch
import numpy as np
import pretty_midi
from tqdm import tqdm

from main.module_midi_dit import MIDIDiTLightningModule
from main.data.dataset_midi import MIDIPatchify


class TextToMIDIGenerator:
    """Text-to-MIDI 生成器"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        patch_size: Tuple[int, int] = (16, 16),
        sample_rate: int = 16,
        pitch_range: Tuple[int, int] = (21, 108)
    ):
        """
        Args:
            checkpoint_path: 模型检查点路径
            device: 运行设备
            patch_size: Patch 大小
            sample_rate: 量化率 (1/16 音符)
            pitch_range: MIDI 音高范围
        """
        self.device = device
        self.patch_size = patch_size
        self.sample_rate = sample_rate
        self.pitch_range = pitch_range
        
        # 加载模型
        print(f"正在从 {checkpoint_path} 加载模型...")
        self.model = MIDIDiTLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location=device
        )
        self.model.eval()
        self.model.to(device)
        print(f"✓ 模型加载成功 (设备: {device})")
        
        # Patchify 工具
        self.patchify = MIDIPatchify(patch_size)
        
    def generate(
        self,
        text: str,
        duration: float = 10.0,
        num_steps: int = 50,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        temperature: float = 1.0,
        class_label: Optional[int] = None
    ) -> np.ndarray:
        """
        从文本描述生成 MIDI piano roll
        
        Args:
            text: 文本描述,如 "piano happy melody in C"
            duration: 生成时长 (秒)
            num_steps: 扩散采样步数
            sigma_min: 最小噪声水平
            sigma_max: 最大噪声水平
            rho: 噪声调度参数
            temperature: 采样温度 (>1 更随机, <1 更确定)
            class_label: 类别标签 (可选)
            
        Returns:
            piano_roll: Piano roll 矩阵 (T, 88)
        """
        print(f"\n生成设置:")
        print(f"  文本: {text}")
        print(f"  时长: {duration}s")
        print(f"  采样步数: {num_steps}")
        print(f"  温度: {temperature}")
        
        # 计算目标形状
        num_time_steps = int(duration * self.sample_rate * 4)  # 每拍 4 个 1/16 音符
        num_pitches = self.pitch_range[1] - self.pitch_range[0]
        
        # 计算 patch 数量
        t_patch, p_patch = self.patch_size
        num_t_patches = (num_time_steps + t_patch - 1) // t_patch
        num_p_patches = (num_pitches + p_patch - 1) // p_patch
        num_patches = num_t_patches * num_p_patches
        patch_dim = t_patch * p_patch
        
        print(f"\n目标形状:")
        print(f"  Piano Roll: ({num_time_steps}, {num_pitches})")
        print(f"  Patches: ({num_patches}, {patch_dim})")
        
        # 创建噪声调度
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=self.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * 
                   (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        
        # 初始化噪声
        shape = (1, num_patches, patch_dim)
        x = torch.randn(shape, device=self.device) * sigma_max * temperature
        
        # 准备条件
        text_descriptions = [text]
        if class_label is None:
            class_label = 0  # 使用默认类别
        class_labels = torch.tensor([class_label], device=self.device, dtype=torch.long)
        
        # 扩散采样
        print(f"\n开始采样...")
        with torch.no_grad():
            for i in tqdm(range(num_steps), desc="采样进度"):
                sigma = t_steps[i]
                sigma_next = t_steps[i + 1]
                
                # 预测噪声
                noise_pred = self.model.model(
                    x,
                    sigma.expand(1),
                    class_labels=class_labels,
                    text_descriptions=text_descriptions
                )
                
                # DDPM 更新步骤
                if sigma_next > 0:
                    x = x + (sigma_next - sigma) * noise_pred
                else:
                    x = x - sigma * noise_pred
        
        # 转换回 piano roll
        print("\n转换为 Piano Roll...")
        patches = x[0]  # (num_patches, patch_dim)
        piano_roll = self.patchify.unpatchify(
            patches, 
            (num_time_steps, num_pitches)
        )
        
        # 转为numpy并检查值域
        piano_roll = piano_roll.cpu().numpy()
        print(f"  原始值域: [{piano_roll.min():.3f}, {piano_roll.max():.3f}]")
        print(f"  原始均值: {piano_roll.mean():.3f}")
        
        # 归一化到 [0, 1]
        piano_roll = (piano_roll - piano_roll.min()) / (piano_roll.max() - piano_roll.min() + 1e-8)
        
        # 使用自适应阈值（基于百分位数，保留top 5-10%的激活）
        threshold = np.percentile(piano_roll, 92)  # 保留约8%的值
        piano_roll_binary = (piano_roll > threshold).astype(np.float32)
        
        # 形态学处理：去除孤立噪点
        try:
            from scipy.ndimage import binary_opening, binary_closing
            structure = np.ones((3, 1))  # 时间维度上的结构元素
            piano_roll_clean = binary_opening(piano_roll_binary, structure=structure)
            piano_roll_clean = binary_closing(piano_roll_clean, structure=structure)
        except ImportError:
            # 如果没有scipy，使用简单的平滑
            piano_roll_clean = piano_roll_binary
        
        # 保留原始强度信息用于velocity
        piano_roll_intensity = piano_roll * piano_roll_clean
        
        print(f"✓ 生成完成! Piano Roll shape: {piano_roll_clean.shape}")
        print(f"  激活音符数: {piano_roll_clean.sum():.0f}")
        print(f"  音符密度: {piano_roll_clean.sum() / num_time_steps:.1f} notes/timestep")
        
        return piano_roll_clean, piano_roll_intensity
    
    def piano_roll_to_midi(
        self,
        piano_roll: np.ndarray,
        output_path: str,
        tempo: int = 120,
        velocity: int = 80,
        intensity_map: np.ndarray = None
    ) -> None:
        """
        将 piano roll 转换为 MIDI 文件
        
        Args:
            piano_roll: Piano roll 矩阵 (T, 88)
            output_path: 输出 MIDI 文件路径
            tempo: BPM
            velocity: 音符力度
            intensity_map: 强度图（用于动态velocity）
        """
        print(f"\n转换为 MIDI...")
        print(f"  输出路径: {output_path}")
        print(f"  Tempo: {tempo} BPM")
        print(f"  Velocity: {velocity}")
        
        # 创建 MIDI 对象
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=0)  # Piano
        
        # 时间步长 (秒)
        time_step = 60.0 / tempo / self.sample_rate
        
        # 提取音符
        T, P = piano_roll.shape
        for pitch_idx in range(P):
            midi_pitch = self.pitch_range[0] + pitch_idx
            
            # 查找音符的开始和结束
            note_on = False
            note_start = 0
            
            for t in range(T):
                if piano_roll[t, pitch_idx] > 0:
                    if not note_on:
                        note_on = True
                        note_start = t
                else:
                    if note_on:
                        # 音符结束
                        note_on = False
                        start_time = note_start * time_step
                        end_time = t * time_step
                        
                        # 计算velocity（使用intensity map或基础值+随机变化）
                        if intensity_map is not None:
                            # 使用音符区间的平均强度
                            avg_intensity = intensity_map[note_start:t, pitch_idx].mean()
                            note_velocity = int(velocity * 0.5 + velocity * avg_intensity * 0.5)
                            note_velocity = np.clip(note_velocity, 40, 120)
                        else:
                            # 添加随机变化
                            note_velocity = int(velocity + np.random.randn() * 10)
                            note_velocity = np.clip(note_velocity, 60, 100)
                        
                        note = pretty_midi.Note(
                            velocity=note_velocity,
                            pitch=midi_pitch,
                            start=start_time,
                            end=end_time
                        )
                        instrument.notes.append(note)
            
            # 处理最后一个音符
            if note_on:
                start_time = note_start * time_step
                end_time = T * time_step
                
                if intensity_map is not None:
                    avg_intensity = intensity_map[note_start:, pitch_idx].mean()
                    note_velocity = int(velocity * 0.5 + velocity * avg_intensity * 0.5)
                    note_velocity = np.clip(note_velocity, 40, 120)
                else:
                    note_velocity = int(velocity + np.random.randn() * 10)
                    note_velocity = np.clip(note_velocity, 60, 100)
                
                note = pretty_midi.Note(
                    velocity=note_velocity,
                    pitch=midi_pitch,
                    start=start_time,
                    end=end_time
                )
                instrument.notes.append(note)
        
        # 添加乐器并保存
        midi.instruments.append(instrument)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        midi.write(output_path)
        print(f"✓ MIDI 文件已保存: {output_path}")
        print(f"  总音符数: {len(instrument.notes)}")
        print(f"  时长: {midi.get_end_time():.2f}s")
    
    def generate_midi(
        self,
        text: str,
        output_path: str,
        duration: float = 10.0,
        num_steps: int = 50,
        temperature: float = 1.0,
        tempo: int = 120,
        velocity: int = 80,
        **kwargs
    ) -> str:
        """
        完整流程: 从文本生成 MIDI 文件
        
        Args:
            text: 文本描述
            output_path: 输出 MIDI 路径
            duration: 时长
            num_steps: 采样步数
            temperature: 采样温度
            tempo: BPM
            velocity: 力度
            
        Returns:
            output_path: 保存的 MIDI 文件路径
        """
        # 生成 piano roll
        piano_roll, intensity_map = self.generate(
            text=text,
            duration=duration,
            num_steps=num_steps,
            temperature=temperature,
            **kwargs
        )
        
        # 转换为 MIDI
        self.piano_roll_to_midi(
            piano_roll=piano_roll,
            output_path=output_path,
            tempo=tempo,
            velocity=velocity,
            intensity_map=intensity_map
        )
        
        return output_path


def main():
    """命令行接口"""
    import argparse
    parser = argparse.ArgumentParser(description="Text-to-MIDI 生成器 (text参数命令行输入，其余参数在文件内设置)")
    parser.add_argument('--text', type=str, required=True, help='文本描述,如 "piano happy melody in C"')
    args = parser.parse_args()

    # 其余参数直接在此处设置
    checkpoint_path = "logs\ckpts\midi_dit_2025-12-01-07-37-12\epoch=43-val_loss=82.9048.ckpt"  # ckpt 路径
    output_path = "generated.mid"  # 输出 MIDI 文件路径
    duration = 10.0  # 生成时长（秒）
    num_steps = 50   # 扩散采样步数
    temperature = 1.0  # 采样温度
    tempo = 120  # BPM
    velocity = 80  # 音符力度
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动选择设备
    class_label = None  # 类别标签（可选）

    # ====== 生成器与推理 ======
    generator = TextToMIDIGenerator(
        checkpoint_path=checkpoint_path,
        device=device
    )

    output_path = generator.generate_midi(
        text=args.text,
        output_path=output_path,
        duration=duration,
        num_steps=num_steps,
        temperature=temperature,
        tempo=tempo,
        velocity=velocity,
        class_label=class_label
    )

    print(f"\n{'='*60}")
    print(f"✅ 生成完成!")
    print(f"{'='*60}")
    print(f"文件: {output_path}")
    print(f"\n可以使用 MIDI 播放器打开该文件")


if __name__ == "__main__":
    main()
