"""
训练损失对比图生成工具
========================

根据实验报告中的数据生成训练损失对比图表

使用方法:
    python utils/generate_loss_plots.py
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def _sanitize_text(s: str) -> str:
    """Remove non-ASCII characters (e.g., Chinese or emoji) to avoid missing-glyph boxes in saved images."""
    if not isinstance(s, str):
        return s
    return ''.join(ch for ch in s if ord(ch) < 128)


# Versions to exclude from plots (e.g. still training)
EXCLUDE_VERSIONS = {"v5.1"}


# 实验数据 (从实验报告中提取)
EXPERIMENT_DATA = {
    "v3.0": {
        "date": "2025-11-28",
        "optimizer": "AdamW",
        "model_size": "125M",
        "data_aug": False,
        "data_files": 1000,
        "best_val_loss": 90.0,  # 估算
        "best_epoch": 30,
        "total_epochs": 33,
        "status": "Baseline"
    },
    "v3.1": {
        "date": "2025-11-29",
        "optimizer": "AdamW",
        "model_size": "125M",
        "data_aug": False,
        "data_files": 1000,
        "best_val_loss": 70.95,
        "best_epoch": 31,
        "total_epochs": 33,
        "status": "Best",
        "train_loss": 72.23
    },
    "v4.0": {
        "date": "2025-11-30",
        "optimizer": "Muon",
        "model_size": "125M",
        "data_aug": False,
        "data_files": 1000,
        "best_val_loss": 120.0,  # 估算
        "best_epoch": 50,
        "total_epochs": 60,
        "status": "Failed"
    },
    "v4.1": {
        "date": "2025-11-30",
        "optimizer": "Muon",
        "model_size": "125M",
        "data_aug": False,
        "data_files": 1000,
        "best_val_loss": 110.0,  # 估算
        "best_epoch": 50,
        "total_epochs": 70,
        "status": "Failed"
    },
    "v4.2": {
        "date": "2025-11-30",
        "optimizer": "Muon",
        "model_size": "125M",
        "data_aug": False,
        "data_files": 1000,
        "best_val_loss": 101.19,
        "best_epoch": 97,
        "total_epochs": 99,
        "status": "Failed",
        "train_loss": 101.84
    },
    "v5.0": {
        "date": "2025-11-30",
        "optimizer": "Muon",
        "model_size": "75M",
        "data_aug": True,
        "aug_strength": "medium",
        "data_files": 2000,
        "best_val_loss": 171.54,
        "best_epoch": 71,
        "total_epochs": 73,
        "status": "Failed",
        "train_loss": 173.18
    },
    "v5.1": {
        "date": "2025-11-30",
        "optimizer": "AdamW",
        "model_size": "75M",
        "data_aug": True,
        "aug_strength": "light",
        "data_files": 2000,
        "best_val_loss": 253.64,  # 初期数据
        "best_epoch": 1,
        "total_epochs": 3,
        "status": "Training",
        "train_loss": 250.42
    }
}


def create_comparison_chart():
    """创建实验对比柱状图"""
    # 准备数据
    versions = [v for v in EXPERIMENT_DATA.keys() if v not in EXCLUDE_VERSIONS]
    val_losses = [EXPERIMENT_DATA[v]["best_val_loss"] for v in versions]
    
    # 颜色编码: 绿色=好, 黄色=中等, 红色=差
    colors = []
    for loss in val_losses:
        if loss < 80:
            colors.append('#2ecc71')  # 绿色 - 优秀
        elif loss < 110:
            colors.append('#f39c12')  # 橙色 - 中等
        elif loss < 180:
            colors.append('#e74c3c')  # 红色 - 差
        else:
            colors.append('#c0392b')  # 深红 - 很差
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制柱状图
    bars = ax.bar(versions, val_losses, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    
    # 添加数值标签
    for bar, loss in zip(bars, val_losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{loss:.2f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 添加最佳基线
    best_loss = EXPERIMENT_DATA["v3.1"]["best_val_loss"]
    ax.axhline(y=best_loss, color='green', linestyle='--', linewidth=2.5,
              label=f'Best baseline (v3.1): {best_loss:.2f}', alpha=0.8, zorder=0)
    
    # 设置标题和标签
    ax.set_title('DIT MIDI Experiment Version Comparison - Best Validation Loss', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Experiment Version', fontsize=13, fontweight='bold')
    ax.set_ylabel('Validation Loss (Val Loss)', fontsize=13, fontweight='bold')
    
    # 网格和图例
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(fontsize=11, loc='upper right')
    
    # 设置y轴范围
    ax.set_ylim(0, max(val_losses) * 1.15)
    
    # 添加版本信息文本框
    info_lines = []
    for version in versions:
        data = EXPERIMENT_DATA[version]
        line = f"{version}: "
        line += f"loss={data['best_val_loss']:.2f}, "
        line += f"{data['optimizer']}, "
        line += f"{data['model_size']}"
        if data.get('data_aug'):
            line += f", aug={data.get('aug_strength', 'yes')}"
        line += f" - {data['status']}"
        info_lines.append(line)

    info_text = "\n".join(info_lines)
    info_text = _sanitize_text(info_text)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
           zorder=10)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "experiments_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存对比图: {output_path}")
    plt.close()
    
    return output_path


def create_detailed_chart():
    """创建详细的实验配置对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Experiment Configuration and Performance Analysis', fontsize=16, fontweight='bold')
    
    versions = [v for v in EXPERIMENT_DATA.keys() if v not in EXCLUDE_VERSIONS]
    
    # 左图: Val Loss 柱状图
    val_losses = [EXPERIMENT_DATA[v]["best_val_loss"] for v in versions]
    colors_val = []
    for loss in val_losses:
        if loss < 80:
            colors_val.append('#2ecc71')
        elif loss < 110:
            colors_val.append('#f39c12')
        elif loss < 180:
            colors_val.append('#e74c3c')
        else:
            colors_val.append('#c0392b')
    
    bars1 = ax1.bar(versions, val_losses, color=colors_val, alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
    
    for bar, loss in zip(bars1, val_losses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.axhline(y=70.95, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_title('Validation Loss Comparison', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Version', fontsize=11)
    ax1.set_ylabel('Val Loss', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(val_losses) * 1.1)
    
    # 右图: 优化器类型分布
    optimizers = {}
    for version in versions:
        opt = EXPERIMENT_DATA[version]["optimizer"]
        if opt not in optimizers:
            optimizers[opt] = []
        optimizers[opt].append(version)
    
    # 创建分组柱状图
    opt_names = list(optimizers.keys())
    x_pos = np.arange(len(versions))
    width = 0.35
    
    adamw_losses = []
    muon_losses = []
    
    for version in versions:
        if EXPERIMENT_DATA[version]["optimizer"] == "AdamW":
            adamw_losses.append(EXPERIMENT_DATA[version]["best_val_loss"])
            muon_losses.append(0)
        else:
            adamw_losses.append(0)
            muon_losses.append(EXPERIMENT_DATA[version]["best_val_loss"])
    
    bars_adamw = ax2.bar(x_pos - width/2, adamw_losses, width, 
                         label='AdamW', color='#3498db', alpha=0.8)
    bars_muon = ax2.bar(x_pos + width/2, muon_losses, width,
                        label='Muon', color='#e67e22', alpha=0.8)
    
    ax2.set_title('Optimizer Performance Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Version', fontsize=11)
    ax2.set_ylabel('Val Loss', fontsize=11)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(versions, rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "experiments_detailed.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存详细对比图: {output_path}")
    plt.close()
    
    return output_path


def create_timeline_chart():
    """创建时间线趋势图"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    versions = [v for v in EXPERIMENT_DATA.keys() if v not in EXCLUDE_VERSIONS]
    val_losses = [EXPERIMENT_DATA[v]["best_val_loss"] for v in versions]
    
    # Plot line chart
    line = ax.plot(versions, val_losses, marker='o', markersize=10, 
                   linewidth=2.5, color='#3498db', label='Val Loss Trend')
    
    # 标注每个点
    for i, (ver, loss) in enumerate(zip(versions, val_losses)):
        ax.annotate(f'{loss:.1f}', 
                   (ver, loss),
                   textcoords="offset points",
                   xytext=(0,10),
                   ha='center',
                   fontsize=10,
                   fontweight='bold')
    
    # Mark important version
    best_idx = versions.index("v3.1")
    ax.plot(versions[best_idx], val_losses[best_idx], 
            marker='*', markersize=20, color='gold', 
            markeredgecolor='green', markeredgewidth=2,
            label='Best version', zorder=10)
    
    # 添加基线
    ax.axhline(y=70.95, color='green', linestyle='--', 
              linewidth=2, alpha=0.6, label='Target baseline')
    
    # Set title and labels
    ax.set_title('Experiment Version Evolution Trend', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Experiment Version', fontsize=13, fontweight='bold')
    ax.set_ylabel('Validation Loss (Val Loss)', fontsize=13, fontweight='bold')
    
    # 网格和图例
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper right')
    
    # Add phase annotations
    ax.axvspan(-0.5, 0.5, alpha=0.1, color='blue', label='Initial exploration')
    ax.axvspan(0.5, 1.5, alpha=0.1, color='green')
    ax.axvspan(1.5, 4.5, alpha=0.1, color='red')
    ax.axvspan(4.5, 6.5, alpha=0.1, color='orange')

    # Add text annotations (sanitize to avoid missing glyphs)
    ax.text(0, max(val_losses)*0.95, _sanitize_text('v3.0-v3.1\nBaseline established'), 
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(3, max(val_losses)*0.95, _sanitize_text('v4.0-v4.2\nMuon failed'), 
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    ax.text(5.5, max(val_losses)*0.95, _sanitize_text('v5.0-v5.1\nOptimization attempts'), 
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "experiments_timeline.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存时间线图: {output_path}")
    plt.close()
    
    return output_path


def print_summary_table():
    """Print experiment summary table"""
    print("\n" + "="*100)
    print("Experiment Version Summary Table".center(100))
    print("="*100)
    print(f"{'Version':<8} {'Date':<12} {'Optimizer':<8} {'Model':<8} {'Data Aug':<12} "
        f"{'Data Files':<8} {'Best Loss':<12} {'Epoch':<8} {'Status':<12}")
    print("-"*100)

    for version, data in EXPERIMENT_DATA.items():
        if version in EXCLUDE_VERSIONS:
            continue
        aug_str = f"{data.get('aug_strength', 'N/A')}" if data.get('data_aug') else "No"
        print(f"{version:<8} {data['date']:<12} {data['optimizer']:<8} "
              f"{data['model_size']:<8} {aug_str:<12} {data['data_files']:<8} "
              f"{data['best_val_loss']:<12.2f} {data['best_epoch']:<8} {data['status']:<12}")

    print("="*100)

    # Key findings (skip excluded versions in findings)
    print("\nKey findings:")
    print("  1. v3.1 (AdamW) achieved best performance: 70.95")
    print("  2. v4.x (Muon) series failed, losses around 100-120")
    print("  3. v5.0 (Muon + smaller model + data augmentation) worst: 171.54")
    if 'v5.1' in EXCLUDE_VERSIONS:
        print("  4. v5.1 is excluded from plots (training not finished)")
    else:
        print("  4. v5.1 (AdamW + smaller model + light aug) training, initial loss high")

    print("\nConclusion: AdamW performs better than Muon for this task")
    print("="*100 + "\n")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("训练损失对比图生成工具".center(60))
    print("="*60 + "\n")
    
    # 创建输出目录
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    print(f"输出目录: {output_dir.absolute()}\n")
    
    # 生成各种图表
    print("正在生成图表...\n")
    
    create_comparison_chart()
    create_detailed_chart()
    create_timeline_chart()
    
    # 打印总结表格
    print_summary_table()
    
    print("✅ 所有图表生成完成!")
    print(f"\n查看图表: {output_dir.absolute()}\\")


if __name__ == "__main__":
    main()
