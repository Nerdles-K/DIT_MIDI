"""
训练曲线绘制工具
====================

从实验报告中提取训练数据并生成可视化图表。
支持自动识别实验版本,生成训练/验证损失曲线。

使用方法:
    # 生成所有实验对比图
    python utils/plot_training_curves.py
    
    # 指定保存路径
    python utils/plot_training_curves.py --output-dir plots/
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import numpy as np
from datetime import datetime
import re
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def _sanitize_text(s: str) -> str:
    """Remove non-ASCII characters (e.g., Chinese or emoji) to avoid missing-glyph boxes in saved images."""
    if not isinstance(s, str):
        return s
    return ''.join(ch for ch in s if ord(ch) < 128)


def _safe_filename(s: str) -> str:
    """Make a filesystem-safe filename fragment from a string (replace unsafe chars with underscore)."""
    if not isinstance(s, str):
        s = str(s)
    # keep alnum, dot, dash, underscore
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', s)


def _find_closest_event_file(run_dir: Path, logs_root: Path = Path('lightning_logs'), max_seconds: int = 600) -> Optional[Path]:
    """Search lightning_logs for the event file whose mtime is closest to the run_dir files mtime.
    Return the Path if within max_seconds, otherwise None.
    """
    try:
        files_dir = run_dir / 'files'
        ref = files_dir.stat().st_mtime if files_dir.exists() else run_dir.stat().st_mtime
    except Exception:
        return None

    candidates = list(logs_root.glob('**/events.*')) if logs_root.exists() else []
    if not candidates:
        return None

    best = None
    best_delta = None
    for c in candidates:
        try:
            dt = abs(c.stat().st_mtime - ref)
        except Exception:
            continue
        if best is None or dt < best_delta:
            best = c
            best_delta = dt

    if best is not None and best_delta is not None and best_delta <= max_seconds:
        return best
    return None


def _load_scalars_from_event(event_path: Path, tags_of_interest=None) -> Dict[str, List[float]]:
    """Load scalar series from a TensorBoard event file for specified tags.
    Returns a dict mapping tag->list of values (ordered by step).
    """
    if tags_of_interest is None:
        tags_of_interest = ['train_loss_step', 'train_loss_epoch', 'val_loss']

    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return {}

    ea = event_accumulator.EventAccumulator(str(event_path))
    try:
        ea.Reload()
    except Exception:
        return {}

    tags = ea.Tags().get('scalars', [])
    out = {}
    for t in tags_of_interest:
        if t in tags:
            vals = ea.Scalars(t)
            out[t] = [v.value for v in vals]
    return out


class TrainingCurvePlotter:
    """训练曲线绘制器"""
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.wandb_dir = self.logs_dir / "wandb"
        self.runs_dir = self.logs_dir / "runs"
        # Versions to exclude from plots (still training)
        self.EXCLUDE_VERSIONS = {"v5.1"}
        
    def get_all_runs(self) -> List[Dict]:
        """获取所有训练run的信息"""
        runs = []
        
        if self.wandb_dir.exists():
            for run_dir in sorted(self.wandb_dir.glob("offline-run-*")):
                run_info = self._parse_run_dir(run_dir)
                if run_info:
                    runs.append(run_info)
        
        return runs
    
    def _parse_run_dir(self, run_dir: Path) -> Optional[Dict]:
        """解析单个run目录"""
        # 从目录名提取信息: offline-run-20251130_142821-0rbg3wyr
        dir_name = run_dir.name
        parts = dir_name.split("-")
        
        if len(parts) >= 4:
            date_time = parts[2]  # 20251130_142821
            run_id = parts[3]     # 0rbg3wyr
            
            # 解析日期时间
            try:
                date_str = date_time.split("_")[0]
                time_str = date_time.split("_")[1]
                year = date_str[:4]
                month = date_str[4:6]
                day = date_str[6:8]
                hour = time_str[:2]
                minute = time_str[2:4]
                
                datetime_str = f"{year}-{month}-{day} {hour}:{minute}"
                
                # 读取run摘要
                summary_file = run_dir / "files" / "wandb-summary.json"
                summary = {}
                if summary_file.exists():
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                
                return {
                    "run_id": run_id,
                    "datetime": datetime_str,
                    "date": f"{year}-{month}-{day}",
                    "path": run_dir,
                    "summary": summary
                }
            except Exception as e:
                print(f"解析run目录失败 {run_dir.name}: {e}")
                return None
        
        return None
    
    def load_run_data(self, run_dir: Path) -> Dict:
        """加载run的训练数据"""
        # 读取summary获取最终指标
        summary_file = run_dir / "files" / "wandb-summary.json"
        summary = {}
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                try:
                    summary = json.load(f)
                except Exception:
                    summary = {}

        # 初始化返回结构
        data = {
            "train_loss": [],
            "val_loss": [],
            "epochs": [],
            "steps": [],
            "lr": [],
            "final_metrics": summary
        }

        # 尝试从 summary 中获取最终loss
        if "val_loss" in summary:
            data["final_val_loss"] = summary["val_loss"]
        if "train_loss_epoch" in summary:
            data["final_train_loss"] = summary["train_loss_epoch"]
        if "epoch" in summary:
            data["total_epochs"] = summary["epoch"]

        # 尝试解析可能的历史文件 (jsonl / csv) 来恢复 train/val loss 曲线
        files_dir = run_dir / "files"
        if files_dir.exists():
            # 找到可能的历史文件
            candidates = list(files_dir.glob("**/*history*.jsonl")) + \
                         list(files_dir.glob("**/*.jsonl")) + \
                         list(files_dir.glob("**/*.csv"))

            for fpath in candidates:
                try:
                    if fpath.suffix == '.jsonl':
                        with open(fpath, 'r', encoding='utf-8') as fh:
                            for line in fh:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    obj = json.loads(line)
                                except Exception:
                                    continue
                                # collect common keys
                                if 'train_loss' in obj and isinstance(obj['train_loss'], (int, float)):
                                    data['train_loss'].append(obj['train_loss'])
                                if 'val_loss' in obj and isinstance(obj['val_loss'], (int, float)):
                                    data['val_loss'].append(obj['val_loss'])
                                # fallback: some logs use 'loss' and 'mode'
                                if 'loss' in obj and isinstance(obj['loss'], (int, float)):
                                    # try to guess if it's validation by presence of 'is_val' or 'mode'
                                    key = obj.get('mode') or obj.get('split') or ''
                                    if 'val' in str(key).lower():
                                        data['val_loss'].append(obj['loss'])
                                    else:
                                        data['train_loss'].append(obj['loss'])
                    elif fpath.suffix == '.csv':
                        with open(fpath, 'r', encoding='utf-8') as fh:
                            header = fh.readline().strip().split(',')
                            cols = {c: i for i, c in enumerate(header)}
                            for ln in fh:
                                parts = ln.strip().split(',')
                                def getcol(name):
                                    if name in cols and cols[name] < len(parts):
                                        try:
                                            return float(parts[cols[name]])
                                        except Exception:
                                            return None
                                    return None
                                t = getcol('train_loss') or getcol('loss_train')
                                v = getcol('val_loss') or getcol('loss_val') or getcol('val')
                                if t is not None:
                                    data['train_loss'].append(t)
                                if v is not None:
                                    data['val_loss'].append(v)
                except Exception:
                    # 忽略单个文件解析错误
                    continue

        # If we didn't find history in files/, try to find a nearby TensorBoard event file
        if not data['train_loss'] and not data['val_loss']:
            event_file = _find_closest_event_file(run_dir)
            if event_file is not None:
                scalars = _load_scalars_from_event(event_file)
                # Prefer epoch-level train loss if present
                if 'train_loss_epoch' in scalars and scalars['train_loss_epoch']:
                    data['train_loss'] = scalars['train_loss_epoch']
                elif 'train_loss_step' in scalars and scalars['train_loss_step']:
                    data['train_loss'] = scalars['train_loss_step']

                if 'val_loss' in scalars and scalars['val_loss']:
                    data['val_loss'] = scalars['val_loss']

        return data
    
    def identify_experiment_version(self, run_info: Dict) -> str:
        """识别实验版本"""
        date = run_info["date"]
        summary = run_info.get("summary", {})
        
        # 根据日期和特征判断版本
        if date == "2025-11-28":
            return "v3.0"
        elif date == "2025-11-29":
            if run_info["datetime"].startswith("2025-11-29 00"):
                return "v3.0"
            else:
                return "v3.1"
        elif date == "2025-11-30":
            time = run_info["datetime"].split()[1]
            if time.startswith("00:") or time.startswith("02:2"):
                return "v4.0/4.1"
            elif time.startswith("02:") and not time.startswith("02:2"):
                return "v4.2"
            elif time.startswith("14:"):
                return "v5.0"
            elif time.startswith("17:"):
                return "v5.1"
            else:
                return "v4.x"
        
        return "unknown"
    
    def plot_single_run(self, run_info: Dict, output_path: Optional[str] = None):
        """绘制单个run的训练曲线"""
        run_data = self.load_run_data(run_info["path"])
        version = self.identify_experiment_version(run_info)

        if version in getattr(self, 'EXCLUDE_VERSIONS', set()):
            print(f"Skipping plotting for {run_info['run_id']} (version {version}) — excluded (still training)")
            return None

        # If we have historical loss arrays, plot them. Otherwise fall back to summary-only display.
        train_loss = run_data.get('train_loss', [])
        val_loss = run_data.get('val_loss', [])

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f'Training Curves - {version} ({run_info["datetime"]})', 
                     fontsize=16, fontweight='bold')

        ax1 = axes[0]
        ax2 = axes[1]

        if train_loss or val_loss:
            # build x-axis (epochs or steps)
            length = max(len(train_loss), len(val_loss))
            xs = list(range(1, length+1))
            if train_loss:
                ax1.plot(range(1, len(train_loss)+1), train_loss, label='Train Loss')
            if val_loss:
                ax1.plot(range(1, len(val_loss)+1), val_loss, label='Val Loss')

            ax1.set_title('Training/Validation Loss', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Epoch/Step', fontsize=11)
            ax1.set_ylabel('Loss', fontsize=11)
            ax1.legend()
            ax1.grid(True, alpha=0.3, linestyle='--')

            # Right panel: textual summary
            summary = run_data.get('final_metrics', {}) or {}
            final_val_loss = summary.get('val_loss', (val_loss[-1] if val_loss else 'N/A'))
            final_train_loss = summary.get('train_loss_epoch', (train_loss[-1] if train_loss else 'N/A'))
            total_epochs = summary.get('epoch', None)

            info_text = f'Final metrics:\n'
            info_text += f'Epochs: {total_epochs}\n' if total_epochs is not None else ''
            info_text += f'Train Loss: {final_train_loss}\n'
            info_text += f'Val Loss: {final_val_loss}\n'

            ax2.axis('off')
            ax2.text(0.1, 0.9, _sanitize_text(info_text), transform=ax2.transAxes,
                     fontsize=11, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        else:
            # No history available — show a minimal info panel using summary
            summary = run_data.get('final_metrics', {}) or {}
            final_val_loss = summary.get('val_loss', 'N/A')
            final_train_loss = summary.get('train_loss_epoch', 'N/A')
            total_epochs = summary.get('epoch', 'N/A')

            ax1.axis('off')
            info_text = f'No historical loss data available.\n\n'
            info_text += f'Run ID: {run_info["run_id"]}\n'
            info_text += f'Epochs: {total_epochs}\n'
            info_text += f'Train Loss: {final_train_loss}\n'
            info_text += f'Val Loss: {final_val_loss}\n'
            ax1.text(0.02, 0.98, _sanitize_text(info_text), transform=ax1.transAxes,
                     fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        
        # 保存图表
        if output_path is None:
            output_dir = Path("plots")
            output_dir.mkdir(exist_ok=True)
            safe_version = _safe_filename(version)
            output_path = output_dir / f"training_curve_{safe_version}_{run_info['run_id']}.png"

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot: {output_path}")
        plt.close()

        return output_path
    
    def plot_comparison(self, runs: List[Dict], output_path: Optional[str] = None):
        """绘制多个实验的对比图"""
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle('Experiment Version Comparison - Validation Loss', fontsize=16, fontweight='bold')
        
        # 收集每个版本的最佳loss
        version_data = {}
        
        for run in runs:
            version = self.identify_experiment_version(run)
            summary = run.get("summary", {})
            val_loss = summary.get("val_loss")
            
            if val_loss and isinstance(val_loss, (int, float)):
                if version not in version_data or val_loss < version_data[version]["val_loss"]:
                    version_data[version] = {
                        "val_loss": val_loss,
                        "train_loss": summary.get("train_loss_epoch", None),
                        "epochs": summary.get("epoch", None),
                        "datetime": run["datetime"]
                    }
        
        # 按版本排序
        version_order = ["v3.0", "v3.1", "v4.0/4.1", "v4.2", "v5.0", "v5.1"]
        sorted_versions = [v for v in version_order if v in version_data and v not in self.EXCLUDE_VERSIONS]
        
        if not sorted_versions:
            print("警告: 没有找到有效的实验数据")
            return
        
        # 绘制柱状图
        val_losses = [version_data[v]["val_loss"] for v in sorted_versions]
        colors = ['#2ecc71' if loss < 100 else '#e74c3c' for loss in val_losses]
        
        bars = ax.bar(sorted_versions, val_losses, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for i, (bar, loss) in enumerate(zip(bars, val_losses)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{loss:.2f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 添加基准线
        if "v3.1" in version_data:
            best_loss = version_data["v3.1"]["val_loss"]
            ax.axhline(y=best_loss, color='green', linestyle='--', linewidth=2, 
                      label=f'Best baseline (v3.1): {best_loss:.2f}', alpha=0.7)
        
        ax.set_xlabel('Experiment Version', fontsize=12, fontweight='bold')
        ax.set_ylabel('Validation Loss (Val Loss)', fontsize=12, fontweight='bold')
        ax.set_title('Best Validation Loss by Version', fontsize=13, pad=20)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 添加版本信息文本
        info_text = "Version info:\n"
        for version in sorted_versions:
            data = version_data[version]
            info_text += f"\n{version}:"
            info_text += f" loss={data['val_loss']:.2f}"
            if data['epochs']:
                info_text += f", epochs={data['epochs']}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图表
        if output_path is None:
            output_dir = Path("plots")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"experiments_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 已保存对比图: {output_path}")
        plt.close()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("Experiment Version Statistics")
        print("="*60)
        for version in sorted_versions:
            data = version_data[version]
            status = "Best" if version == "v3.1" else ("Failed" if data['val_loss'] > 100 else "Ongoing")
            print(f"{version:12s} | Val Loss: {data['val_loss']:8.2f} | Epochs: {data['epochs']:3d} | {status}")
        print("="*60)
        
        return output_path


def main():
    parser = argparse.ArgumentParser(description='绘制训练曲线')
    parser.add_argument('--logs-dir', type=str, default='logs', 
                       help='日志目录路径')
    parser.add_argument('--run-id', type=str, default=None,
                       help='指定run ID')
    parser.add_argument('--compare-all', action='store_true',
                       help='生成所有实验的对比图')
    parser.add_argument('--per-run', action='store_true', dest='per_run',
                       help='Generate per-run loss plots for all runs (excluding excluded versions)')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='输出目录')
    parser.add_argument('--latest', action='store_true', default=True,
                       help='绘制最新一次训练 (默认)')
    
    args = parser.parse_args()
    
    # 创建绘图器
    plotter = TrainingCurvePlotter(args.logs_dir)
    
    # 获取所有runs
    runs = plotter.get_all_runs()
    
    if not runs:
        print("错误: 没有找到任何训练run")
        return
    
    print(f"找到 {len(runs)} 个训练run")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.compare_all:
        # 生成对比图
        print("\n生成实验对比图...")
        plotter.plot_comparison(runs, output_dir / "experiments_comparison.png")

    if args.per_run:
        print("\nGenerating per-run loss plots (excluding unfinished versions)...")
        created = 0
        for run in runs:
            version = plotter.identify_experiment_version(run)
            if version in plotter.EXCLUDE_VERSIONS:
                print(f"Skipping {run['run_id']} (version {version}) - excluded")
                continue
            safe_version = _safe_filename(version)
            outpath = output_dir / f"training_curve_{safe_version}_{run['run_id']}.png"
            res = plotter.plot_single_run(run, outpath)
            if res:
                created += 1
        print(f"Generated {created} per-run plots in {output_dir}")
    
    if args.run_id:
        # 绘制指定run
        target_run = None
        for run in runs:
            if run["run_id"] == args.run_id:
                target_run = run
                break
        
        if target_run:
            print(f"\n绘制run: {args.run_id}")
            plotter.plot_single_run(target_run, 
                                   output_dir / f"training_curve_{args.run_id}.png")
        else:
            print(f"错误: 找不到run ID: {args.run_id}")
    
    elif args.latest and not args.compare_all:
        # 绘制最新run
        latest_run = runs[-1]
        print(f"\n绘制最新训练: {latest_run['run_id']} ({latest_run['datetime']})")
        plotter.plot_single_run(latest_run,
                               output_dir / f"training_curve_latest.png")
    
    # 默认生成对比图
    if not args.run_id and not args.compare_all:
        print("\n生成实验对比图...")
        plotter.plot_comparison(runs, output_dir / "experiments_comparison.png")
    
    print(f"\n✅ 所有图表已保存到: {output_dir}/")


if __name__ == "__main__":
    main()
