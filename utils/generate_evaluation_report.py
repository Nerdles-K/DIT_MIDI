#!/usr/bin/env python3
"""
Evaluation Report Generator
Aggregates metrics and generates visualization + markdown report
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Metrics display names and target ranges
METRIC_INFO = {
    'note_density': {'name': 'Note Density', 'unit': 'notes/s', 'target': (10, 30), 'higher_better': None},
    'pitch_entropy': {'name': 'Pitch Entropy', 'unit': 'bits', 'target': (3.0, 5.5), 'higher_better': True},
    'duration_entropy': {'name': 'Duration Entropy', 'unit': 'bits', 'target': (1.5, 3.0), 'higher_better': True},
    'empty_bar_rate': {'name': 'Empty Bar Rate', 'unit': '%', 'target': (0, 0.1), 'higher_better': False},
    'pitch_kl_divergence': {'name': 'Pitch KL Div', 'unit': '', 'target': (0, 1.5), 'higher_better': False},
    'velocity_mean': {'name': 'Avg Velocity', 'unit': '', 'target': (60, 100), 'higher_better': None},
    'velocity_std': {'name': 'Velocity Std', 'unit': '', 'target': (10, 25), 'higher_better': None},
    'pitch_range': {'name': 'Pitch Range', 'unit': 'semitones', 'target': (24, 60), 'higher_better': None},
    'avg_ioi': {'name': 'Avg IOI', 'unit': 's', 'target': (0.1, 0.5), 'higher_better': None},
}


def load_results(results_path: Path) -> tuple:
    """Load evaluation results and summary"""
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    summary_path = results_path.parent / 'evaluation_summary.json'
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    else:
        summary = {}
    
    return results, summary


def plot_metrics_by_difficulty(results: List[Dict], output_dir: Path):
    """Generate box plots comparing metrics across difficulty levels"""
    difficulties = ['easy', 'medium', 'hard']
    metrics = ['note_density', 'pitch_entropy', 'duration_entropy', 'empty_bar_rate']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data_by_diff = {d: [] for d in difficulties}
        
        for r in results:
            if 'error' in r and r['error'] != 'no_notes':
                continue
            diff = r.get('difficulty', 'unknown')
            if diff in difficulties and metric in r:
                data_by_diff[diff].append(r[metric])
        
        # Box plot
        bp = ax.boxplot(
            [data_by_diff[d] for d in difficulties],
            labels=difficulties,
            patch_artist=True
        )
        
        # Color boxes
        colors = ['#3498db', '#e67e22', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        info = METRIC_INFO.get(metric, {})
        ax.set_title(info.get('name', metric))
        ax.set_ylabel(info.get('unit', ''))
        ax.set_xlabel('Difficulty')
        ax.grid(True, alpha=0.3)
        
        # Add target range if available
        if 'target' in info:
            low, high = info['target']
            ax.axhspan(low, high, alpha=0.1, color='green', zorder=0)
    
    plt.tight_layout()
    out_path = output_dir / 'metrics_by_difficulty.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_radar_chart(summary: Dict, output_dir: Path):
    """Generate radar chart for normalized metrics"""
    # Select key metrics for radar
    radar_metrics = ['note_density', 'pitch_entropy', 'duration_entropy', 'pitch_range']
    
    values = []
    labels = []
    for metric in radar_metrics:
        key = f'all_{metric}_mean'
        if key in summary:
            val = summary[key]
            info = METRIC_INFO.get(metric, {})
            
            # Normalize to 0-1 based on target range
            if 'target' in info:
                low, high = info['target']
                normalized = (val - low) / (high - low) if high > low else 0.5
                normalized = np.clip(normalized, 0, 1)
            else:
                normalized = 0.5
            
            values.append(normalized)
            labels.append(info.get('name', metric))
    
    if len(values) == 0:
        print("No metrics available for radar chart")
        return None
    
    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
    values += values[:1]  # Close the loop
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
    ax.fill(angles, values, alpha=0.25, color='#3498db')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title('Normalized Metrics (Target Range)', pad=20, fontsize=14, weight='bold')
    ax.grid(True)
    
    plt.tight_layout()
    out_path = output_dir / 'radar_chart.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_metrics_distribution(results: List[Dict], output_dir: Path):
    """Plot histograms for key metrics"""
    metrics = ['note_count', 'pitch_range', 'avg_ioi', 'velocity_mean']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [r[metric] for r in results if metric in r and ('error' not in r or r.get('error') == 'no_notes')]
        
        if len(values) == 0:
            continue
        
        ax.hist(values, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
        
        info = METRIC_INFO.get(metric, {'name': metric, 'unit': ''})
        ax.set_xlabel(f"{info['name']} ({info['unit']})" if info['unit'] else info['name'])
        ax.set_ylabel('Count')
        ax.set_title(f"{info['name']} Distribution")
        ax.grid(True, alpha=0.3)
        
        # Add mean line
        mean_val = np.mean(values)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.legend()
    
    plt.tight_layout()
    out_path = output_dir / 'metrics_distribution.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def generate_markdown_report(
    results: List[Dict],
    summary: Dict,
    output_dir: Path,
    experiment_name: str = "Baseline"
):
    """Generate comprehensive markdown report"""
    
    report_lines = [
        f"# {experiment_name} Evaluation Report",
        f"",
        f"Generated: {Path(output_dir).name}",
        f"",
        f"## Summary",
        f"",
        f"- **Total Samples**: {summary.get('total_samples', 0)}",
        f"- **Successful**: {summary.get('successful_samples', 0)}",
        f"- **Success Rate**: {summary.get('successful_samples', 0) / summary.get('total_samples', 1) * 100:.1f}%",
        f"",
        f"## Key Metrics",
        f"",
        f"| Metric | Mean | Std | Min | Max | Target Range | Status |",
        f"|--------|------|-----|-----|-----|--------------|--------|",
    ]
    
    # Add metric rows
    for metric, info in METRIC_INFO.items():
        mean_key = f'all_{metric}_mean'
        if mean_key in summary:
            mean = summary[mean_key]
            std = summary.get(f'all_{metric}_std', 0)
            min_val = summary.get(f'all_{metric}_min', 0)
            max_val = summary.get(f'all_{metric}_max', 0)
            
            target = info.get('target', (0, 0))
            if target[0] <= mean <= target[1]:
                status = "✓"
            else:
                status = "⚠"
            
            target_str = f"{target[0]:.2f} - {target[1]:.2f}"
            
            report_lines.append(
                f"| {info['name']} | {mean:.4f} | {std:.4f} | {min_val:.4f} | {max_val:.4f} | {target_str} | {status} |"
            )
    
    report_lines.extend([
        f"",
        f"## Metrics by Difficulty",
        f"",
    ])
    
    # Aggregate by difficulty
    for difficulty in ['easy', 'medium', 'hard']:
        diff_results = [r for r in results if r.get('difficulty') == difficulty and ('error' not in r or r.get('error') == 'no_notes')]
        if len(diff_results) == 0:
            continue
        
        report_lines.append(f"### {difficulty.capitalize()}")
        report_lines.append(f"")
        report_lines.append(f"- Samples: {len(diff_results)}")
        
        for metric in ['note_density', 'pitch_entropy', 'duration_entropy', 'empty_bar_rate']:
            values = [r[metric] for r in diff_results if metric in r]
            if values:
                mean = np.mean(values)
                std = np.std(values)
                info = METRIC_INFO.get(metric, {'name': metric})
                report_lines.append(f"- {info['name']}: {mean:.4f} ± {std:.4f}")
        
        report_lines.append(f"")
    
    report_lines.extend([
        f"## Visualizations",
        f"",
        f"- `metrics_by_difficulty.png` - Box plots comparing metrics across difficulty levels",
        f"- `radar_chart.png` - Normalized metrics radar chart",
        f"- `metrics_distribution.png` - Distribution histograms for key metrics",
        f"",
        f"## Generated Files",
        f"",
        f"- `evaluation_results.json` - Detailed per-sample results",
        f"- `evaluation_summary.json` - Aggregated statistics",
        f"- `evaluation_report.md` - This report",
        f"",
        f"## Observations",
        f"",
    ])
    
    # Auto-generate observations
    if 'all_empty_bar_rate_mean' in summary:
        ebr = summary['all_empty_bar_rate_mean']
        if ebr < 0.1:
            report_lines.append(f"- ✓ **Good**: Empty bar rate ({ebr:.3f}) is within target (<0.1)")
        else:
            report_lines.append(f"- ⚠ **Warning**: Empty bar rate ({ebr:.3f}) exceeds target (>0.1)")
    
    if 'all_pitch_entropy_mean' in summary:
        pe = summary['all_pitch_entropy_mean']
        if 3.0 <= pe <= 5.5:
            report_lines.append(f"- ✓ **Good**: Pitch entropy ({pe:.3f}) is within healthy range (3.0-5.5)")
        else:
            report_lines.append(f"- ⚠ **Note**: Pitch entropy ({pe:.3f}) is outside typical range")
    
    if 'all_duration_entropy_mean' in summary:
        de = summary['all_duration_entropy_mean']
        if de > 1.5:
            report_lines.append(f"- ✓ **Good**: Duration entropy ({de:.3f}) indicates diverse note lengths")
        else:
            report_lines.append(f"- ⚠ **Note**: Duration entropy ({de:.3f}) is low, may indicate repetitive patterns")
    
    report_lines.append(f"")
    report_lines.append(f"---")
    report_lines.append(f"*Report generated by generate_evaluation_report.py*")
    
    # Write report
    report_path = output_dir / 'evaluation_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Saved: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='Generate evaluation report with visualizations')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to evaluation_results.json')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as results)')
    parser.add_argument('--name', type=str, default='Baseline',
                        help='Experiment name for report title')
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from {results_path}...")
    results, summary = load_results(results_path)
    print(f"Loaded {len(results)} samples")
    
    print("\nGenerating visualizations...")
    plot_metrics_by_difficulty(results, output_dir)
    plot_radar_chart(summary, output_dir)
    plot_metrics_distribution(results, output_dir)
    
    print("\nGenerating markdown report...")
    generate_markdown_report(results, summary, output_dir, experiment_name=args.name)
    
    print(f"\n✓ Report generation complete!")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    main()
