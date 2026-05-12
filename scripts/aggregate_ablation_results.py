#!/usr/bin/env python3
"""
Aggregate ablation run summaries into a CSV and update experiment summary markdown.
"""
import json
from pathlib import Path
import csv

root = Path('evaluation_outputs/ablation_runs')
out_csv = Path('experiment_results/ablation_aggregate.csv')
rows = []

for sub in sorted(root.iterdir()):
    if not sub.is_dir():
        continue
    summary_file = sub / 'evaluation_summary.json'
    if not summary_file.exists():
        continue
    try:
        s = json.load(open(summary_file, 'r', encoding='utf-8'))
    except Exception as e:
        print(f"Failed to read {summary_file}: {e}")
        continue

    # pick a few key metrics if present
    def get(k):
        return s.get(k) or s.get(f'all_{k}_mean') or ''

    row = {
        'run': sub.name,
        'note_density_mean': get('note_density'),
        'pitch_entropy_mean': get('pitch_entropy'),
        'duration_entropy_mean': get('duration_entropy'),
        'empty_bar_rate_mean': get('empty_bar_rate'),
        'pitch_kl_divergence_mean': get('pitch_kl_divergence'),
        'total_samples': s.get('total_samples', ''),
        'successful_samples': s.get('successful_samples', ''),
    }
    rows.append(row)

# write CSV
with out_csv.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['run','note_density_mean','pitch_entropy_mean','duration_entropy_mean','empty_bar_rate_mean','pitch_kl_divergence_mean','total_samples','successful_samples'])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Wrote aggregate CSV to {out_csv}")

# Append summary to experiment summary markdown
md = Path('experiment_results/EXPERIMENT_SUMMARY.md')
if md.exists():
    with md.open('a', encoding='utf-8') as f:
        f.write('\n\n**Ablation Study Aggregate**\n\n')
        f.write('Aggregated results (CSV): `experiment_results/ablation_aggregate.csv`.\n\n')
        f.write('Per-run figures are available under `experiment_results/ablation_figures/` for each run.\n')

print('Appended ablation summary to EXPERIMENT_SUMMARY.md')
