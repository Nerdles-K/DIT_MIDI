#!/usr/bin/env python3
"""
Run ablation grid based on `exp/ablation_study.yaml`.
This script imports `generate_and_evaluate` from `scripts/run_evaluation.py` and runs
multiple evaluations with modified `fixed_params`.
"""
import sys
from pathlib import Path
import yaml
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.run_evaluation import generate_and_evaluate, load_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation-config', type=str, default='exp/ablation_study.yaml')
    parser.add_argument('--prompts-file', type=str, default='exp/evaluation_prompts.yaml')
    parser.add_argument('--output-root', type=str, default='evaluation_outputs/ablation_runs')
    parser.add_argument('--checkpoint', type=str, default='logs/ckpts/midi_dit_2025-11-29-00-54-13/epoch=25-val_loss=67.8794.ckpt')
    parser.add_argument('--repeats', type=int, default=None)
    args = parser.parse_args()

    ablation_cfg = yaml.safe_load(open(args.ablation_config, 'r', encoding='utf-8'))
    # Load full prompts and baseline eval config
    prompts_all, eval_config_baseline = load_prompts(args.prompts_file)

    # Build subset prompts: ablation list contains keys
    ablation_keys = ablation_cfg.get('ablation_prompts', [])
    prompts_subset = {k: prompts_all[k] for k in ablation_keys}

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Sampling steps ablation
    steps_cfg = ablation_cfg['experiments']['sampling_steps']
    steps_list = [steps_cfg.get('baseline')] + steps_cfg.get('variations', [])
    temp_for_steps = steps_cfg.get('fixed_params', {}).get('temperature', eval_config_baseline['fixed_params']['temperature'])
    duration_for_steps = steps_cfg.get('fixed_params', {}).get('duration', eval_config_baseline['fixed_params']['duration'])

    # Temperature ablation
    temp_cfg = ablation_cfg['experiments']['temperature']
    temp_list = [temp_cfg.get('baseline')] + temp_cfg.get('variations', [])
    steps_for_temp = temp_cfg.get('fixed_params', {}).get('num_steps', eval_config_baseline['fixed_params']['num_steps'])
    duration_for_temp = temp_cfg.get('fixed_params', {}).get('duration', eval_config_baseline['fixed_params']['duration'])

    num_repeats = args.repeats if args.repeats is not None else ablation_cfg.get('evaluation', {}).get('num_repeats', 2)

    print(f"Running ablation with {len(prompts_subset)} prompts, repeats={num_repeats}")

    # Run sampling steps experiments
    for s in steps_list:
        out_dir = output_root / f"sampling_steps_{s}"
        eval_cfg = {
            'num_repeats': num_repeats,
            'fixed_params': {
                'num_steps': s,
                'temperature': temp_for_steps,
                'duration': duration_for_steps
            }
        }
        print(f"\n-> Running sampling steps = {s} (temp={temp_for_steps}) -> {out_dir}")
        generate_and_evaluate(prompts_subset, eval_cfg, args.checkpoint, out_dir, num_repeats)

    # Run temperature experiments
    for t in temp_list:
        out_dir = output_root / f"temperature_{t}"
        eval_cfg = {
            'num_repeats': num_repeats,
            'fixed_params': {
                'num_steps': steps_for_temp,
                'temperature': t,
                'duration': duration_for_temp
            }
        }
        print(f"\n-> Running temperature = {t} (steps={steps_for_temp}) -> {out_dir}")
        generate_and_evaluate(prompts_subset, eval_cfg, args.checkpoint, out_dir, num_repeats)

    print('\nAll ablation runs complete!')


if __name__ == '__main__':
    main()
