#!/usr/bin/env python3
"""
Evaluation Script for Text-to-MIDI Generation
Generates MIDI samples from prompts and computes objective metrics
"""

import sys
import os
from pathlib import Path
import yaml
import json
import argparse
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
import pretty_midi
from scripts.inference_text_to_midi import TextToMIDIGenerator


def load_prompts(yaml_path: str) -> Tuple[Dict, Dict]:
    """Load evaluation prompts from YAML config"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['prompts'], config['evaluation']


def compute_metrics(midi_path: str) -> Dict:
    """
    Compute objective quality metrics for a MIDI file
    Returns dict with: empty_bar_rate, pitch_entropy, duration_entropy, note_density, etc.
    """
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"Error loading {midi_path}: {e}")
        return None
    
    # Collect all notes
    all_notes = []
    for instrument in pm.instruments:
        all_notes.extend(instrument.notes)
    
    if len(all_notes) == 0:
        return {
            'empty_bar_rate': 1.0,
            'note_count': 0,
            'note_density': 0.0,
            'pitch_entropy': 0.0,
            'duration_entropy': 0.0,
            'velocity_mean': 0.0,
            'velocity_std': 0.0,
            'pitch_range': 0,
            'avg_ioi': 0.0,
            'error': 'no_notes'
        }
    
    # Sort by start time
    all_notes.sort(key=lambda n: n.start)
    
    # Basic stats
    total_duration = max(n.end for n in all_notes)
    note_count = len(all_notes)
    note_density = note_count / total_duration if total_duration > 0 else 0
    
    # Pitch distribution
    pitches = [n.pitch for n in all_notes]
    pitch_counts = np.bincount(pitches, minlength=128)
    pitch_probs = pitch_counts / pitch_counts.sum()
    pitch_probs = pitch_probs[pitch_probs > 0]  # Remove zeros for entropy calc
    pitch_entropy = -np.sum(pitch_probs * np.log2(pitch_probs))
    pitch_range = max(pitches) - min(pitches)
    
    # Duration distribution
    durations = [n.end - n.start for n in all_notes]
    # Bin durations into categories (eighth, quarter, half, whole, etc.)
    duration_bins = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    duration_hist, _ = np.histogram(durations, bins=duration_bins + [float('inf')])
    duration_probs = duration_hist / duration_hist.sum()
    duration_probs = duration_probs[duration_probs > 0]
    duration_entropy = -np.sum(duration_probs * np.log2(duration_probs))
    
    # Velocity stats
    velocities = [n.velocity for n in all_notes]
    velocity_mean = np.mean(velocities)
    velocity_std = np.std(velocities)
    
    # Inter-onset intervals (IOI)
    onsets = [n.start for n in all_notes]
    iois = np.diff(onsets)
    avg_ioi = np.mean(iois) if len(iois) > 0 else 0.0
    
    # Empty bar rate (bars with < 2 notes)
    # Assume 4/4 time, 120 BPM -> 2 seconds per bar
    bar_duration = 2.0
    num_bars = int(np.ceil(total_duration / bar_duration))
    bar_note_counts = []
    for i in range(num_bars):
        bar_start = i * bar_duration
        bar_end = (i + 1) * bar_duration
        bar_notes = [n for n in all_notes if bar_start <= n.start < bar_end]
        bar_note_counts.append(len(bar_notes))
    
    empty_bars = sum(1 for count in bar_note_counts if count < 2)
    empty_bar_rate = empty_bars / num_bars if num_bars > 0 else 0.0
    
    # Pitch KL divergence from uniform (in used range)
    used_pitches = list(range(min(pitches), max(pitches) + 1))
    uniform_prob = 1.0 / len(used_pitches)
    observed_probs = []
    for p in used_pitches:
        count = pitch_counts[p]
        observed_probs.append((count + 1e-8) / note_count)  # Add smoothing
    
    kl_div = sum(p * np.log2(p / uniform_prob) for p in observed_probs)
    
    return {
        'note_count': note_count,
        'note_density': note_density,
        'pitch_entropy': pitch_entropy,
        'duration_entropy': duration_entropy,
        'velocity_mean': velocity_mean,
        'velocity_std': velocity_std,
        'pitch_range': pitch_range,
        'avg_ioi': avg_ioi,
        'empty_bar_rate': empty_bar_rate,
        'pitch_kl_divergence': kl_div,
        'total_duration': total_duration,
    }


def generate_and_evaluate(
    prompts: Dict,
    eval_config: Dict,
    checkpoint: str,
    output_dir: Path,
    num_repeats: int = None
):
    """
    Generate MIDI for all prompts and compute metrics
    """
    if num_repeats is None:
        num_repeats = eval_config['num_repeats']
    
    fixed_params = eval_config['fixed_params']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    print("Initializing model...")
    generator = TextToMIDIGenerator(checkpoint_path=checkpoint)
    print("Model ready!")
    
    all_results = []
    metrics_summary = defaultdict(list)
    
    # Progress bar
    total_generations = len(prompts) * num_repeats
    pbar = tqdm(total=total_generations, desc="Generating & Evaluating")
    
    for prompt_id, prompt_config in prompts.items():
        text = prompt_config['text']
        difficulty = prompt_config['difficulty']
        category = prompt_config['category']
        
        for repeat in range(num_repeats):
            # Generate filename
            filename = f"{prompt_id}_rep{repeat:02d}.mid"
            midi_path = output_dir / filename
            
            try:
                # Generate MIDI
                generator.generate_midi(
                    text=text,
                    output_path=str(midi_path),
                    num_steps=fixed_params['num_steps'],
                    temperature=fixed_params['temperature'],
                    duration=fixed_params['duration']
                )
                
                # Compute metrics
                metrics = compute_metrics(str(midi_path))
                
                if metrics:
                    result = {
                        'prompt_id': prompt_id,
                        'text': text,
                        'difficulty': difficulty,
                        'category': category,
                        'repeat': repeat,
                        'filename': filename,
                        **metrics
                    }
                    all_results.append(result)
                    
                    # Aggregate by difficulty
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            metrics_summary[f"{difficulty}_{key}"].append(value)
                            metrics_summary[f"all_{key}"].append(value)
                
            except Exception as e:
                print(f"\nError generating {filename}: {e}")
                all_results.append({
                    'prompt_id': prompt_id,
                    'text': text,
                    'difficulty': difficulty,
                    'category': category,
                    'repeat': repeat,
                    'filename': filename,
                    'error': str(e)
                })
            
            pbar.update(1)
    
    pbar.close()
    
    # Save detailed results
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Compute summary statistics
    summary = {}
    for key, values in metrics_summary.items():
        if len(values) > 0:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
            summary[f"{key}_min"] = float(np.min(values))
            summary[f"{key}_max"] = float(np.max(values))
    
    summary['total_samples'] = len(all_results)
    summary['successful_samples'] = len([r for r in all_results if 'error' not in r or r.get('error') == 'no_notes'])
    
    # Save summary
    summary_path = output_dir / 'evaluation_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Results saved to: {results_path}")
    print(f"  Summary saved to: {summary_path}")
    print(f"  Generated {len(all_results)} samples")
    print(f"  Success rate: {summary['successful_samples']}/{summary['total_samples']}")
    
    # Print key metrics
    print("\n=== Key Metrics (Mean ± Std) ===")
    for metric_name in ['note_density', 'pitch_entropy', 'duration_entropy', 'empty_bar_rate', 'pitch_kl_divergence']:
        if f'all_{metric_name}_mean' in summary:
            mean = summary[f'all_{metric_name}_mean']
            std = summary[f'all_{metric_name}_std']
            print(f"{metric_name:25s}: {mean:8.4f} ± {std:.4f}")
    
    return all_results, summary


def main():
    parser = argparse.ArgumentParser(description='Run evaluation on Text-to-MIDI model')
    parser.add_argument('--prompts', type=str, default='exp/evaluation_prompts.yaml',
                        help='Path to prompts YAML file')
    parser.add_argument('--checkpoint', type=str, 
                        default='logs/ckpts/midi_dit_2025-11-29-00-54-13/epoch=25-val_loss=67.8794.ckpt',
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='evaluation_outputs/baseline',
                        help='Output directory for generated MIDI and results')
    parser.add_argument('--num-repeats', type=int, default=None,
                        help='Number of repeats per prompt (overrides config)')
    
    args = parser.parse_args()
    
    # Load prompts
    print(f"Loading prompts from {args.prompts}...")
    prompts, eval_config = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts")
    
    # Run evaluation
    output_dir = Path(args.output_dir)
    print(f"\nStarting evaluation...")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output dir: {output_dir}")
    print(f"  Repeats per prompt: {args.num_repeats or eval_config['num_repeats']}")
    print(f"  Fixed params: {eval_config['fixed_params']}")
    
    generate_and_evaluate(
        prompts=prompts,
        eval_config=eval_config,
        checkpoint=args.checkpoint,
        output_dir=output_dir,
        num_repeats=args.num_repeats
    )


if __name__ == '__main__':
    main()
