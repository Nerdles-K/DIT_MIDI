import os
from pathlib import Path
import pretty_midi
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze_folder(folder: str):
    folder = Path(folder)
    files = sorted(folder.glob('generated_*.mid'))
    if not files:
        print('No generated midi files found in', folder)
        return None

    names = []
    note_counts = []
    durations = []
    avg_vels = []
    pitch_spans = []

    for f in files:
        try:
            pm = pretty_midi.PrettyMIDI(str(f))
        except Exception as e:
            print('Failed to read', f, e)
            continue

        names.append(f.name)
        total_notes = sum(len(i.notes) for i in pm.instruments)
        dur = pm.get_end_time()
        velocities = [n.velocity for i in pm.instruments for n in i.notes]
        avg_vel = np.mean(velocities) if velocities else 0
        pitches = [n.pitch for i in pm.instruments for n in i.notes]
        if pitches:
            pitch_span = (min(pitches), max(pitches))
        else:
            pitch_span = (0, 0)

        note_counts.append(total_notes)
        durations.append(dur)
        avg_vels.append(avg_vel)
        pitch_spans.append(pitch_span)

    # Compute note density
    densities = [nc / d if d > 0 else 0 for nc, d in zip(note_counts, durations)]

    out_dir = Path('plots')
    out_dir.mkdir(exist_ok=True)

    # 1) Combined bar: Note Count and Density
    x = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(x - 0.2, note_counts, width=0.4, label='Note Count')
    ax1.set_ylabel('Note Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, densities, width=0.4, color='orange', label='Note Density (notes/s)')
    ax2.set_ylabel('Note Density (notes/s)')
    plt.title('Benchmark Generation Results')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    out_path1 = out_dir / 'benchmark_results.png'
    plt.savefig(out_path1, dpi=300, bbox_inches='tight')
    plt.close()

    # 2) Histogram: note counts
    fig = plt.figure(figsize=(8, 4))
    plt.hist(note_counts, bins=min(10, max(3, len(note_counts))), color='#3498db', edgecolor='black')
    plt.xlabel('Note Count')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Note Counts')
    plt.tight_layout()
    out_path2 = out_dir / 'note_count_hist.png'
    plt.savefig(out_path2, dpi=300, bbox_inches='tight')
    plt.close()

    # 3) Average velocity per sample (bar)
    fig = plt.figure(figsize=(12, 4))
    plt.bar(x, avg_vels, color='#2ecc71')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Avg Velocity')
    plt.title('Average Note Velocity per Sample')
    plt.tight_layout()
    out_path3 = out_dir / 'avg_velocity.png'
    plt.savefig(out_path3, dpi=300, bbox_inches='tight')
    plt.close()

    # 4) Pitch span width vs note density (scatter)
    pitch_widths = [p[1] - p[0] for p in pitch_spans]
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(pitch_widths, densities, c='purple', s=80)
    for i, name in enumerate(names):
        plt.annotate(name, (pitch_widths[i], densities[i]), textcoords='offset points', xytext=(5,5), fontsize=8)
    plt.xlabel('Pitch Span Width (semitones)')
    plt.ylabel('Note Density (notes/s)')
    plt.title('Pitch Span vs Note Density')
    plt.tight_layout()
    out_path4 = out_dir / 'pitchspan_vs_density.png'
    plt.savefig(out_path4, dpi=300, bbox_inches='tight')
    plt.close()

    print('Saved benchmark plots to:', out_path1, out_path2, out_path3, out_path4)
    # 5) Pitch-class histogram (0-11)
    pitch_classes = []
    for f in files:
        try:
            pm = pretty_midi.PrettyMIDI(str(f))
        except Exception:
            continue
        pitches = [n.pitch for i in pm.instruments for n in i.notes]
        pitch_classes.extend([p % 12 for p in pitches])

    if pitch_classes:
        fig = plt.figure(figsize=(8, 4))
        plt.hist(pitch_classes, bins=np.arange(13)-0.5, rwidth=0.8, color='#9b59b6', edgecolor='black')
        plt.xticks(range(12), ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'])
        plt.xlabel('Pitch Class')
        plt.ylabel('Count')
        plt.title('Pitch-Class Distribution (all samples)')
        plt.tight_layout()
        out_path5 = out_dir / 'pitchclass_hist.png'
        plt.savefig(out_path5, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        out_path5 = None

    # 6) Time-bin note density heatmap (samples x timebins)
    timebins = 20
    heat = []
    for f in files:
        try:
            pm = pretty_midi.PrettyMIDI(str(f))
        except Exception:
            continue
        dur = pm.get_end_time()
        if dur <= 0:
            heat.append([0]*timebins)
            continue
        bins = np.linspace(0, dur, timebins+1)
        counts = np.zeros(timebins, dtype=int)
        for i in pm.instruments:
            for n in i.notes:
                # put onset into a bin
                idx = np.searchsorted(bins, n.start, side='right') - 1
                if idx >= 0 and idx < timebins:
                    counts[idx] += 1
        heat.append(counts)

    if heat:
        heat_arr = np.array(heat)
        fig = plt.figure(figsize=(10, 6))
        plt.imshow(heat_arr, aspect='auto', interpolation='nearest', cmap='viridis')
        plt.colorbar(label='Note Onsets')
        plt.xlabel('Time bin')
        plt.ylabel('Sample index')
        plt.title('Note Onset Density across Time (samples x timebins)')
        out_path6 = out_dir / 'time_density_heatmap.png'
        plt.tight_layout()
        plt.savefig(out_path6, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        out_path6 = None

    # 7) Piano-roll preview for first 3 samples
    preview_paths = []
    max_preview = min(3, len(files))
    for i in range(max_preview):
        f = files[i]
        try:
            pm = pretty_midi.PrettyMIDI(str(f))
        except Exception:
            continue
        # create piano roll (time x pitch)
        T = int(pm.get_end_time() * 100)  # 0.01s bins
        P = 88
        pr = np.zeros((T, P), dtype=int)
        for instr in pm.instruments:
            for n in instr.notes:
                pitch_idx = n.pitch - 21
                if pitch_idx < 0 or pitch_idx >= P:
                    continue
                start = max(0, int(n.start * 100))
                end = min(T, int(n.end * 100))
                pr[start:end, pitch_idx] = 1

        fig = plt.figure(figsize=(12, 3))
        plt.imshow(pr.T, aspect='auto', origin='lower', cmap='Greys')
        plt.xlabel('Time (0.01s bins)')
        plt.ylabel('Pitch (21-108)')
        plt.title(f'Piano-roll preview: {f.name}')
        outp = out_dir / f'piano_preview_{i:02d}.png'
        plt.tight_layout()
        plt.savefig(outp, dpi=200, bbox_inches='tight')
        plt.close()
        preview_paths.append(outp)

    saved = [p for p in [out_path1, out_path2, out_path3, out_path4, out_path5, out_path6] if p]
    saved.extend(preview_paths)
    print('Saved additional benchmark plots:', saved)
    return saved


def extra_diagnostics(folder: str, out_dir: Path = Path('plots')):
    """Generate extra diagnostic plots:
    - velocity_hist.png
    - note_length_hist.png
    - ioi_hist.png (inter-onset intervals)
    - bar_density_heatmap.png (per-bar or per-beat density heatmap)
    - pitchclass_over_time.png (stacked area for 12 pitch classes)
    """
    folder = Path(folder)
    files = sorted(folder.glob('generated_*.mid'))
    if not files:
        print('No generated midi files found for extra diagnostics in', folder)
        return []

    all_vel = []
    note_lengths = []
    all_onsets = []  # list of arrays of onsets per sample
    max_dur = 0.0

    for f in files:
        try:
            pm = pretty_midi.PrettyMIDI(str(f))
        except Exception:
            continue
        notes = [n for i in pm.instruments for n in i.notes]
        onsets = sorted([n.start for n in notes])
        durations = [n.end - n.start for n in notes if n.end > n.start]
        velocities = [n.velocity for n in notes]
        pitches = [n.pitch for n in notes]

        all_vel.extend(velocities)
        note_lengths.extend(durations)
        all_onsets.append(onsets)
        if onsets:
            max_dur = max(max_dur, max(onsets))

    # velocity histogram
    out_paths = []
    if all_vel:
        plt.figure(figsize=(8,4))
        plt.hist(all_vel, bins=range(0,129,4), color='#e67e22', edgecolor='black')
        plt.xlabel('Velocity')
        plt.ylabel('Count')
        plt.title('Velocity Distribution (all samples)')
        outp = out_dir / 'velocity_hist.png'
        plt.tight_layout()
        plt.savefig(outp, dpi=300, bbox_inches='tight')
        plt.close()
        out_paths.append(outp)

    # note length histogram
    if note_lengths:
        plt.figure(figsize=(8,4))
        plt.hist(note_lengths, bins=50, color='#1abc9c', edgecolor='black')
        plt.xlabel('Note Length (s)')
        plt.ylabel('Count')
        plt.title('Note Length Distribution')
        outp = out_dir / 'note_length_hist.png'
        plt.tight_layout()
        plt.savefig(outp, dpi=300, bbox_inches='tight')
        plt.close()
        out_paths.append(outp)

    # IOI histogram (inter-onset intervals)
    iois = []
    for onsets in all_onsets:
        if len(onsets) > 1:
            diffs = np.diff(onsets)
            iois.extend(list(diffs))
    if iois:
        plt.figure(figsize=(8,4))
        plt.hist(iois, bins=50, color='#3498db', edgecolor='black')
        plt.xlabel('Inter-Onset Interval (s)')
        plt.ylabel('Count')
        plt.title('IOI Distribution (all samples)')
        outp = out_dir / 'ioi_hist.png'
        plt.tight_layout()
        plt.savefig(outp, dpi=300, bbox_inches='tight')
        plt.close()
        out_paths.append(outp)

    # bar / beat density heatmap: use fixed number of bins across samples
    bins = 40
    heat = []
    for onsets in all_onsets:
        if not onsets:
            heat.append([0]*bins)
            continue
        dur = max(onsets)
        edges = np.linspace(0, dur, bins+1)
        counts, _ = np.histogram(onsets, bins=edges)
        heat.append(counts)
    if heat:
        arr = np.array(heat)
        plt.figure(figsize=(10,6))
        plt.imshow(arr, aspect='auto', cmap='magma', interpolation='nearest')
        plt.colorbar(label='Onset Counts')
        plt.xlabel('Time bin')
        plt.ylabel('Sample index')
        plt.title('Per-sample Onset Density (bins)')
        outp = out_dir / 'bar_density_heatmap.png'
        plt.tight_layout()
        plt.savefig(outp, dpi=300, bbox_inches='tight')
        plt.close()
        out_paths.append(outp)

    # pitch-class over time stacked area
    tc = 40
    pc_mat = np.zeros((tc,12), dtype=float)
    counts_per_bin = np.zeros(tc, dtype=int)
    for onsets_list, f in zip(all_onsets, files):
        try:
            pm = pretty_midi.PrettyMIDI(str(f))
        except Exception:
            continue
        dur = max(onsets_list) if onsets_list else 0.0
        if dur <= 0:
            continue
        edges = np.linspace(0, dur, tc+1)
        for instr in pm.instruments:
            for n in instr.notes:
                idx = np.searchsorted(edges, n.start, side='right') - 1
                if idx < 0 or idx >= tc:
                    continue
                pc = n.pitch % 12
                pc_mat[idx, pc] += 1
        counts_per_bin += 1
    denom = counts_per_bin.copy()
    denom[denom==0] = 1
    pc_mean = pc_mat / denom[:,None]
    if pc_mean.sum() > 0:
        plt.figure(figsize=(10,4))
        x = np.arange(tc)
        labels = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        plt.stackplot(x, [pc_mean[:,i] for i in range(12)], labels=labels)
        plt.legend(loc='upper right', ncol=3, fontsize=8)
        plt.xlabel('Time bin')
        plt.ylabel('Avg occurrences (per sample)')
        plt.title('Pitch-class Over Time (stacked)')
        outp = out_dir / 'pitchclass_over_time.png'
        plt.tight_layout()
        plt.savefig(outp, dpi=300, bbox_inches='tight')
        plt.close()
        out_paths.append(outp)

    print('Saved extra diagnostics:', out_paths)
    return out_paths


if __name__ == '__main__':
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else 'benchmark_outputs'
    analyze_folder(folder)
