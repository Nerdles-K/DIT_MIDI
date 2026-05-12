import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.plot_training_curves import TrainingCurvePlotter

plotter = TrainingCurvePlotter('logs')
runs = plotter.get_all_runs()
print(f"Found {len(runs)} runs")
for r in runs:
    data = plotter.load_run_data(r['path'])
    tlen = len(data.get('train_loss', []))
    vlen = len(data.get('val_loss', []))
    summary_keys = list(r.get('summary', {}).keys())
    print(f"{r['run_id']:12s} | version={plotter.identify_experiment_version(r):10s} | train_len={tlen:3d} | val_len={vlen:3d} | summary_keys={summary_keys}")
    if tlen>0 or vlen>0:
        print("  sample train:", data.get('train_loss')[:5])
        print("  sample val:  ", data.get('val_loss')[:5])
