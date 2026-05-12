from pathlib import Path
import time

evt = Path('lightning_logs/version_0/events.out.tfevents.1764266250.computer.19524.0')
if not evt.exists():
    print('No event file found')
else:
    evt_mtime = evt.stat().st_mtime
    print('Event file mtime:', time.ctime(evt_mtime))

    runs_dir = Path('logs/wandb')
    if not runs_dir.exists():
        print('No runs dir')
    else:
        for run_dir in sorted(runs_dir.glob('offline-run-*')):
            # use files dir mtime if present
            fd = run_dir / 'files'
            if fd.exists():
                try:
                    m = fd.stat().st_mtime
                except Exception:
                    m = run_dir.stat().st_mtime
            else:
                m = run_dir.stat().st_mtime
            print(run_dir.name, 'mtime', time.ctime(m), 'delta (s)=', abs(evt_mtime - m))
