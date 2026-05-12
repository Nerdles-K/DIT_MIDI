from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

evt = Path('lightning_logs/version_0/events.out.tfevents.1764266250.computer.19524.0')
if not evt.exists():
    print('No event file found at', evt)
else:
    ea = event_accumulator.EventAccumulator(str(evt))
    ea.Reload()
    tags = ea.Tags()
    print('Tags:', tags)
    scalars = tags.get('scalars', [])
    for tag in scalars:
        vals = ea.Scalars(tag)
        print(f"Tag: {tag}, #points={len(vals)}")
        for v in vals[:5]:
            print('  ', v.step, v.value)
        print('---')
