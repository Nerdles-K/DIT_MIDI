"""
Auto plot callback
==================

Lightning callback that generates training/validation loss plots after training ends.
This ensures plots are created even if Hydra config does not include automatic plotting.
"""
from pathlib import Path
import traceback
import logging

import pytorch_lightning as pl


logger = logging.getLogger(__name__)



# 只生成最新 run 的损失曲线图
class AutoPlotCallback(pl.Callback):
    """Generate only the latest run's training curve after training finishes."""
    def __init__(self, logs_dir: str = 'logs', output_dir: str = 'plots'):
        super().__init__()
        self.logs_dir = str(logs_dir)
        self.output_dir = str(output_dir)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        try:
            from utils.plot_training_curves import TrainingCurvePlotter
            plotter = TrainingCurvePlotter(logs_dir=self.logs_dir)
            runs = plotter.get_all_runs()
            if not runs:
                logger.warning("AutoPlotCallback: no runs found under %s", self.logs_dir)
                return
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            latest_run = runs[-1]
            outp = output_path / "training_curve_latest.png"
            plotter.plot_single_run(latest_run, outp)
            logger.info("AutoPlotCallback: only latest run plotted: %s", outp)
        except Exception:
            logger.warning("AutoPlotCallback: error during plotting:\n%s", traceback.format_exc())
