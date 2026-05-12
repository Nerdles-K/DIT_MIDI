"""
MIDI DiT Training Script
=======================

Main training script for MIDI generation model using BabySlakh dataset.
"""

import warnings
# Suppress warnings before any other imports
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
# Enable Tensor Core optimization for RTX GPUs (15-20% speedup)
torch.set_float32_matmul_precision('medium')

import os
import sys
from pathlib import Path

import hydra
# 添加 open_dict
from omegaconf import DictConfig, OmegaConf, open_dict
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    RichProgressBar,
    RichModelSummary,
    LearningRateMonitor
)
from utils.plot_training_curves import TrainingCurvePlotter
# Optional automatic plotting callback (ensures plots are created after training)
try:
    from main.auto_plot_callback import AutoPlotCallback
except Exception:
    AutoPlotCallback = None

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from main.module_midi_dit import MIDIDiTLightningModule, MIDIDataModule


@hydra.main(version_base=None, config_path="exp", config_name="train_babyslakh_midi_dit")
def train_midi_dit(cfg: DictConfig) -> None:
    """
    Train MIDI DiT model.
    
    Args:
        cfg: Hydra configuration
    """
    
    print("=" * 60)
    print("MIDI DiT Training with AdaLN-Zero")
    print("=" * 60)
    
    # Print configuration
    print(f"Working directory: {os.getcwd()}")
    print(f"Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    # Initialize data module
    print("\nInitializing data module...")
    try:
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        print(f"[OK] Data module initialized")
        print(f"  Dataset path: {cfg.datamodule.data_path}")
        print(f"  Batch size: {cfg.datamodule.batch_size}")
        print(f"  Patch size: {cfg.datamodule.patch_size}")
    except Exception as e:
        print(f"[FAIL] Error initializing data module: {e}")
        return
    
    # Initialize model
    print("\nInitializing model...")
    try:
        model = hydra.utils.instantiate(cfg.model)
        print(f"[OK] Model initialized")
        print(f"  Architecture: DiT with AdaLN-Zero")
        print(f"  Embedding dim: {cfg.model.embed_dim}")
        print(f"  Depth: {cfg.model.depth}")
        print(f"  Attention heads: {cfg.model.num_heads}")
        print(f"  Patch dim: {cfg.model.patch_dim}")
    except Exception as e:
        print(f"[FAIL] Error initializing model: {e}")
        return
        
    # Setup callbacks
    print("\nSetting up callbacks...")
    callbacks = []
    
    if "callbacks" in cfg:
        for callback_name, callback_cfg in cfg.callbacks.items():
            try:
                callback = hydra.utils.instantiate(callback_cfg)
                callbacks.append(callback)
                print(f"  [OK] {callback_name}")
            except Exception as e:
                print(f"  [FAIL] {callback_name}: {e}")

    # Always append AutoPlotCallback (best-effort) to ensure plotting after training
    try:
        if AutoPlotCallback is not None:
            logs_dir = cfg.get('logs_dir', 'logs') if isinstance(cfg, dict) or hasattr(cfg, 'get') else 'logs'
            plot_output_dir = cfg.get('plot_output_dir', 'plots') if isinstance(cfg, dict) or hasattr(cfg, 'get') else 'plots'
            ap_cb = AutoPlotCallback(logs_dir=str(logs_dir), output_dir=str(plot_output_dir))
            callbacks.append(ap_cb)
            print("  [OK] AutoPlotCallback (appended, only latest run will be plotted)")
    except Exception as e:
        print(f"  [WARN] AutoPlotCallback not appended: {e}")
    
    # Setup loggers
    loggers = []
    if "loggers" in cfg:
        for logger_name, logger_cfg in cfg.loggers.items():
            try:
                logger = hydra.utils.instantiate(logger_cfg)
                loggers.append(logger)
                print(f"  [OK] {logger_name} logger")
            except Exception as e:
                print(f"  [FAIL] {logger_name} logger: {e}")
    
    # Initialize trainer
    print("\nInitializing trainer...")
    
    # --- [修改开始] 处理 PyTorch Lightning 2.0+ 的 checkpoint 兼容性 ---
    resume_ckpt_path = None
    
    # 1. 检查是否通过命令行传入了 +trainer.resume_from_checkpoint
    if "resume_from_checkpoint" in cfg.trainer:
        resume_ckpt_path = cfg.trainer.resume_from_checkpoint
        print(f"Found resume_from_checkpoint in trainer config: {resume_ckpt_path}")
        
        # 关键步骤：必须从 cfg.trainer 中删除这个键，因为 Trainer.__init__ 不再接受它
        # 使用 open_dict 解锁配置以进行修改
        # 直接使用 open_dict
        with open_dict(cfg):
            del cfg.trainer.resume_from_checkpoint
    # --- [修改结束] ---

    try:
        trainer = hydra.utils.instantiate(
            cfg.trainer,
            callbacks=callbacks,
            logger=loggers if loggers else None
        )
        print(f"[OK] Trainer initialized")
        print(f"  Max epochs: {cfg.trainer.max_epochs}")
        print(f"  Devices: {cfg.trainer.devices}")
        print(f"  Precision: {cfg.trainer.precision}")
    except Exception as e:
        print(f"[FAIL] Error initializing trainer: {e}")
        return
    
    # Check if resuming from checkpoint
    # 逻辑整合：如果命令行没传 +trainer...，则检查配置文件里的自定义 ckpt 字段
    if resume_ckpt_path is None and "ckpt" in cfg and cfg.ckpt:
        resume_ckpt_path = cfg.ckpt
        
    if resume_ckpt_path:
        print(f"\nResuming from checkpoint: {resume_ckpt_path}")
    
    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    try:
        # 将路径传给 fit 函数的 ckpt_path 参数
        trainer.fit(model, datamodule, ckpt_path=resume_ckpt_path)
        print("\n[OK] Training completed successfully!")
    
        # Print best checkpoint info
        if hasattr(trainer.checkpoint_callback, 'best_model_path') and trainer.checkpoint_callback.best_model_path:
            print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")
            if trainer.checkpoint_callback.best_model_score is not None:
                print(f"Best validation loss: {trainer.checkpoint_callback.best_model_score:.6f}")
        
        # 自动生成训练/验证损失曲线 (保存为图片)
        try:
            print("\nGenerating training curve plot for latest run...")
            # 使用配置中的 logs_dir（在 hydra 配置中定义为 `logs`）
            logs_dir = cfg.get('logs_dir', 'logs') if isinstance(cfg, dict) or hasattr(cfg, 'get') else 'logs'
            plotter = TrainingCurvePlotter(logs_dir=str(logs_dir))
            runs = plotter.get_all_runs()
            if runs:
                latest_run = runs[-1]
                # 如果配置了 plot_output_dir，则保存到该目录下
                plot_output_dir = None
                try:
                    plot_output_dir = cfg.get('plot_output_dir', None)
                except Exception:
                    plot_output_dir = None

                if plot_output_dir:
                    from pathlib import Path as _Path
                    out_dir = _Path(str(plot_output_dir))
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"training_curve_{latest_run.get('run_id','latest')}.png"
                    out = plotter.plot_single_run(latest_run, output_path=out_path)
                else:
                    out = plotter.plot_single_run(latest_run)

                if out:
                    print(f"Saved training curve: {out}")
                else:
                    print("No plot generated for latest run (no history found).")
            else:
                print("No wandb/offline runs found under logs directory; skipping plot generation.")
        except Exception as e:
            print(f"Failed to generate training curve plot: {e}")
            
    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted by user")
    except Exception as e:
        print(f"\n[FAIL] Training failed: {e}")
        raise


def quick_test():
    """Quick test to verify model can overfit on a small dataset."""
    
    print("Running overfitting test...")
    
    # Minimal config for testing
    test_config = {
        'model': {
            '_target_': 'main.module_midi_dit.MIDIDiTLightningModule',
            'patch_dim': 256,
            'embed_dim': 384,  # Smaller for testing
            'depth': 6,        # Fewer layers
            'num_heads': 6,
            'max_seq_len': 512,
            'mlp_ratio': 4.0,
            'dropout': 0.0,    # No dropout for overfitting
            'num_classes': 5,
            'lr': 1e-3,        # Higher learning rate
            'patch_size': [16, 16]
        },
        'datamodule': {
            '_target_': 'main.module_midi_dit.MIDIDataModule',
            'data_path': 'midi_data/babyslakh/babyslakh_16k',
            'batch_size': 4,   # Small batch size
            'num_workers': 0,
            'patch_size': [16, 16],
            'max_files': 20,   # Very small dataset
            'val_split': 0.2
        },
        'trainer': {
            '_target_': 'pytorch_lightning.Trainer',
            'max_epochs': 10,
            'devices': 1,
            'precision': 16,
            'fast_dev_run': False,
            'overfit_batches': 10,  # Overfit on 10 batches
            'enable_checkpointing': False,
            'enable_progress_bar': True,
            'log_every_n_steps': 1
        }
    }
    
    # Convert to OmegaConf
    cfg = OmegaConf.create(test_config)
    
    # Run test
    try:
        model = hydra.utils.instantiate(cfg.model)
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        trainer = hydra.utils.instantiate(cfg.trainer)
        
        print("Starting overfitting test...")
        trainer.fit(model, datamodule)
        print("[OK] Overfitting test completed successfully!")
        
    except Exception as e:
        print(f"[FAIL] Overfitting test failed: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", 
                       help="Run quick overfitting test")
    
    args, unknown = parser.parse_known_args()
    
    if args.test:
        quick_test()
    else:
        train_midi_dit()