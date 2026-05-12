"""
MIDI DiT Training Module
=======================

PyTorch Lightning module for training DiT-based MIDI generation model.
"""

import os
import random
import typing as tp
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

from main.midi_dit import create_midi_dit, MIDIDiTWrapper
from main.data.dataset_midi import (
    TextToMIDIDataset,
    collate_fn_text_to_midi,
    MIDIPatchify
)
from main.muon_optimizer import Muon, MuonWithDecoupledWeightDecay


class DiffusionLoss(nn.Module):
    """Diffusion training loss."""
    
    def __init__(self, 
                 sigma_min: float = 0.002,
                 sigma_max: float = 80.0,
                 rho: float = 7.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        
    def sample_sigma(self, batch_size: int, device: str) -> torch.Tensor:
        """Sample noise levels for training."""
        rnd_uniform = torch.rand(batch_size, device=device)
        sigma = (self.sigma_max ** (1 / self.rho) + rnd_uniform * 
                (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        return sigma
        
    def forward(self, 
                model_output: torch.Tensor,
                target: torch.Tensor,
                sigma: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute diffusion loss.
        
        Args:
            model_output: Model prediction (B, N, D)
            target: Target noise (B, N, D)  
            sigma: Noise levels (B,)
            mask: Sequence mask (B, N)
            
        Returns:
            Loss value
        """
        # MSE loss
        loss = F.mse_loss(model_output, target, reduction='none')  # (B, N, D)
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()
            
        return loss


class MIDIDiTLightningModule(pl.LightningModule):
    """PyTorch Lightning module for MIDI DiT training."""
    
    def __init__(self,
                 # Model parameters
                 patch_dim: int = 256,
                 embed_dim: int = 512,
                 depth: int = 8,
                 num_heads: int = 8,
                 max_seq_len: int = 2048,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 num_classes: int = 10,
                 use_text_conditioning: bool = True,
                 
                 # Training parameters
                 lr: float = 1e-4,
                 lr_beta1: float = 0.9,
                 lr_beta2: float = 0.999,
                 lr_eps: float = 1e-8,
                 lr_weight_decay: float = 0.01,
                 lr_warmup_steps: int = 10000,
                 
                 # Diffusion parameters
                 sigma_min: float = 0.002,
                 sigma_max: float = 80.0,
                 rho: float = 7.0,
                 
                 # Data parameters
                 patch_size: Tuple[int, int] = (16, 16)):
        
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Create model
        self.model = create_midi_dit(
            patch_dim=patch_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_classes=num_classes,
            use_text_conditioning=use_text_conditioning
        )
        
        # Loss function
        self.loss_fn = DiffusionLoss(sigma_min, sigma_max, rho)
        
        # For unpatchifying during evaluation
        self.patchify = MIDIPatchify(patch_size)
        
        # Training metrics
        self.automatic_optimization = True
        
    def forward(self, x: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x, sigma, **kwargs)
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        
        patches = batch['patches']  # (B, N, patch_dim)
        masks = batch['masks']      # (B, N)
        text_descriptions = batch.get('text_descriptions', None)  # (B,) list of strings
        
        B, N, D = patches.shape
        
        # Sample noise levels
        sigma = self.loss_fn.sample_sigma(B, patches.device)
        
        # Add noise to clean data
        noise = torch.randn_like(patches)
        noisy_patches = patches + sigma.view(B, 1, 1) * noise
        
        # Generate class labels (random for now, could be conditioned on genre/style)
        class_labels = torch.randint(0, self.hparams.num_classes, (B,), device=patches.device)
        
        # Model prediction with text conditioning
        model_output = self.model(
            noisy_patches, 
            sigma, 
            class_labels=class_labels, 
            text_descriptions=text_descriptions,
            mask=masks
        )
        
        # Compute loss (predict noise)
        loss = self.loss_fn(model_output, noise, sigma, masks)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        
        patches = batch['patches']
        masks = batch['masks']
        text_descriptions = batch.get('text_descriptions', None)
        
        B, N, D = patches.shape
        
        # Sample noise levels
        sigma = self.loss_fn.sample_sigma(B, patches.device)
        
        # Add noise
        noise = torch.randn_like(patches)
        noisy_patches = patches + sigma.view(B, 1, 1) * noise
        
        # Generate class labels
        class_labels = torch.randint(0, self.hparams.num_classes, (B,), device=patches.device)
        
        # Model prediction with text conditioning
        with torch.no_grad():
            model_output = self.model(
                noisy_patches, 
                sigma, 
                class_labels=class_labels,
                text_descriptions=text_descriptions,
                mask=masks
            )
            
        # Compute loss
        loss = self.loss_fn(model_output, noise, sigma, masks)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
        
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler with AdamW."""
        
        # v5.1: Switch back to AdamW after Muon failed in v4.0-v5.0
        # Use standard AdamW with all parameters
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.lr_beta1, self.hparams.lr_beta2),
            eps=self.hparams.lr_eps,
            weight_decay=self.hparams.lr_weight_decay
        )
        
        print(f"Using AdamW optimizer with lr={self.hparams.lr:.2e}, weight_decay={self.hparams.lr_weight_decay}")
        
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < self.hparams.lr_warmup_steps:
                return step / self.hparams.lr_warmup_steps
            else:
                return 0.5 * (1 + np.cos(np.pi * (step - self.hparams.lr_warmup_steps) / 
                                        (self.trainer.max_steps - self.hparams.lr_warmup_steps)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
        
    def generate_samples(self, 
                        batch_size: int = 4,
                        seq_len: int = 512,
                        num_steps: int = 50,
                        class_labels: Optional[torch.Tensor] = None,
                        device: str = 'cuda') -> torch.Tensor:
        """
        Generate MIDI samples.
        
        Args:
            batch_size: Number of samples to generate
            seq_len: Sequence length
            num_steps: Number of denoising steps
            class_labels: Class conditioning
            device: Device to generate on
            
        Returns:
            Generated patches (B, N, D)
        """
        self.eval()
        
        if class_labels is None:
            class_labels = torch.randint(0, self.hparams.num_classes, (batch_size,), device=device)
            
        shape = (batch_size, seq_len, self.hparams.patch_dim)
        
        with torch.no_grad():
            samples = self.model.sample(
                shape=shape,
                num_steps=num_steps,
                device=device,
                class_labels=class_labels
            )
            
        return samples
        
    def samples_to_piano_roll(self, samples: torch.Tensor, original_shapes: torch.Tensor) -> list:
        """
        Convert generated patch samples back to piano roll format.
        
        Args:
            samples: Generated patches (B, N, D)
            original_shapes: Original piano roll shapes (B, 2)
            
        Returns:
            List of piano roll matrices
        """
        piano_rolls = []
        
        for i in range(samples.shape[0]):
            sample = samples[i]  # (N, D)
            orig_shape = tuple(original_shapes[i].tolist())
            
            # Convert back to piano roll
            piano_roll = self.patchify.unpatchify(sample, orig_shape)
            piano_rolls.append(piano_roll.detach().cpu().numpy())
            
        return piano_rolls


class MIDIDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for MIDI data."""
    
    def __init__(self,
                 data_path: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 patch_size: Tuple[int, int] = (16, 16),
                 sample_rate: int = 16,
                 pitch_range: Tuple[int, int] = (21, 108),
                 max_duration: float = 30.0,
                 min_notes: int = 10,
                 max_files: Optional[int] = None,
                 val_split: float = 0.1,
                 use_augmentation: bool = True,
                 augmentation_strength: str = 'medium'):
        
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        
        # Create train dataset with augmentation
        train_dataset = TextToMIDIDataset(
            data_path=self.hparams.data_path,
            patch_size=self.hparams.patch_size,
            sample_rate=self.hparams.sample_rate,
            pitch_range=self.hparams.pitch_range,
            max_duration=self.hparams.max_duration,
            min_notes=self.hparams.min_notes,
            max_files=self.hparams.max_files,
            use_cache=True,
            use_augmentation=self.hparams.use_augmentation,
            augmentation_strength=self.hparams.augmentation_strength,
            mode='train'
        )
        
        # Create val dataset without augmentation
        val_dataset = TextToMIDIDataset(
            data_path=self.hparams.data_path,
            patch_size=self.hparams.patch_size,
            sample_rate=self.hparams.sample_rate,
            pitch_range=self.hparams.pitch_range,
            max_duration=self.hparams.max_duration,
            min_notes=self.hparams.min_notes,
            max_files=self.hparams.max_files,
            use_cache=True,
            use_augmentation=False,  # No augmentation for validation
            mode='val'
        )
        
        # Split into train/val
        total_size = len(train_dataset)
        val_size = int(total_size * self.hparams.val_split)
        train_size = total_size - val_size
        
        # Use random split for now (ideally should be fixed indices)
        self.train_dataset, _ = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        _, self.val_dataset = torch.utils.data.random_split(
            val_dataset, [train_size, val_size]
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn_text_to_midi,
            pin_memory=True
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn_text_to_midi,
            pin_memory=True
        )


def train_midi_dit(config: Dict[str, Any]):
    """
    Train MIDI DiT model with given configuration.
    
    Args:
        config: Training configuration dictionary
    """
    
    # Setup data module
    data_module = MIDIDataModule(**config['data'])
    
    # Setup model
    model = MIDIDiTLightningModule(**config['model'])
    
    # Setup trainer
    trainer = pl.Trainer(**config['trainer'])
    
    # Train model
    trainer.fit(model, data_module)
    
    return model, trainer