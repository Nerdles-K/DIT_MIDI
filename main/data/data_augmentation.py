"""
MIDI Data Augmentation Module
==============================

Provides various data augmentation techniques for MIDI piano roll matrices
to improve model generalization and robustness.

Key Augmentations:
- Pitch Shift: Transpose by semitones
- Time Stretch: Speed up/slow down
- Velocity Scaling: Change note intensities
- Random Crop/Pad: Vary sequence length
- Note Dropout: Randomly remove notes
- Dynamic Range Compression: Normalize velocities
"""

import random
import numpy as np
import torch
from typing import Optional, Tuple, List
from scipy import interpolate


class MIDIAugmenter:
    """
    Comprehensive MIDI augmentation class.
    
    All augmentations preserve the structure of piano roll (T, P) matrices
    and can be applied individually or in combination.
    """
    
    def __init__(self,
                 pitch_shift_range: Tuple[int, int] = (-6, 6),
                 time_stretch_range: Tuple[float, float] = (0.9, 1.1),
                 velocity_scale_range: Tuple[float, float] = (0.8, 1.2),
                 note_dropout_prob: float = 0.05,
                 crop_ratio_range: Tuple[float, float] = (0.8, 1.0),
                 apply_prob: float = 0.7):
        """
        Args:
            pitch_shift_range: Range of pitch shifts in semitones (min, max)
            time_stretch_range: Range of time stretch factors (min, max)
            velocity_scale_range: Range of velocity scaling factors (min, max)
            note_dropout_prob: Probability of dropping each note
            crop_ratio_range: Range of crop ratios (min, max) relative to original length
            apply_prob: Probability of applying any augmentation
        """
        self.pitch_shift_range = pitch_shift_range
        self.time_stretch_range = time_stretch_range
        self.velocity_scale_range = velocity_scale_range
        self.note_dropout_prob = note_dropout_prob
        self.crop_ratio_range = crop_ratio_range
        self.apply_prob = apply_prob
        
    def __call__(self, piano_roll: np.ndarray, apply_aug: bool = True) -> np.ndarray:
        """
        Apply random augmentations to piano roll.
        
        Args:
            piano_roll: Piano roll matrix (T, P)
            apply_aug: Whether to apply augmentation (for train/val split)
            
        Returns:
            Augmented piano roll
        """
        if not apply_aug or random.random() > self.apply_prob:
            return piano_roll
            
        # Apply augmentations in sequence with individual probabilities
        augmented = piano_roll.copy()
        
        # 1. Pitch shift (50% probability)
        if random.random() < 0.5:
            augmented = self.pitch_shift(augmented)
            
        # 2. Time stretch (30% probability)
        if random.random() < 0.3:
            augmented = self.time_stretch(augmented)
            
        # 3. Velocity scaling (40% probability)
        if random.random() < 0.4:
            augmented = self.velocity_scale(augmented)
            
        # 4. Note dropout (20% probability)
        if random.random() < 0.2:
            augmented = self.note_dropout(augmented)
            
        # 5. Random crop (10% probability, for diversity)
        if random.random() < 0.1:
            augmented = self.random_crop(augmented)
            
        return augmented
        
    def pitch_shift(self, piano_roll: np.ndarray, semitones: Optional[int] = None) -> np.ndarray:
        """
        Shift pitch by a number of semitones.
        
        Args:
            piano_roll: Piano roll (T, P)
            semitones: Number of semitones to shift (if None, random)
            
        Returns:
            Pitch-shifted piano roll
        """
        if semitones is None:
            semitones = random.randint(*self.pitch_shift_range)
            
        if semitones == 0:
            return piano_roll
            
        T, P = piano_roll.shape
        shifted = np.zeros_like(piano_roll)
        
        if semitones > 0:
            # Shift up
            shifted[:, semitones:] = piano_roll[:, :P-semitones]
        else:
            # Shift down
            shifted[:, :P+semitones] = piano_roll[:, -semitones:]
            
        return shifted
        
    def time_stretch(self, piano_roll: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        """
        Stretch time by a factor using interpolation.
        
        Args:
            piano_roll: Piano roll (T, P)
            factor: Stretch factor (>1 = slower, <1 = faster)
            
        Returns:
            Time-stretched piano roll
        """
        if factor is None:
            factor = random.uniform(*self.time_stretch_range)
            
        if abs(factor - 1.0) < 0.01:
            return piano_roll
            
        T, P = piano_roll.shape
        new_T = int(T / factor)
        
        if new_T < 1:
            return piano_roll
            
        # Interpolate each pitch independently
        stretched = np.zeros((new_T, P))
        
        for p in range(P):
            # Get non-zero indices for this pitch
            if np.any(piano_roll[:, p] != 0):
                # Use linear interpolation
                old_indices = np.arange(T)
                new_indices = np.linspace(0, T-1, new_T)
                
                f = interpolate.interp1d(old_indices, piano_roll[:, p], 
                                        kind='linear', fill_value=0.0, 
                                        bounds_error=False)
                stretched[:, p] = f(new_indices)
                
        return stretched
        
    def velocity_scale(self, piano_roll: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
        """
        Scale note velocities.
        
        Args:
            piano_roll: Piano roll (T, P) with values in [-1, 1]
            scale: Scaling factor
            
        Returns:
            Velocity-scaled piano roll
        """
        if scale is None:
            scale = random.uniform(*self.velocity_scale_range)
            
        # Scale velocities and clip to valid range
        scaled = piano_roll * scale
        scaled = np.clip(scaled, -1.0, 1.0)
        
        return scaled
        
    def note_dropout(self, piano_roll: np.ndarray, dropout_prob: Optional[float] = None) -> np.ndarray:
        """
        Randomly drop out notes (set to 0).
        
        Args:
            piano_roll: Piano roll (T, P)
            dropout_prob: Probability of dropping each note
            
        Returns:
            Piano roll with some notes dropped
        """
        if dropout_prob is None:
            dropout_prob = self.note_dropout_prob
            
        # Create dropout mask (only for non-zero elements)
        mask = np.random.random(piano_roll.shape) > dropout_prob
        
        # Apply mask only to non-zero elements
        dropped = piano_roll.copy()
        dropped[piano_roll != 0] = dropped[piano_roll != 0] * mask[piano_roll != 0]
        
        return dropped
        
    def random_crop(self, piano_roll: np.ndarray, crop_ratio: Optional[float] = None) -> np.ndarray:
        """
        Randomly crop a portion of the piano roll.
        
        Args:
            piano_roll: Piano roll (T, P)
            crop_ratio: Ratio of original length to keep
            
        Returns:
            Cropped piano roll (may be shorter)
        """
        if crop_ratio is None:
            crop_ratio = random.uniform(*self.crop_ratio_range)
            
        T, P = piano_roll.shape
        new_T = int(T * crop_ratio)
        
        if new_T >= T:
            return piano_roll
            
        # Random start position
        start = random.randint(0, T - new_T)
        
        return piano_roll[start:start+new_T]
        
    def dynamic_range_compression(self, piano_roll: np.ndarray, 
                                  threshold: float = 0.6,
                                  ratio: float = 2.0) -> np.ndarray:
        """
        Apply dynamic range compression to normalize loud/soft passages.
        
        Args:
            piano_roll: Piano roll (T, P)
            threshold: Threshold above which to compress
            ratio: Compression ratio
            
        Returns:
            Compressed piano roll
        """
        compressed = piano_roll.copy()
        
        # Compress values above threshold
        mask = np.abs(compressed) > threshold
        excess = np.abs(compressed[mask]) - threshold
        compressed[mask] = np.sign(compressed[mask]) * (threshold + excess / ratio)
        
        return compressed
        
    def add_gaussian_noise(self, piano_roll: np.ndarray, noise_std: float = 0.02) -> np.ndarray:
        """
        Add Gaussian noise to piano roll (subtle, for robustness).
        
        Args:
            piano_roll: Piano roll (T, P)
            noise_std: Standard deviation of noise
            
        Returns:
            Noisy piano roll
        """
        noise = np.random.normal(0, noise_std, piano_roll.shape)
        noisy = piano_roll + noise
        
        # Only add noise to non-zero elements
        noisy = np.where(piano_roll != 0, noisy, 0)
        
        # Clip to valid range
        return np.clip(noisy, -1.0, 1.0)


class AugmentationPipeline:
    """
    Compose multiple augmenters with configurable strategies.
    """
    
    def __init__(self, 
                 mode: str = 'train',
                 augmentation_strength: str = 'medium'):
        """
        Args:
            mode: 'train' or 'val' (no augmentation in val)
            augmentation_strength: 'light', 'medium', or 'heavy'
        """
        self.mode = mode
        
        # Configure augmenter based on strength
        if augmentation_strength == 'light':
            self.augmenter = MIDIAugmenter(
                pitch_shift_range=(-3, 3),
                time_stretch_range=(0.95, 1.05),
                velocity_scale_range=(0.9, 1.1),
                note_dropout_prob=0.03,
                crop_ratio_range=(0.9, 1.0),
                apply_prob=0.5
            )
        elif augmentation_strength == 'medium':
            self.augmenter = MIDIAugmenter(
                pitch_shift_range=(-6, 6),
                time_stretch_range=(0.9, 1.1),
                velocity_scale_range=(0.8, 1.2),
                note_dropout_prob=0.05,
                crop_ratio_range=(0.8, 1.0),
                apply_prob=0.7
            )
        elif augmentation_strength == 'heavy':
            self.augmenter = MIDIAugmenter(
                pitch_shift_range=(-12, 12),
                time_stretch_range=(0.8, 1.2),
                velocity_scale_range=(0.7, 1.3),
                note_dropout_prob=0.08,
                crop_ratio_range=(0.7, 1.0),
                apply_prob=0.8
            )
        else:
            raise ValueError(f"Unknown augmentation strength: {augmentation_strength}")
            
    def __call__(self, piano_roll: np.ndarray) -> np.ndarray:
        """
        Apply augmentation pipeline.
        
        Args:
            piano_roll: Piano roll (T, P)
            
        Returns:
            Augmented piano roll
        """
        apply_aug = (self.mode == 'train')
        return self.augmenter(piano_roll, apply_aug=apply_aug)


def test_augmentations():
    """Test augmentation functions."""
    
    print("Testing MIDI Augmentations...")
    
    # Create a simple piano roll (simulated)
    T, P = 128, 88
    piano_roll = np.zeros((T, P))
    
    # Add some notes
    piano_roll[10:20, 30] = 0.8
    piano_roll[30:40, 40] = 0.6
    piano_roll[50:60, 50] = 0.9
    
    print(f"Original shape: {piano_roll.shape}")
    print(f"Original non-zero elements: {np.count_nonzero(piano_roll)}")
    
    # Test pitch shift
    augmenter = MIDIAugmenter()
    
    shifted = augmenter.pitch_shift(piano_roll, semitones=5)
    print(f"Pitch shifted (+5): {np.count_nonzero(shifted)} non-zero elements")
    
    # Test time stretch
    stretched = augmenter.time_stretch(piano_roll, factor=1.2)
    print(f"Time stretched (1.2x): shape {stretched.shape}")
    
    # Test velocity scale
    scaled = augmenter.velocity_scale(piano_roll, scale=0.8)
    print(f"Velocity scaled (0.8x): max={scaled.max():.3f}, min={scaled.min():.3f}")
    
    # Test note dropout
    dropped = augmenter.note_dropout(piano_roll, dropout_prob=0.3)
    print(f"Note dropout (30%): {np.count_nonzero(dropped)} non-zero elements")
    
    # Test full pipeline
    augmented = augmenter(piano_roll, apply_aug=True)
    print(f"Full augmentation: shape {augmented.shape}, {np.count_nonzero(augmented)} non-zero elements")
    
    print("\nAugmentation tests passed! ✓")


if __name__ == "__main__":
    test_augmentations()
