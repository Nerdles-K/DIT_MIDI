"""
MIDI Dataset Module for BabySlakh - Text-to-MIDI Version
=========================================================

This module handles MIDI data preprocessing for Text-to-MIDI generation.
It extracts musical features from MIDI files and converts them to text descriptions,
then pairs them with piano roll matrices for training.

Key Features:
- Automatic text caption generation from MIDI features
- Support for instrument, tempo, key, mood descriptors
- Text encoding with CLIP or simple embeddings
- (Text, MIDI) paired dataset for conditional generation
"""

import warnings
# Suppress MIDI file format warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import glob
import random
import json
from typing import List, Optional, Dict, Tuple, Union
from collections import defaultdict, Counter
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pretty_midi
from torch.utils.data import Dataset, DataLoader
import webdataset as wds
from tqdm import tqdm

# Import augmentation module
try:
    from main.data.data_augmentation import AugmentationPipeline
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    print("Warning: Data augmentation module not available")


class MIDIPatchify:
    """Convert piano roll matrices to patch sequences for DiT input."""
    
    def __init__(self, patch_size: Tuple[int, int] = (16, 16)):
        """
        Args:
            patch_size: (time_patch_size, pitch_patch_size) for patchification
        """
        self.patch_size = patch_size
        
    def patchify(self, piano_roll: np.ndarray) -> torch.Tensor:
        """
        Convert piano roll to patch sequence.
        
        Args:
            piano_roll: Shape (T, P) where T is time steps, P is pitch dimension
            
        Returns:
            patches: Shape (num_patches, patch_size[0] * patch_size[1])
        """
        T, P = piano_roll.shape
        t_patch, p_patch = self.patch_size
        
        # Pad to make divisible by patch size
        T_pad = ((T + t_patch - 1) // t_patch) * t_patch
        P_pad = ((P + p_patch - 1) // p_patch) * p_patch
        
        padded = np.pad(piano_roll, ((0, T_pad - T), (0, P_pad - P)), 'constant')
        
        # Reshape to patches
        num_t_patches = T_pad // t_patch
        num_p_patches = P_pad // p_patch
        
        patches = padded.reshape(num_t_patches, t_patch, num_p_patches, p_patch)
        patches = np.transpose(patches, (0, 2, 1, 3))  # (num_t_patches, num_p_patches, t_patch, p_patch)
        patches = patches.reshape(-1, t_patch * p_patch)  # (num_patches, patch_dim)
        
        return torch.from_numpy(patches).float()
    
    def unpatchify(self, patches: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Convert patch sequence back to piano roll.
        
        Args:
            patches: Shape (num_patches, patch_dim)
            original_shape: (T, P) original piano roll shape
            
        Returns:
            piano_roll: Shape (T, P)
        """
        T, P = original_shape
        t_patch, p_patch = self.patch_size
        
        # Calculate padded dimensions
        T_pad = ((T + t_patch - 1) // t_patch) * t_patch
        P_pad = ((P + p_patch - 1) // p_patch) * p_patch
        
        num_t_patches = T_pad // t_patch
        num_p_patches = P_pad // p_patch
        
        # Reshape patches
        patches = patches.reshape(num_t_patches, num_p_patches, t_patch, p_patch)
        patches = patches.permute(0, 2, 1, 3)  # (num_t_patches, t_patch, num_p_patches, p_patch)
        
        # Reconstruct full matrix
        reconstructed = patches.reshape(T_pad, P_pad)
        
        # Crop to original size
        return reconstructed[:T, :P]


class MusicFeatureExtractor:
    """Extract musical features from MIDI for text caption generation."""
    
    # Instrument mapping
    INSTRUMENT_NAMES = {
        0: "piano", 1: "bright_piano", 4: "electric_piano",
        24: "acoustic_guitar", 25: "electric_guitar", 32: "bass",
        40: "violin", 41: "viola", 48: "strings",
        56: "trumpet", 64: "saxophone", 73: "flute",
        80: "synth_lead", 88: "synth_pad"
    }
    
    # Mood keywords based on tempo and key
    TEMPO_MOODS = {
        "very_slow": ["calm", "peaceful", "meditative"],
        "slow": ["sad", "melancholic", "emotional"],
        "moderate": ["relaxed", "smooth", "gentle"],
        "fast": ["energetic", "happy", "upbeat"],
        "very_fast": ["exciting", "intense", "dramatic"]
    }
    
    def __init__(self):
        pass
    
    def extract_features(self, midi_path: str) -> Dict[str, any]:
        """
        Extract comprehensive musical features from MIDI file.
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            Dictionary containing:
            - instruments: List of instrument names
            - tempo: Average tempo in BPM
            - tempo_category: slow/moderate/fast/very_fast
            - key: Estimated key signature
            - duration: Total duration in seconds
            - note_density: Notes per second
            - pitch_range: (min, max) pitch
            - avg_velocity: Average note velocity
            - text_description: Generated text caption
        """
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            
            # Extract instruments
            instruments = []
            for inst in midi.instruments:
                if not inst.is_drum and len(inst.notes) > 0:
                    prog = inst.program
                    inst_name = self.INSTRUMENT_NAMES.get(prog, "unknown")
                    instruments.append(inst_name)
            
            # Remove duplicates while preserving order
            instruments = list(dict.fromkeys(instruments))
            
            # Extract tempo
            tempo_changes = midi.get_tempo_changes()
            if len(tempo_changes[1]) > 0:
                avg_tempo = np.mean(tempo_changes[1])
            else:
                avg_tempo = 120.0  # Default
            
            # Categorize tempo
            if avg_tempo < 60:
                tempo_cat = "very_slow"
            elif avg_tempo < 90:
                tempo_cat = "slow"
            elif avg_tempo < 120:
                tempo_cat = "moderate"
            elif avg_tempo < 150:
                tempo_cat = "fast"
            else:
                tempo_cat = "very_fast"
            
            # Extract duration
            duration = midi.get_end_time()
            
            # Count notes and calculate density
            total_notes = sum(len(inst.notes) for inst in midi.instruments if not inst.is_drum)
            note_density = total_notes / duration if duration > 0 else 0
            
            # Extract pitch range
            all_pitches = []
            all_velocities = []
            for inst in midi.instruments:
                if not inst.is_drum:
                    for note in inst.notes:
                        all_pitches.append(note.pitch)
                        all_velocities.append(note.velocity)
            
            pitch_range = (min(all_pitches), max(all_pitches)) if all_pitches else (60, 72)
            avg_velocity = np.mean(all_velocities) if all_velocities else 64
            
            # Estimate key (simplified - use pitch class distribution)
            key = self._estimate_key(all_pitches) if all_pitches else "C"
            
            # Generate text description
            text_description = self._generate_text_description(
                instruments, tempo_cat, avg_tempo, key, note_density, avg_velocity
            )
            
            return {
                "instruments": instruments,
                "tempo": avg_tempo,
                "tempo_category": tempo_cat,
                "key": key,
                "duration": duration,
                "note_density": note_density,
                "pitch_range": pitch_range,
                "avg_velocity": avg_velocity,
                "text_description": text_description
            }
            
        except Exception as e:
            print(f"Error extracting features from {midi_path}: {e}")
            return None
    
    def _estimate_key(self, pitches: List[int]) -> str:
        """Estimate key signature from pitch distribution."""
        # Count pitch classes
        pitch_classes = [p % 12 for p in pitches]
        pc_counts = Counter(pitch_classes)
        
        # Major and minor key profiles (Krumhansl-Schmuckler)
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        
        # Calculate correlation for each key
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        best_key = "C"
        best_corr = -1
        
        for shift in range(12):
            # Test major key
            shifted_counts = [(pc_counts.get((i + shift) % 12, 0)) for i in range(12)]
            corr = np.corrcoef(shifted_counts, major_profile)[0, 1] if len(shifted_counts) > 1 else 0
            if corr > best_corr:
                best_corr = corr
                best_key = keys[shift]
        
        return best_key
    
    def _generate_text_description(self, instruments, tempo_cat, tempo, key, 
                                   note_density, avg_velocity) -> str:
        """Generate natural language description of the music."""
        
        # Start with instruments
        if len(instruments) == 0:
            desc = "musical piece"
        elif len(instruments) == 1:
            desc = f"{instruments[0]}"
        elif len(instruments) == 2:
            desc = f"{instruments[0]} and {instruments[1]}"
        else:
            desc = f"{', '.join(instruments[:-1])}, and {instruments[-1]}"
        
        # Add tempo/mood
        moods = self.TEMPO_MOODS.get(tempo_cat, ["musical"])
        mood = random.choice(moods)
        
        # Add intensity based on velocity
        if avg_velocity > 100:
            intensity = "powerful"
        elif avg_velocity > 80:
            intensity = "expressive"
        elif avg_velocity < 60:
            intensity = "gentle"
        else:
            intensity = ""
        
        # Construct description
        parts = [desc]
        
        if intensity:
            parts.append(f"{intensity} {mood} melody")
        else:
            parts.append(f"{mood} melody")
        
        parts.append(f"in {key}")
        
        if note_density > 5:
            parts.append("with rich harmonies")
        
        return " ".join(parts)


class MIDIProcessor:
    """Process MIDI files to piano roll matrices."""
    
    def __init__(self, 
                 sample_rate: int = 16,  # 1/16 notes per step
                 pitch_range: Tuple[int, int] = (21, 108),  # 88 keys
                 max_duration: float = 30.0,  # seconds
                 min_notes: int = 10):
        """
        Args:
            sample_rate: Quantization rate (steps per quarter note)
            pitch_range: (min_pitch, max_pitch) MIDI note numbers
            max_duration: Maximum track duration in seconds
            min_notes: Minimum number of notes to keep track
        """
        self.sample_rate = sample_rate
        self.pitch_range = pitch_range
        self.max_duration = max_duration
        self.min_notes = min_notes
        self.pitch_dim = pitch_range[1] - pitch_range[0] + 1
        
    def midi_to_piano_roll(self, midi_path: str, track_name: str = None) -> Optional[np.ndarray]:
        """
        Convert MIDI file to piano roll matrix.
        
        Args:
            midi_path: Path to MIDI file
            track_name: Specific track name to extract (if None, use all tracks)
            
        Returns:
            piano_roll: Shape (T, P) where T is time steps, P is pitch dimension
                       Values are normalized velocities in [-1, 1], 0 for silence
        """
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            
            if midi.get_end_time() > self.max_duration:
                return None
                
            # Calculate time steps
            total_steps = int(midi.get_end_time() * self.sample_rate * 4)  # 4 quarter notes per second assumed
            
            if total_steps == 0:
                return None
                
            # Initialize piano roll
            piano_roll = np.zeros((total_steps, self.pitch_dim))
            
            # Process instruments
            total_notes = 0
            for instrument in midi.instruments:
                if instrument.is_drum:
                    continue
                    
                if track_name and instrument.name != track_name:
                    continue
                    
                for note in instrument.notes:
                    # Check pitch range
                    if note.pitch < self.pitch_range[0] or note.pitch > self.pitch_range[1]:
                        continue
                        
                    # Convert to time steps
                    start_step = int(note.start * self.sample_rate * 4)
                    end_step = int(note.end * self.sample_rate * 4)
                    
                    if start_step >= total_steps or end_step <= 0:
                        continue
                        
                    start_step = max(0, start_step)
                    end_step = min(total_steps, end_step)
                    
                    # Convert pitch to index
                    pitch_idx = note.pitch - self.pitch_range[0]
                    
                    # Normalize velocity to [-1, 1]
                    velocity = (note.velocity / 127.0) * 2.0 - 1.0
                    
                    # Fill piano roll
                    piano_roll[start_step:end_step, pitch_idx] = velocity
                    total_notes += 1
            
            # Filter out tracks with too few notes
            if total_notes < self.min_notes:
                return None
                
            return piano_roll
            
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            return None


class TextToMIDIDataset(Dataset):
    """Dataset class for Text-to-MIDI generation using BabySlakh data."""
    
    def __init__(self,
                 data_path: str,
                 patch_size: Tuple[int, int] = (16, 16),
                 sample_rate: int = 16,
                 pitch_range: Tuple[int, int] = (21, 108),
                 max_duration: float = 30.0,
                 min_notes: int = 10,
                 max_files: Optional[int] = None,
                 cache_dir: Optional[str] = None,
                 use_cache: bool = True,
                 use_augmentation: bool = True,
                 augmentation_strength: str = 'medium',
                 mode: str = 'train'):
        """
        Args:
            data_path: Path to BabySlakh dataset
            patch_size: Patch size for DiT input
            sample_rate: Quantization rate
            pitch_range: MIDI pitch range
            max_duration: Maximum track duration
            min_notes: Minimum notes to keep track
            max_files: Maximum number of files to load
            cache_dir: Directory to cache processed data
            use_cache: Whether to use cached data
            use_augmentation: Whether to apply data augmentation
            augmentation_strength: 'light', 'medium', or 'heavy'
            mode: 'train' or 'val' (determines augmentation application)
        """
        self.data_path = data_path
        self.processor = MIDIProcessor(sample_rate, pitch_range, max_duration, min_notes)
        self.feature_extractor = MusicFeatureExtractor()
        self.patchify = MIDIPatchify(patch_size)
        
        # Setup data augmentation
        self.use_augmentation = use_augmentation and AUGMENTATION_AVAILABLE
        self.mode = mode
        if self.use_augmentation:
            self.augmentation = AugmentationPipeline(
                mode=mode,
                augmentation_strength=augmentation_strength
            )
            print(f"Data augmentation enabled (strength={augmentation_strength}, mode={mode})")
        else:
            self.augmentation = None
        
        # Setup cache
        self.cache_dir = cache_dir or os.path.join(data_path, ".cache")
        self.use_cache = use_cache
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Find all MIDI files
        self.midi_files = []
        self.text_features = []  # Store (text, features) pairs
        self._load_file_list(max_files)
        
        print(f"Loaded {len(self.midi_files)} MIDI files with text descriptions")
        
    def _load_file_list(self, max_files: Optional[int]):
        """Load list of MIDI files and extract text features."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset path {self.data_path} not found")
        
        # Check for cached features
        cache_file = os.path.join(self.cache_dir, "text_features.json")
        
        if self.use_cache and os.path.exists(cache_file):
            print(f"Loading cached features from {cache_file}...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                self.midi_files = cache_data['midi_files']
                self.text_features = cache_data['text_features']
            if max_files:
                self.midi_files = self.midi_files[:max_files]
                self.text_features = self.text_features[:max_files]
            print(f"Loaded {len(self.midi_files)} cached entries")
            return
        
        # Search for MIDI files in BabySlakh structure
        print("Scanning for MIDI files...")
        pattern = os.path.join(self.data_path, "**", "*.mid")
        all_files = glob.glob(pattern, recursive=True)
        
        if max_files:
            all_files = all_files[:max_files]
        
        # Extract features from each file
        print("Extracting text features from MIDI files...")
        valid_files = []
        valid_features = []
        
        for midi_path in tqdm(all_files, desc="Processing MIDI files"):
            features = self.feature_extractor.extract_features(midi_path)
            if features is not None:
                valid_files.append(midi_path)
                valid_features.append(features)
        
        self.midi_files = valid_files
        self.text_features = valid_features
        
        # Cache the results
        if self.use_cache:
            print(f"Caching features to {cache_file}...")
            with open(cache_file, 'w') as f:
                json.dump({
                    'midi_files': self.midi_files,
                    'text_features': self.text_features
                }, f)
        
        print(f"Processed {len(self.midi_files)} MIDI files")
        
    def __len__(self):
        return len(self.midi_files)
        
    def __getitem__(self, idx):
        # Try up to 5 different files to avoid infinite recursion
        for attempt in range(5):
            try_idx = (idx + attempt) % len(self.midi_files)
            midi_path = self.midi_files[try_idx]
            text_feature = self.text_features[try_idx]
            
            # Process MIDI to piano roll
            piano_roll = self.processor.midi_to_piano_roll(midi_path)
            
            if piano_roll is not None:
                # Convert to patches
                patches = self.patchify.patchify(piano_roll)
                
                return {
                    'patches': patches,
                    'original_shape': torch.tensor(piano_roll.shape),
                    'text_description': text_feature['text_description'],
                    'instruments': text_feature['instruments'],
                    'tempo': text_feature['tempo'],
                    'key': text_feature['key'],
                    'file_path': midi_path
                }
        
        # If all attempts failed, return a dummy sample
        dummy_shape = (128, 88)  # Default piano roll shape
        dummy_patches = torch.zeros(48, 256)  # Default patch size
        
        return {
            'patches': dummy_patches,
            'original_shape': torch.tensor(dummy_shape),
            'text_description': "simple piano melody",
            'instruments': ["piano"],
            'tempo': 120.0,
            'key': "C",
            'file_path': "dummy"
        }


def collate_fn_text_to_midi(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for Text-to-MIDI dataset.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched data with text conditioning
    """
    # Find max sequence length in batch
    max_seq_len = max([item['patches'].shape[0] for item in batch])
    
    batch_size = len(batch)
    patch_dim = batch[0]['patches'].shape[1]
    
    # Initialize tensors
    patches = torch.zeros(batch_size, max_seq_len, patch_dim)
    masks = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    original_shapes = torch.stack([item['original_shape'] for item in batch])
    
    # Fill tensors
    for i, item in enumerate(batch):
        seq_len = item['patches'].shape[0]
        patches[i, :seq_len] = item['patches']
        masks[i, :seq_len] = True
    
    # Collect text data
    text_descriptions = [item['text_description'] for item in batch]
    instruments = [item['instruments'] for item in batch]
    tempos = torch.tensor([item['tempo'] for item in batch], dtype=torch.float32)
    keys = [item['key'] for item in batch]
        
    return {
        'patches': patches,
        'masks': masks,
        'original_shapes': original_shapes,
        'text_descriptions': text_descriptions,
        'instruments': instruments,
        'tempos': tempos,
        'keys': keys,
        'file_paths': [item['file_path'] for item in batch]
    }

def create_text_to_midi_dataset(
    path: str,
    patch_size: Tuple[int, int] = (16, 16),
    sample_rate: int = 16,
    pitch_range: Tuple[int, int] = (21, 108),
    max_duration: float = 30.0,
    min_notes: int = 10,
    max_files: Optional[int] = None,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    use_augmentation: bool = True,
    augmentation_strength: str = 'medium',
    mode: str = 'train',
    **kwargs
) -> TextToMIDIDataset:
    """
    Factory function to create Text-to-MIDI dataset.
    
    Args:
        path: Path to dataset
        patch_size: Patch size for patchification
        sample_rate: Quantization rate
        pitch_range: MIDI pitch range  
        max_duration: Maximum duration in seconds
        min_notes: Minimum notes per track
        max_files: Maximum files to load
        cache_dir: Cache directory for processed data
        use_cache: Whether to use cached features
        use_augmentation: Whether to use data augmentation
        augmentation_strength: 'light', 'medium', or 'heavy'
        mode: 'train' or 'val'
        
    Returns:
        Dataset instance
    """
    return TextToMIDIDataset(
        data_path=path,
        patch_size=patch_size,
        sample_rate=sample_rate,
        pitch_range=pitch_range,
        max_duration=max_duration,
        min_notes=min_notes,
        max_files=max_files,
        cache_dir=cache_dir,
        use_cache=use_cache,
        use_augmentation=use_augmentation,
        augmentation_strength=augmentation_strength,
        mode=mode
    )

# Keep old name for backward compatibility
def create_babyslakh_dataset(*args, **kwargs):
    """Alias for backward compatibility."""
    return create_text_to_midi_dataset(*args, **kwargs)

# Update class alias
BabySlakhMIDIDataset = TextToMIDIDataset


def create_midi_dataloader(
    dataset: TextToMIDIDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader for Text-to-MIDI dataset.
    
    Args:
        dataset: Text-to-MIDI dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_text_to_midi,
        pin_memory=True,
        **kwargs
    )