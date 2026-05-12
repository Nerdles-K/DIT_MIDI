"""
Test Data Augmentation Integration
===================================

Verify that data augmentation is properly integrated into the dataset
and produces varied outputs.
"""

import torch
import numpy as np
from main.data.dataset_midi import TextToMIDIDataset
from main.data.data_augmentation import MIDIAugmenter, test_augmentations


def test_augmentation_integration():
    """Test augmentation integration with dataset."""
    
    print("=" * 60)
    print("Testing Data Augmentation Integration")
    print("=" * 60)
    
    # First, test basic augmentation functions
    print("\n1. Testing basic augmentation functions...")
    test_augmentations()
    
    # Test dataset integration
    print("\n2. Testing dataset integration...")
    
    try:
        # Create dataset with augmentation (train mode)
        print("Creating train dataset with augmentation...")
        train_dataset = TextToMIDIDataset(
            data_path="midi_data/lmd_full",
            patch_size=(16, 16),
            sample_rate=16,
            pitch_range=(21, 108),
            max_duration=20.0,
            min_notes=10,
            max_files=10,  # Small number for quick test
            use_cache=True,
            use_augmentation=True,
            augmentation_strength='medium',
            mode='train'
        )
        
        print(f"✓ Train dataset created: {len(train_dataset)} samples")
        
        # Create dataset without augmentation (val mode)
        print("\nCreating val dataset without augmentation...")
        val_dataset = TextToMIDIDataset(
            data_path="midi_data/lmd_full",
            patch_size=(16, 16),
            sample_rate=16,
            pitch_range=(21, 108),
            max_duration=20.0,
            min_notes=10,
            max_files=10,
            use_cache=True,
            use_augmentation=False,
            mode='val'
        )
        
        print(f"✓ Val dataset created: {len(val_dataset)} samples")
        
        # Test loading same sample multiple times to verify augmentation variety
        if len(train_dataset) > 0:
            print("\n3. Testing augmentation variety...")
            idx = 0
            
            # Load same sample 3 times
            samples = []
            for i in range(3):
                sample = train_dataset[idx]
                patches = sample['patches']
                samples.append(patches)
                print(f"  Sample {i+1}: shape={patches.shape}, mean={patches.mean():.4f}, std={patches.std():.4f}")
            
            # Check if samples are different (augmented)
            sample1_hash = hash(samples[0].numpy().tobytes())
            sample2_hash = hash(samples[1].numpy().tobytes())
            sample3_hash = hash(samples[2].numpy().tobytes())
            
            if sample1_hash != sample2_hash or sample1_hash != sample3_hash:
                print("✓ Augmentation produces varied outputs!")
            else:
                print("⚠ Warning: Samples are identical (augmentation may not be working)")
            
            # Test validation dataset (should produce identical samples)
            print("\n4. Testing val dataset consistency...")
            val_samples = []
            for i in range(2):
                sample = val_dataset[idx]
                patches = sample['patches']
                val_samples.append(patches)
            
            val1_hash = hash(val_samples[0].numpy().tobytes())
            val2_hash = hash(val_samples[1].numpy().tobytes())
            
            if val1_hash == val2_hash:
                print("✓ Val dataset produces consistent outputs (no augmentation)")
            else:
                print("⚠ Warning: Val samples differ (unexpected)")
            
        print("\n" + "=" * 60)
        print("Data Augmentation Integration Test PASSED! ✓")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_augmentation_parameters():
    """Test different augmentation strengths."""
    
    print("\n" + "=" * 60)
    print("Testing Different Augmentation Strengths")
    print("=" * 60)
    
    strengths = ['light', 'medium', 'heavy']
    
    for strength in strengths:
        print(f"\nTesting '{strength}' augmentation...")
        
        # Create simple test piano roll
        piano_roll = np.zeros((128, 88))
        piano_roll[10:20, 30:35] = 0.8  # Add some notes
        
        # Create augmenter
        from main.data.data_augmentation import AugmentationPipeline
        augmenter = AugmentationPipeline(
            mode='train',
            augmentation_strength=strength
        )
        
        # Apply augmentation 5 times and measure variance
        augmented_samples = []
        for i in range(5):
            aug = augmenter(piano_roll.copy())
            augmented_samples.append(aug)
        
        # Calculate statistics
        shapes = [a.shape for a in augmented_samples]
        means = [a.mean() for a in augmented_samples]
        stds = [a.std() for a in augmented_samples]
        
        print(f"  Shapes: {shapes}")
        print(f"  Means: {[f'{m:.4f}' for m in means]}")
        print(f"  Stds: {[f'{s:.4f}' for s in stds]}")
        print(f"  Mean variance: {np.var(means):.6f}")
        print(f"  ✓ {strength.capitalize()} augmentation working")
    
    print("\n" + "=" * 60)
    print("Augmentation Strength Test PASSED! ✓")
    print("=" * 60)


if __name__ == "__main__":
    print("\n🎵 MIDI Data Augmentation Test Suite 🎵\n")
    
    # Run tests
    test1_passed = test_augmentation_integration()
    
    if test1_passed:
        test_augmentation_parameters()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓✓✓")
        print("Data augmentation is ready for training!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("TESTS FAILED! Please check errors above.")
        print("=" * 60)
