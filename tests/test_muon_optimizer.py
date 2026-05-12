"""
Test Muon Optimizer Integration
================================

Quick test to verify Muon optimizer works correctly with the model.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from main.muon_optimizer import Muon, MuonWithDecoupledWeightDecay

def test_muon_basic():
    """Test basic Muon optimizer functionality."""
    print("Testing Muon optimizer...")
    
    # Create simple 2D parameters
    param1 = torch.randn(128, 256, requires_grad=True)
    param2 = torch.randn(256, 128, requires_grad=True)
    
    optimizer = Muon([param1, param2], lr=0.02, momentum=0.95)
    
    # Simulate forward and backward
    loss = (param1.sum() + param2.sum()) ** 2
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    print("✓ Basic Muon test passed")

def test_muon_with_weight_decay():
    """Test Muon with weight decay."""
    print("\nTesting Muon with weight decay...")
    
    param = torch.randn(100, 200, requires_grad=True)
    
    optimizer = MuonWithDecoupledWeightDecay(
        [param], 
        lr=0.02, 
        momentum=0.95,
        weight_decay=0.01
    )
    
    loss = param.sum() ** 2
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print("✓ Muon with weight decay test passed")

def test_mixed_optimizer():
    """Test mixed Muon + AdamW setup."""
    print("\nTesting mixed Muon + AdamW setup...")
    
    # 2D params for Muon
    muon_params = [
        torch.randn(128, 256, requires_grad=True),
        torch.randn(256, 128, requires_grad=True)
    ]
    
    # 1D params for AdamW
    adamw_params = [
        torch.randn(256, requires_grad=True),  # bias
        torch.randn(128, requires_grad=True)   # norm
    ]
    
    muon_opt = MuonWithDecoupledWeightDecay(
        muon_params,
        lr=0.06,  # 3x higher
        momentum=0.95,
        weight_decay=0.01
    )
    
    adamw_opt = torch.optim.AdamW(
        adamw_params,
        lr=0.02,
        weight_decay=0.01
    )
    
    # Simulate training step
    loss = sum(p.sum() for p in muon_params + adamw_params) ** 2
    loss.backward()
    
    muon_opt.step()
    adamw_opt.step()
    
    muon_opt.zero_grad()
    adamw_opt.zero_grad()
    
    print("✓ Mixed optimizer test passed")

def test_optimizer_states():
    """Test optimizer state management."""
    print("\nTesting optimizer state management...")
    
    param = torch.randn(100, 100, requires_grad=True)
    optimizer = Muon([param], lr=0.02, momentum=0.95)
    
    # First step
    loss = param.sum() ** 2
    loss.backward()
    optimizer.step()
    
    # Check state
    assert len(optimizer.state) == 1, "State should have 1 param"
    state = optimizer.state[param]
    assert 'momentum_buffer' in state, "Should have momentum buffer"
    assert 'step' in state, "Should have step counter"
    assert state['step'] == 1, "Step should be 1"
    
    print("✓ Optimizer state test passed")

if __name__ == "__main__":
    print("=" * 60)
    print("Muon Optimizer Test Suite")
    print("=" * 60)
    
    try:
        test_muon_basic()
        test_muon_with_weight_decay()
        test_mixed_optimizer()
        test_optimizer_states()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nMuon optimizer is ready to use.")
        print("To train with Muon, simply run: python train_midi.py +exp=train_babyslakh_midi_dit")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
