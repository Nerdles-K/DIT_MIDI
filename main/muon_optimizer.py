"""
Muon Optimizer Implementation
==============================

Muon: Momentum Orthogonalized by Newton-schulz
A high-performance optimizer designed for large language models and transformers.

Reference: https://github.com/KellerJordan/Muon
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Tuple


def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    We opt to use a quintic iteration whose coefficients are selected to maximize
    the slope at zero. For the purpose of minimizing steps, it turns out to be empirically
    effective to keep increasing the slope at zero even beyond the point where the
    iteration no longer converges all the way to one everywhere on the interval.
    This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to
    hurt model performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(Optimizer):
    """
    Muon - Momentum Orthogonalized by Newton-schulz
    
    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.
    
    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
      parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): 
            Learning rate (default: 0.02).
        momentum (float): 
            Momentum value in  the range [0,1) (default: 0.95).
        nesterov (bool): 
            Enables Nesterov momentum (default: True).
        ns_steps (int):
            Number of Newton-Schulz iterations (default: 5).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                state['step'] += 1

                # Standard momentum update
                buf.mul_(momentum).add_(g)
                
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Orthogonalize the update for 2D parameters
                if g.dim() == 2:
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                
                # Apply update
                p.data.add_(g, alpha=-lr)

        return loss


class MuonWithDecoupledWeightDecay(Optimizer):
    """
    Muon with Decoupled Weight Decay (similar to AdamW's approach)
    
    This variant adds decoupled weight decay, which is applied directly to parameters
    without being orthogonalized. This can help with regularization while maintaining
    Muon's optimization benefits.
    
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): 
            Learning rate (default: 0.02).
        momentum (float): 
            Momentum value in the range [0,1) (default: 0.95).
        nesterov (bool): 
            Enables Nesterov momentum (default: True).
        ns_steps (int):
            Number of Newton-Schulz iterations (default: 5).
        weight_decay (float):
            Weight decay coefficient (default: 0.01).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, 
                 ns_steps=5, weight_decay=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, 
                       ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Apply decoupled weight decay
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                g = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                state['step'] += 1

                # Standard momentum update
                buf.mul_(momentum).add_(g)
                
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Orthogonalize the update for 2D parameters
                if g.dim() == 2:
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                
                # Apply update
                p.data.add_(g, alpha=-lr)

        return loss
