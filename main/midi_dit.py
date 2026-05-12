"""
DiT with AdaLN-Zero for MIDI Generation
=======================================

Diffusion Transformer with Adaptive Layer Normalization and Zero Initialization
specifically designed for MIDI sequence generation.
"""

import math
import typing as tp
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from stable_audio_tools.models.blocks import FourierFeatures
from main.text_encoder import create_text_encoder, MUSIC_VOCAB


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with Zero Initialization.
    
    This is the core component that enables stable training and better conditioning.
    """
    
    def __init__(self, dim: int, cond_dim: int):
        """
        Args:
            dim: Feature dimension
            cond_dim: Conditioning dimension (time + class embeddings)
        """
        super().__init__()
        
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # MLP to generate scale (gamma) and shift (beta) parameters
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * dim, bias=True)
        )
        
        # Zero initialization for stable training
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, N, D)
            cond: Conditioning tensor (B, cond_dim)
            
        Returns:
            Normalized and conditioned tensor
        """
        # Layer normalization
        x_norm = self.norm(x)
        
        # Generate adaptive parameters
        cond_params = self.mlp(cond)  # (B, 2*D)
        gamma, beta = cond_params.chunk(2, dim=-1)  # Each (B, D)
        
        # Apply adaptive normalization: (1 + gamma) * norm(x) + beta
        return (1 + gamma.unsqueeze(1)) * x_norm + beta.unsqueeze(1)


class TimestepEmbedding(nn.Module):
    """Timestep embedding with Fourier features."""
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        # Fourier features for timestep
        self.fourier_features = FourierFeatures(1, dim)
        
        # MLP for timestep embedding
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep tensor (B,)
            
        Returns:
            Timestep embeddings (B, dim)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
            
        # Get Fourier features
        fourier_emb = self.fourier_features(t)  # (B, dim)
        
        # Apply MLP
        return self.mlp(fourier_emb)


class PatchEmbedding(nn.Module):
    """Convert patches to embeddings."""
    
    def __init__(self, patch_dim: int, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(patch_dim, embed_dim)
        
        # Initialize with proper scaling
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (B, N, patch_dim)
            
        Returns:
            Embeddings: (B, N, embed_dim)
        """
        return self.linear(patches)


class PositionalEmbedding(nn.Module):
    """Learnable positional embeddings."""
    
    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, N, D)
            
        Returns:
            x + positional embeddings
        """
        B, N, D = x.shape
        return x + self.pos_emb[:, :N, :]


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block with AdaLN-Zero.
    
    This implements the core DiT block with:
    - AdaLN-Zero normalization
    - Multi-head self-attention  
    - MLP with zero-initialized output
    - Residual connections with learnable gating
    """
    
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 cond_dim: int,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # AdaLN-Zero for attention
        self.norm1 = AdaLNZero(embed_dim, cond_dim)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # AdaLN-Zero for MLP
        self.norm2 = AdaLNZero(embed_dim, cond_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Zero initialization for MLP output
        nn.init.zeros_(self.mlp[-2].weight)
        nn.init.zeros_(self.mlp[-2].bias)
        
        # Learnable gating parameters (alpha) for residual connections
        self.gate_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2, bias=True)
        )
        
        # Zero initialization for gates
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)
        
    def forward(self, 
                x: torch.Tensor, 
                cond: torch.Tensor,
                mask: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, N, D)
            cond: Conditioning tensor (B, cond_dim)
            mask: Attention mask (B, N) or (B, N, N)
            
        Returns:
            Output tensor (B, N, D)
        """
        # Generate gating parameters
        gates = self.gate_mlp(cond)  # (B, 2)
        gate_attn, gate_mlp = gates.chunk(2, dim=-1)  # Each (B, 1)
        
        # Attention block with residual connection and gating
        x_norm1 = self.norm1(x, cond)
        
        if mask is not None:
            if mask.dim() == 2:  # (B, N) -> (N, N) for all samples in batch
                B, N = mask.shape
                # Create causal mask where padded positions are masked out
                # For now, we'll use the simpler key_padding_mask approach
                key_padding_mask = ~mask.bool()  # True for positions to ignore
                attn_out, _ = self.attention(x_norm1, x_norm1, x_norm1, key_padding_mask=key_padding_mask)
            else:
                attn_out, _ = self.attention(x_norm1, x_norm1, x_norm1, attn_mask=mask)
        else:
            attn_out, _ = self.attention(x_norm1, x_norm1, x_norm1)
        x = x + gate_attn.unsqueeze(1) * attn_out
        
        # MLP block with residual connection and gating
        x_norm2 = self.norm2(x, cond)
        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x


class MIDIDiT(nn.Module):
    """
    Diffusion Transformer for MIDI Generation.
    
    This implements the complete DiT architecture with:
    - Patch embedding
    - Positional encoding
    - Multiple DiT blocks with AdaLN-Zero
    - Final output projection
    """
    
    def __init__(self,
                 patch_dim: int = 256,  # 16x16 patches
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 max_seq_len: int = 2048,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 num_classes: int = 10,  # For class conditioning
                 use_text_conditioning: bool = True):  # 是否使用文本条件
        super().__init__()
        
        self.patch_dim = patch_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_text_conditioning = use_text_conditioning
        
        # Timestep embedding
        self.timestep_embed = TimestepEmbedding(embed_dim)
        
        # Class embedding (optional)
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        self.num_classes = num_classes
        
        # Text encoder (optional)
        if use_text_conditioning:
            self.text_encoder = create_text_encoder(
                vocab_size=len(MUSIC_VOCAB),
                embed_dim=embed_dim,
                num_layers=4,
                num_heads=8,
                max_seq_len=32
            )
        else:
            self.text_encoder = None
        
        # Conditioning dimension (timestep + class + text)
        cond_dim = embed_dim * 3 if use_text_conditioning else embed_dim * 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_dim, embed_dim)
        
        # Positional embedding
        self.pos_embed = PositionalEmbedding(max_seq_len, embed_dim)
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, cond_dim, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.final_norm = AdaLNZero(embed_dim, cond_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, patch_dim)
        
        # Zero initialization for output
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        # Initialize embeddings
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize class embeddings
        nn.init.normal_(self.class_embed.weight, std=0.02)
        
    def forward(self, 
                x: torch.Tensor,
                t: torch.Tensor,
                class_labels: tp.Optional[torch.Tensor] = None,
                text_descriptions: tp.Optional[tp.List[str]] = None,
                mask: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Noisy patches (B, N, patch_dim)
            t: Timesteps (B,)
            class_labels: Class labels (B,), optional
            text_descriptions: Text descriptions (B,), optional
            mask: Sequence mask (B, N), optional
            
        Returns:
            Predicted noise (B, N, patch_dim)
        """
        B, N, _ = x.shape
        
        # Get timestep embeddings
        t_emb = self.timestep_embed(t)  # (B, embed_dim)
        
        # Get class embeddings
        if class_labels is not None:
            c_emb = self.class_embed(class_labels)  # (B, embed_dim)
        else:
            # Use null class token
            c_emb = self.class_embed(torch.zeros(B, dtype=torch.long, device=x.device))
        
        # Get text embeddings
        if self.use_text_conditioning and text_descriptions is not None:
            # 编码文本描述
            encoded = MUSIC_VOCAB.encode_batch(text_descriptions, max_length=32)
            token_ids = encoded['token_ids'].to(x.device)
            attention_mask = encoded['attention_mask'].to(x.device)
            text_emb = self.text_encoder(token_ids, attention_mask)  # (B, embed_dim)
        else:
            # Use zero embedding if no text
            text_emb = torch.zeros(B, self.embed_dim, device=x.device)
            
        # Combine conditioning (timestep + class + text)
        if self.use_text_conditioning:
            cond = torch.cat([t_emb, c_emb, text_emb], dim=-1)  # (B, 3*embed_dim)
        else:
            cond = torch.cat([t_emb, c_emb], dim=-1)  # (B, 2*embed_dim)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, embed_dim)
        
        # Add positional embeddings
        x = self.pos_embed(x)
        
        # Apply DiT blocks
        for block in self.blocks:
            x = block(x, cond, mask)
            
        # Final normalization
        x = self.final_norm(x, cond)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class MIDIDiTWrapper(nn.Module):
    """
    Wrapper for MIDI DiT to handle diffusion training and sampling.
    """
    
    def __init__(self, model: MIDIDiT):
        super().__init__()
        self.model = model
        
    def forward(self, 
                x: torch.Tensor,
                sigma: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """
        Args:
            x: Noisy input (B, N, D)
            sigma: Noise levels (B,)
            
        Returns:
            Model output
        """
        return self.model(x, sigma, **kwargs)
        
    def sample(self,
               shape: tp.Tuple[int, ...],
               num_steps: int = 50,
               sigma_min: float = 0.002,
               sigma_max: float = 80.0,
               rho: float = 7.0,
               device: str = 'cuda',
               **kwargs) -> torch.Tensor:
        """
        Sample from the model using DDPM sampling.
        
        Args:
            shape: Output shape (B, N, D)
            num_steps: Number of denoising steps
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            rho: Noise schedule parameter
            device: Device to run on
            
        Returns:
            Generated samples
        """
        # Create noise schedule
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * 
                   (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        
        # Initialize with noise
        x = torch.randn(shape, device=device, dtype=torch.float32) * sigma_max
        
        # Denoising loop
        for i in range(num_steps):
            sigma = t_steps[i]
            sigma_next = t_steps[i + 1]
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.model(x, sigma.expand(x.shape[0]), **kwargs)
                
            # DDPM update step
            if sigma_next > 0:
                x = x + (sigma_next - sigma) * noise_pred
            else:
                x = x - sigma * noise_pred
                
        return x
        

def create_midi_dit(
    patch_dim: int = 256,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    max_seq_len: int = 2048,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
    num_classes: int = 10,
    use_text_conditioning: bool = True
) -> MIDIDiTWrapper:
    """
    Factory function to create MIDI DiT model.
    
    Args:
        patch_dim: Patch dimension
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
        num_classes: Number of class labels
        use_text_conditioning: Whether to use text conditioning
        
    Returns:
        Wrapped DiT model
    """
    dit = MIDIDiT(
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
    
    return MIDIDiTWrapper(dit)