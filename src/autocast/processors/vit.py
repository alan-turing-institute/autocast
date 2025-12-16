from typing import Any
import torch
from torch import nn
from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor


class PatchEmbedding(nn.Module):
    """Converts image into patches and embeds them."""
    
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        img_size: tuple[int, int],
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        
        # Linear projection of flattened patches
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
    def forward(self, x: Tensor) -> Tensor:
        # x shape: (B, C, H, W)
        x = self.projection(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block with multi-head attention."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class ViTProcessor(Processor[EncodedBatch]):
    """Vision Transformer Processor for spatiotemporal prediction.
    
    A discrete processor that uses a Vision Transformer (ViT) to learn
    mappings between function spaces for spatiotemporal prediction.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    img_size : tuple[int, int]
        Spatial dimensions (height, width) of input.
    patch_size : int, optional
        Size of image patches. Default is 16.
    embed_dim : int, optional
        Embedding dimension. Default is 768.
    num_heads : int, optional
        Number of attention heads. Default is 12.
    depth : int, optional
        Number of transformer blocks. Default is 12.
    mlp_ratio : float, optional
        Ratio of mlp hidden dim to embedding dim. Default is 4.0.
    dropout : float, optional
        Dropout rate. Default is 0.0.
    loss_func : nn.Module, optional
        Loss function. Defaults to MSELoss.
    learning_rate : float, optional
        Learning rate for optimizer. Default is 1e-3.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: tuple[int, int],
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        loss_func: nn.Module | None = None,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Decoder head to reconstruct spatial output
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, patch_size * patch_size * out_channels),
        )
        
        self.loss_func = loss_func or nn.MSELoss()
        self.learning_rate = learning_rate
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through ViT.
        
        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, C, H, W)
            
        Returns
        -------
        Tensor
            Output tensor of shape (B, out_channels, H, W)
        """
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # Decode patches back to spatial dimensions
        x = self.decoder(x)  # (B, n_patches, patch_size^2 * out_channels)
        
        # Reshape to spatial output
        n_patches_h = H // self.patch_size
        n_patches_w = W // self.patch_size
        
        x = x.reshape(
            B,
            n_patches_h,
            n_patches_w,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
        
        # Rearrange to (B, out_channels, H, W)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, out_channels, n_h, p, n_w, p)
        x = x.reshape(B, self.out_channels, H, W)
        
        return x
    
    def map(self, x: Tensor) -> Tensor:
        """Map input to output (alias for forward)."""
        return self(x)
    
    def loss(self, batch: EncodedBatch) -> Tensor:
        """Compute loss for a batch."""
        output = self.map(batch.encoded_inputs)
        return self.loss_func(output, batch.encoded_output_fields)
