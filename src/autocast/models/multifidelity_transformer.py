import torch
from torch import nn


class TransformerBlock(nn.Module):
    """A standard Transformer block applying MHA and FFN natively."""

    def __init__(self, n_heads, embedding_dimension, ffn_hidden_dimension, dropout):
        super().__init__()
        # Multihead Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dimension,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Normalization layers
        self.norm_attn = nn.LayerNorm(embedding_dimension)
        self.norm_ffn = nn.LayerNorm(embedding_dimension)
        # Dropouts
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        # FeedForward layer
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dimension, ffn_hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dimension, embedding_dimension),
        )

    def forward(self, x, attn_mask=None):
        attn_x, _ = self.multihead_attn(
            query=x, key=x, value=x, key_padding_mask=attn_mask, need_weights=False
        )
        x = self.norm_attn(x + self.dropout_attn(attn_x))
        ffn_x = self.ffn(x)
        output = self.norm_ffn(x + self.dropout_ffn(ffn_x))
        return output


class AttentionMixer(nn.Module):
    """
    Encapsulates the Transformer sequence-to-sequence mixing logic.

    Expects pre-encoded latent embeddings.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        dropout: float = 0.2,
        n_transformer_blocks: int = 1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_heads=n_heads,
                    embedding_dimension=embedding_dim,
                    ffn_hidden_dimension=embedding_dim * 4,
                    dropout=dropout,
                )
                for _ in range(n_transformer_blocks)
            ]
        )

    def forward(
        self, encoded_sequence: torch.Tensor, levels_mask: torch.Tensor | None = None
    ):
        # encoded_sequence: (Batch, Dataset/Levels, embedding_dim)
        # levels_mask: (Batch, Dataset/Levels) - True if level is missing

        # Process with the transformer blocks each embedding.
        for transformer_block in self.transformer_blocks:
            encoded_sequence = transformer_block(
                x=encoded_sequence, attn_mask=levels_mask
            )  # (Batch, Dataset/Levels, embedding_dim)

        # Return full sequence without pooling
        # Since we no longer pool the latents together, we DO NOT perform
        # a masked_fill with 0.0. The inputs at the masked positions use
        # their transformer queries to draw contextual information from the
        # unmasked levels, meaning their output states are rich inferences
        # that the Decoder needs to successfully predict the targets!
        return encoded_sequence
