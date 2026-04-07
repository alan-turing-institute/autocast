import torch
from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self, n_heads, embedding_dimension, ffn_hidden_dimension, dropout):
        super(TransformerBlock, self).__init__()
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
    Encapsulates the Transformer mixing and pooling logic (BLOCK 2 & 3) of the MultifidelityTransformer.
    Expects pre-encoded latent embeddings.
    """
    def __init__(self, embedding_dim: int, n_heads: int, dropout: float = 0.2, n_transformer_blocks: int = 1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_heads=n_heads,
                    embedding_dimension=embedding_dim,
                    ffn_hidden_dimension=embedding_dim * 4,
                    dropout=dropout
                ) for _ in range(n_transformer_blocks)
            ]
        )

    def forward(self, encoded_sequence: torch.Tensor, levels_mask: torch.Tensor):
        # encoded_sequence: (Batch, Dataset/Levels, embedding_dim)
        # levels_mask: (Batch, Dataset/Levels) - True if level is missing

        # Process with the transformer blocks each embedding.
        for transformer_block in self.transformer_blocks:
            encoded_sequence = transformer_block(
                x=encoded_sequence, attn_mask=levels_mask
            )  # (Batch, Dataset/Levels, embedding_dim)

        # Pooling:
        # Set to 0 all the embeddings of the levels that we don't have
        transformer_output = encoded_sequence.masked_fill(
            mask=levels_mask.unsqueeze(-1), value=0.0
        )
        
        # Average the embeddings for the levels that we have
        # Clamp valid_counts to at least 1 to avoid division by zero if everything is masked
        valid_counts = (levels_mask == False).sum(dim=-1).unsqueeze(-1)
        valid_counts = torch.clamp(valid_counts, min=1)
        final_embedding = transformer_output.sum(dim=-2) / valid_counts

        return final_embedding
