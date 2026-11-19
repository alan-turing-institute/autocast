import pytest
import torch
import torch.nn.functional as F
from azula.nn.attention import MultiheadSelfAttention

from autocast.nn.graph_vit import GatheredGraphAttention, periodic_grid_neighbors


@pytest.mark.parametrize(
    ("height", "width", "hops", "degree"),
    [(16, 16, 2, 13), (4, 3, 1, 5), (2, 2, 2, 4), (1, 4, 5, 4), (3, 5, 0, 1)],
)
def test_periodic_grid_neighbors(height, width, hops, degree):
    neighbors = periodic_grid_neighbors(height, width, hops, device=torch.device("cpu"))
    assert neighbors.shape == (height * width, degree)

    nodes = torch.arange(height * width)
    rows, cols = nodes // width, nodes % width
    dy = (rows[:, None] - rows).abs()
    dx = (cols[:, None] - cols).abs()
    distance = torch.minimum(dy, height - dy) + torch.minimum(dx, width - dx)
    expected = torch.where(distance <= hops)[1].reshape(height * width, degree)
    torch.testing.assert_close(neighbors.sort(dim=1).values, expected)


def test_gathered_attention_passes_only_neighbor_pairs_to_sdpa(monkeypatch):
    attention = GatheredGraphAttention(
        MultiheadSelfAttention(16, attention_heads=2, pos_features=2, rpb=True)
    )
    tokens = torch.randn(2, 12, 16)
    pos = torch.cartesian_prod(torch.arange(4.0), torch.arange(3.0))
    neighbors = periodic_grid_neighbors(4, 3, 1, device=tokens.device)
    original = F.scaled_dot_product_attention
    calls = []

    def record_attention(**kwargs):
        calls.append(
            {
                name: tensor.shape
                for name, tensor in kwargs.items()
                if name != "dropout_p"
            }
        )
        return original(**kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", record_attention)
    assert attention(tokens, pos, neighbors).shape == tokens.shape
    assert calls == [
        {
            "query": (2, 24, 1, 8),
            "key": (2, 24, 5, 8),
            "value": (2, 24, 5, 8),
            "attn_mask": (1, 24, 1, 5),
        },
    ]
