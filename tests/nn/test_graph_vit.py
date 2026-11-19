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


def test_checkpointed_gather_saves_less_memory_and_preserves_dropout():
    reference = GatheredGraphAttention(
        MultiheadSelfAttention(16, attention_heads=2, dropout=0.2),
        checkpoint_gather=False,
    )
    checkpointed = GatheredGraphAttention(
        MultiheadSelfAttention(16, attention_heads=2, dropout=0.2)
    )
    checkpointed.load_state_dict(reference.state_dict(), strict=True)
    tokens = torch.randn(2, 16, 16, requires_grad=True)
    pos = torch.cartesian_prod(torch.arange(4.0), torch.arange(4.0))
    neighbors = periodic_grid_neighbors(4, 4, 2, device=tokens.device)

    def run(attention):
        storages = {}

        def pack(tensor):
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
            return tensor

        with (
            torch.random.fork_rng(devices=[]),
            torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor),
        ):
            torch.manual_seed(123)
            output = attention(tokens, pos, neighbors)
        gradients = torch.autograd.grad(
            output.square().mean(), (tokens, *attention.parameters())
        )
        return output, gradients, sum(storages.values())

    expected, expected_gradients, reference_bytes = run(reference)
    actual, actual_gradients, checkpointed_bytes = run(checkpointed)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradients, expected_gradients)
    assert checkpointed_bytes < reference_bytes
