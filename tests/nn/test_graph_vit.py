import pytest
import torch

from autocast.nn.graph_vit import periodic_grid_neighbors


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
