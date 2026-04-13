import pytest
import torch

from autocast.data.multidataset import MultiSpatioTemporalDataset
from autocast.types.batch import ListSample, Sample


@pytest.fixture
def dummy_spatiotemporal_data_1():
    """Create dummy spatiotemporal data for dataset with 2 channels."""
    # 2 trajectories, 10 timesteps, 4x4 spatial grid, 2 channels
    data = torch.randn(2, 10, 4, 4, 2)
    return {"data": data, "constant_scalars": None, "constant_fields": None}


@pytest.fixture
def dummy_spatiotemporal_data_2():
    """Create dummy spatiotemporal data for dataset with 3 channels."""
    # 2 trajectories, 10 timesteps, 4x4 spatial grid, 3 channels
    data = torch.randn(2, 10, 4, 4, 3)
    return {"data": data, "constant_scalars": None, "constant_fields": None}


@pytest.fixture
def dummy_spatiotemporal_data_3():
    """Create dummy spatiotemporal data for dataset with 1 channel."""
    # 2 trajectories, 10 timesteps, 4x4 spatial grid, 1 channel
    data = torch.randn(2, 10, 4, 4, 1)
    return {"data": data, "constant_scalars": None, "constant_fields": None}


@pytest.fixture
def normalization_stats_1():
    """Normalization stats for dataset 1 (2 channels: U, V)."""
    return {
        "stats": {
            "mean": {"U": 1.0, "V": 2.0},
            "std": {"U": 0.5, "V": 1.0},
            "mean_delta": {"U": 0.0, "V": 0.0},
            "std_delta": {"U": 0.1, "V": 0.2},
        },
        "core_field_names": ["U", "V"],
        "constant_field_names": [],
    }


@pytest.fixture
def normalization_stats_2():
    """Normalization stats for dataset 2 (3 channels: A, B, C)."""
    return {
        "stats": {
            "mean": {"A": 3.0, "B": 4.0, "C": 5.0},
            "std": {"A": 1.5, "B": 2.0, "C": 2.5},
            "mean_delta": {"A": 0.0, "B": 0.0, "C": 0.0},
            "std_delta": {"A": 0.3, "B": 0.4, "C": 0.5},
        },
        "core_field_names": ["A", "B", "C"],
        "constant_field_names": [],
    }


# ============================================================================
# Test create_mask static method
# ============================================================================


def test_create_mask_sequential_2_levels():
    """Test sequential mask creation with 2 levels."""
    mask = MultiSpatioTemporalDataset.create_mask(n_levels=2, mode="sequential")

    assert mask.shape == (2, 2)

    # Expected pattern:
    # [[False, True],
    #  [False, False]]
    # Meaning: when using mask column 0, only dataset 0 is available (dataset 1 is masked)
    #          when using mask column 1, both datasets are available
    expected = torch.tensor([[False, True], [False, False]], dtype=torch.bool)
    assert torch.equal(mask, expected)


def test_create_mask_sequential_3_levels():
    """Test sequential mask creation with 3 levels."""
    mask = MultiSpatioTemporalDataset.create_mask(n_levels=3, mode="sequential")

    assert mask.shape == (3, 3)

    # Expected pattern:
    # [[False, True,  True],
    #  [False, False, True],
    #  [False, False, False]]
    expected = torch.tensor(
        [[False, True, True], [False, False, True], [False, False, False]],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)


def test_create_mask_combinatorial_2_levels():
    """Test combinatorial mask creation with 2 levels."""
    mask = MultiSpatioTemporalDataset.create_mask(n_levels=2, mode="combinatorial")

    # Should have 3 combinations (2^2 - 1, excluding all masked)
    assert mask.shape == (2, 3)

    # Expected patterns (order may vary, but should not include [True, True]):
    # [[False, True,  False],
    #  [False, False, True]]
    # This represents: (0,0), (1,0), (0,1) — all combos except (1,1)

    # Check no column is all True
    assert not torch.all(mask, dim=0).any()

    # Check all unique combinations
    assert mask.T.unique(dim=0).shape[0] == 3


def test_create_mask_combinatorial_3_levels():
    """Test combinatorial mask creation with 3 levels."""
    mask = MultiSpatioTemporalDataset.create_mask(n_levels=3, mode="combinatorial")

    # Should have 7 combinations (2^3 - 1)
    assert mask.shape == (3, 7)

    # Check no column is all True
    assert not torch.all(mask, dim=0).any()

    # Check all unique combinations
    assert mask.T.unique(dim=0).shape[0] == 7


def test_create_mask_invalid_n_levels():
    """Test that invalid n_levels raises error."""
    with pytest.raises(ValueError, match="n_levels must be > 0"):
        MultiSpatioTemporalDataset.create_mask(n_levels=0, mode="sequential")

    with pytest.raises(ValueError, match="n_levels must be > 0"):
        MultiSpatioTemporalDataset.create_mask(n_levels=-1, mode="sequential")


def test_create_mask_invalid_mode():
    """Test that invalid mode raises error."""
    with pytest.raises(ValueError, match="Unknown mask mode"):
        MultiSpatioTemporalDataset.create_mask(n_levels=2, mode="invalid")


# ============================================================================
# Test initialization
# ============================================================================


def test_init_custom_mask(dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2):
    """Test initialization with custom mask tensor."""
    custom_mask = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    data_list = [dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2]

    dataset = MultiSpatioTemporalDataset(
        data_paths=None,
        data=data_list,
        masks=custom_mask,
        n_steps_input=2,
        n_steps_output=1,
    )

    assert torch.equal(dataset.masks, custom_mask)


def test_init_no_masks(dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2):
    """Test initialization with no masks."""
    data_list = [dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2]

    dataset = MultiSpatioTemporalDataset(
        data_paths=None,
        data=data_list,
        masks=None,
        n_steps_input=2,
        n_steps_output=1,
    )

    assert dataset.masks is None


def test_init_mask_dimension_mismatch(dummy_spatiotemporal_data_1):
    """Test that mask dimensions must match number of datasets."""
    wrong_mask = torch.tensor(
        [[True, False], [False, True], [True, True]], dtype=torch.bool
    )

    with pytest.raises(ValueError, match="Mask first dimension .* must match"):
        MultiSpatioTemporalDataset(
            data_paths=None,
            data=[dummy_spatiotemporal_data_1],  # Only 1 dataset
            masks=wrong_mask,  # But mask has 3 datasets
            n_steps_input=2,
            n_steps_output=1,
        )


def test_init_mismatched_dataset_lengths():
    """Test that datasets with different lengths raise an error."""
    # Create two datasets with different lengths
    data_1 = {
        "data": torch.randn(2, 10, 4, 4, 2),  # 2 trajectories
        "constant_scalars": None,
        "constant_fields": None,
    }
    data_2 = {
        "data": torch.randn(3, 10, 4, 4, 2),  # 3 trajectories (different!)
        "constant_scalars": None,
        "constant_fields": None,
    }

    with pytest.raises(ValueError, match="All datasets must have the same length"):
        MultiSpatioTemporalDataset(
            data_paths=None,
            data=[data_1, data_2],
            masks="sequential",
            n_steps_input=2,
            n_steps_output=1,
        )


# ============================================================================
# Test __len__
# ============================================================================


def test_len(dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2):
    """Test __len__ method."""
    data_list = [dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2]

    dataset = MultiSpatioTemporalDataset(
        data_paths=None,
        data=data_list,
        masks="sequential",
        n_steps_input=2,
        n_steps_output=1,
    )

    # Each dataset has 2 trajectories, 10 timesteps
    # With n_steps_input=2, n_steps_output=1, stride=1
    # Number of samples = 2 * (10 - 2 - 1 + 1) = 2 * 8 = 16
    expected_length = len(dataset.datasets[0])
    assert len(dataset) == expected_length


def test_len_empty_dataset():
    """Test __len__ on empty dataset."""
    dataset = MultiSpatioTemporalDataset.__new__(MultiSpatioTemporalDataset)
    dataset.datasets = []
    assert len(dataset) == 0


# ============================================================================
# Test __getitem__
# ============================================================================


def test_getitem_returns_list_sample(
    dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2
):
    """Test that __getitem__ returns ListSample."""
    data_list = [dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2]

    dataset = MultiSpatioTemporalDataset(
        data_paths=None,
        data=data_list,
        masks="sequential",
        n_steps_input=2,
        n_steps_output=1,
    )

    sample = dataset[0]

    assert isinstance(sample, ListSample)
    assert isinstance(sample.inner, list)
    assert len(sample.inner) == 2
    assert all(isinstance(s, Sample) for s in sample.inner)


def test_getitem_mask_shape(dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2):
    """Test that mask in ListSample has correct shape."""
    data_list = [dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2]

    dataset = MultiSpatioTemporalDataset(
        data_paths=None,
        data=data_list,
        masks="sequential",
        n_steps_input=2,
        n_steps_output=1,
    )

    sample = dataset[0]

    assert sample.mask is not None
    assert sample.mask.shape == (2, 2)  # (n_datasets, n_mask_combinations)


def test_getitem_sample_shapes(
    dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2
):
    """Test that samples have correct shapes."""
    data_list = [dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2]

    dataset = MultiSpatioTemporalDataset(
        data_paths=None,
        data=data_list,
        masks="sequential",
        n_steps_input=2,
        n_steps_output=1,
    )

    sample = dataset[0]

    # First dataset has 2 channels
    assert sample.inner[0].input_fields.shape == (2, 4, 4, 2)  # (T_in, W, H, C)
    assert sample.inner[0].output_fields.shape == (1, 4, 4, 2)  # (T_out, W, H, C)

    # Second dataset has 3 channels
    assert sample.inner[1].input_fields.shape == (2, 4, 4, 3)
    assert sample.inner[1].output_fields.shape == (1, 4, 4, 3)


# ============================================================================
# Integration tests
# ============================================================================


def test_integration_sequential_mask_3_datasets(
    dummy_spatiotemporal_data_1,
    dummy_spatiotemporal_data_2,
    dummy_spatiotemporal_data_3,
):
    """Integration test with 3 datasets and sequential masking."""
    data_list = [
        dummy_spatiotemporal_data_1,
        dummy_spatiotemporal_data_2,
        dummy_spatiotemporal_data_3,
    ]

    dataset = MultiSpatioTemporalDataset(
        data_paths=None,
        data=data_list,
        masks="sequential",
        n_steps_input=3,
        n_steps_output=2,
        stride=1,
    )

    # Check datasets created
    assert len(dataset.datasets) == 3

    # Check mask is sequential
    expected_mask = torch.tensor(
        [[False, True, True], [False, False, True], [False, False, False]],
        dtype=torch.bool,
    )
    assert torch.equal(dataset.masks, expected_mask)

    # Check __len__
    assert len(dataset) > 0

    # Check __getitem__
    sample = dataset[0]
    assert isinstance(sample, ListSample)
    assert len(sample.inner) == 3

    # Verify each sample has correct input/output shapes
    assert sample.inner[0].input_fields.shape[0] == 3  # n_steps_input
    assert sample.inner[0].output_fields.shape[0] == 2  # n_steps_output

    # Check mask is attached
    assert torch.equal(sample.mask, expected_mask)


def test_integration_combinatorial_mask_2_datasets(
    dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2
):
    """Integration test with 2 datasets and combinatorial masking."""
    data_list = [dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2]

    dataset = MultiSpatioTemporalDataset(
        data_paths=None,
        data=data_list,
        masks="combinatorial",
        n_steps_input=2,
        n_steps_output=1,
    )

    # Check mask has 3 columns (all combinations except all-masked)
    assert dataset.masks.shape == (2, 3)

    # Verify no all-masked combination
    assert not torch.all(dataset.masks, dim=0).any()

    # Sample and check
    sample = dataset[0]
    assert len(sample.inner) == 2
    assert sample.mask.shape == (2, 3)

    # Verify each mask pattern represents a valid combination:
    # Should have: [False, False], [True, False], [False, True]
    mask_patterns = sample.mask.T.tolist()
    assert [False, False] in mask_patterns
    assert [True, False] in mask_patterns
    assert [False, True] in mask_patterns


def test_integration_iterate_through_dataset(
    dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2
):
    """Integration test iterating through entire dataset."""
    data_list = [dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2]

    dataset = MultiSpatioTemporalDataset(
        data_paths=None,
        data=data_list,
        masks="sequential",
        n_steps_input=2,
        n_steps_output=1,
    )

    # Iterate through all samples
    for i in range(len(dataset)):
        sample = dataset[i]

        # Each sample should be valid
        assert isinstance(sample, ListSample)
        assert len(sample.inner) == 2
        assert sample.mask is not None

        # Shapes should be consistent
        assert sample.inner[0].input_fields.shape[0] == 2
        assert sample.inner[0].output_fields.shape[0] == 1


def test_integration_with_different_channels(
    dummy_spatiotemporal_data_1,
    dummy_spatiotemporal_data_2,
    dummy_spatiotemporal_data_3,
):
    """Integration test with datasets having different channel counts."""
    # Dataset 1: 2 channels, Dataset 2: 3 channels, Dataset 3: 1 channel
    data_list = [
        dummy_spatiotemporal_data_1,
        dummy_spatiotemporal_data_2,
        dummy_spatiotemporal_data_3,
    ]

    dataset = MultiSpatioTemporalDataset(
        data_paths=None,
        data=data_list,
        masks="combinatorial",
        n_steps_input=2,
        n_steps_output=1,
    )

    sample = dataset[0]

    # Verify each dataset maintains its own channel count
    assert sample.inner[0].input_fields.shape[-1] == 2  # 2 channels
    assert sample.inner[1].input_fields.shape[-1] == 3  # 3 channels
    assert sample.inner[2].input_fields.shape[-1] == 1  # 1 channel


def test_integration_with_stride(
    dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2
):
    """Integration test with different stride values."""
    data_list = [dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2]

    dataset_stride_1 = MultiSpatioTemporalDataset(
        data_paths=None,
        data=data_list,
        masks="sequential",
        n_steps_input=2,
        n_steps_output=1,
        stride=1,
    )

    dataset_stride_2 = MultiSpatioTemporalDataset(
        data_paths=None,
        data=data_list,
        masks="sequential",
        n_steps_input=2,
        n_steps_output=1,
        stride=2,
    )

    # Stride 2 should produce fewer samples than stride 1
    assert len(dataset_stride_2) < len(dataset_stride_1)


# ============================================================================
# Test per-dataset normalization
# ============================================================================


def test_normalization_per_dataset_stats_list(
    dummy_spatiotemporal_data_1,
    dummy_spatiotemporal_data_2,
    normalization_stats_1,
    normalization_stats_2,
):
    """Test that per-dataset normalization stats are applied correctly."""
    data_list = [dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2]

    dataset = MultiSpatioTemporalDataset(
        data_paths=None,
        data=data_list,
        masks="sequential",
        n_steps_input=2,
        n_steps_output=1,
        use_normalization=True,
        normalization_stats=[normalization_stats_1, normalization_stats_2],
    )

    # Both datasets should have normalization set up
    assert dataset.datasets[0].norm is not None
    assert dataset.datasets[1].norm is not None

    # Each should use its own stats
    assert dataset.datasets[0].norm.core_field_names == ["U", "V"]
    assert dataset.datasets[1].norm.core_field_names == ["A", "B", "C"]


def test_normalization_stats_list_length_mismatch(
    dummy_spatiotemporal_data_1,
    dummy_spatiotemporal_data_2,
    normalization_stats_1,
):
    """Test that mismatched normalization stats list length raises error."""
    data_list = [dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2]

    # Only 1 stats dict but 2 datasets
    with pytest.raises(ValueError, match="Length of normalization_stats list"):
        MultiSpatioTemporalDataset(
            data_paths=None,
            data=data_list,
            masks="sequential",
            n_steps_input=2,
            n_steps_output=1,
            use_normalization=True,
            normalization_stats=[normalization_stats_1],  # Only 1, need 2
        )


def test_normalization_no_stats_when_disabled(
    dummy_spatiotemporal_data_1,
    dummy_spatiotemporal_data_2,
    normalization_stats_1,
    normalization_stats_2,
):
    """Test that normalization stats are ignored when use_normalization=False."""
    data_list = [dummy_spatiotemporal_data_1, dummy_spatiotemporal_data_2]

    dataset = MultiSpatioTemporalDataset(
        data_paths=None,
        data=data_list,
        masks="sequential",
        n_steps_input=2,
        n_steps_output=1,
        use_normalization=False,
        normalization_stats=[normalization_stats_1, normalization_stats_2],
    )

    # No normalization should be set up
    assert dataset.datasets[0].norm is None
    assert dataset.datasets[1].norm is None


def test_integration_per_dataset_normalization_3_datasets(
    dummy_spatiotemporal_data_1,
    dummy_spatiotemporal_data_2,
    dummy_spatiotemporal_data_3,
    normalization_stats_1,
    normalization_stats_2,
):
    """Integration test with 3 datasets and per-dataset normalization."""
    # Create third normalization stats
    normalization_stats_3 = {
        "stats": {
            "mean": {"X": 0.0},
            "std": {"X": 1.0},
            "mean_delta": {"X": 0.0},
            "std_delta": {"X": 0.1},
        },
        "core_field_names": ["X"],
        "constant_field_names": [],
    }

    data_list = [
        dummy_spatiotemporal_data_1,
        dummy_spatiotemporal_data_2,
        dummy_spatiotemporal_data_3,
    ]

    dataset = MultiSpatioTemporalDataset(
        data_paths=None,
        data=data_list,
        masks="combinatorial",
        n_steps_input=2,
        n_steps_output=1,
        use_normalization=True,
        normalization_stats=[
            normalization_stats_1,
            normalization_stats_2,
            normalization_stats_3,
        ],
    )

    # All datasets should have normalization
    assert len(dataset.datasets) == 3
    assert all(ds.norm is not None for ds in dataset.datasets)

    # Each should have correct field names
    assert dataset.datasets[0].norm.core_field_names == ["U", "V"]
    assert dataset.datasets[1].norm.core_field_names == ["A", "B", "C"]
    assert dataset.datasets[2].norm.core_field_names == ["X"]
