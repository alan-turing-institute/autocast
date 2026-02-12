"""Tests for autocast.scripts.utils module."""

import pytest

from autocast.scripts.utils import RunCollator


@pytest.fixture
def sample_config():
    """Sample config dictionary mimicking resolved_config.yaml structure."""
    return {
        "datamodule": {
            "batch_size": 64,
            "use_normalization": False,
            "n_steps_input": 1,
        },
        "model": {
            "encoder": {
                "_target_": "autocast.encoders.permute_concat.PermuteConcat",
                "in_channels": 7,
            },
            "decoder": {
                "_target_": "autocast.decoders.channels_last.ChannelsLast",
                "output_channels": 4,
            },
            "processor": {
                "_target_": "autocast.processors.vit.AViTProcessor",
                "hidden_dim": 256,
                "n_layers": 4,
                "num_heads": 4,
            },
            "loss_func": {
                "_target_": "autocast.losses.ensemble.CRPSLoss",
            },
            "n_members": 10,
            "train_in_latent_space": False,
        },
        "optimizer": {
            "learning_rate": 0.0002,
            "optimizer": "adamw",
            "scheduler": "cosine",
        },
    }


@pytest.fixture
def alternate_processor_config():
    """Config with different processor that uses hidden_channels instead."""
    return {
        "model": {
            "processor": {
                "_target_": "autocast.processors.fno.FNOProcessor",
                "hidden_channels": 128,  # Different key name
                "n_layers": 8,
            },
        },
    }


@pytest.fixture
def collator(tmp_path):
    """Create a RunCollator instance for testing."""
    # Create fake outputs directory
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    return RunCollator(outputs_dir=str(outputs_dir))


class TestFindMatchingPaths:
    """Test the _find_matching_paths method."""

    def test_exact_path_single_level(self, collator, sample_config):
        """Test exact path matching at single level."""
        matches = collator._find_matching_paths(sample_config, "datamodule")
        assert len(matches) == 1
        assert matches[0][0] == "datamodule"
        assert isinstance(matches[0][1], dict)

    def test_exact_path_nested(self, collator, sample_config):
        """Test exact path matching through nested structure."""
        matches = collator._find_matching_paths(sample_config, "model.encoder._target_")
        assert len(matches) == 1
        assert matches[0][0] == "model.encoder._target_"
        assert matches[0][1] == "autocast.encoders.permute_concat.PermuteConcat"

    def test_exact_path_no_match(self, collator, sample_config):
        """Test exact path that doesn't exist."""
        matches = collator._find_matching_paths(sample_config, "model.nonexistent")
        assert len(matches) == 0

    def test_wildcard_single_key(self, collator, sample_config):
        """Test wildcard matching a single key level."""
        matches = collator._find_matching_paths(sample_config, "model.*.hidden_dim")
        assert len(matches) == 1
        assert matches[0][0] == "model.processor.hidden_dim"
        assert matches[0][1] == 256

    def test_wildcard_end_pattern(self, collator, sample_config):
        """Test wildcard at end of pattern."""
        matches = collator._find_matching_paths(
            sample_config, "model.processor.hidden_*"
        )
        assert len(matches) == 1
        assert matches[0][0] == "model.processor.hidden_dim"
        assert matches[0][1] == 256

    def test_wildcard_multiple_matches(self, collator, sample_config):
        """Test wildcard matching multiple keys at same level."""
        matches = collator._find_matching_paths(sample_config, "model.*._target_")
        assert len(matches) == 4  # encoder, decoder, processor, loss_func
        paths = [m[0] for m in matches]
        assert "model.encoder._target_" in paths
        assert "model.decoder._target_" in paths
        assert "model.processor._target_" in paths
        assert "model.loss_func._target_" in paths

    def test_wildcard_star_matches_everything(self, collator, sample_config):
        """Test that * matches any characters."""
        matches = collator._find_matching_paths(sample_config, "model.*")
        assert (
            len(matches) == 6
        )  # encoder, decoder, processor, loss_func, n_members, train_in_latent_space

    def test_question_mark_wildcard(self, collator):
        """Test ? wildcard matching single character."""
        config = {
            "model": {
                "process1": {"value": 1},
                "process2": {"value": 2},
                "processor": {"value": 3},
            }
        }
        matches = collator._find_matching_paths(config, "model.process?")
        assert len(matches) == 2  # Matches process1 and process2, not processor

    def test_multiple_wildcards_nested(self, collator, sample_config):
        """Test multiple wildcards in pattern."""
        matches = collator._find_matching_paths(sample_config, "model.*.*")
        assert len(matches) > 3  # Multiple nested values under model components

    def test_empty_config(self, collator):
        """Test with empty config."""
        matches = collator._find_matching_paths({}, "model.encoder._target_")
        assert len(matches) == 0

    def test_non_dict_value_in_path(self, collator):
        """Test traversal stops at non-dict values."""
        config = {"model": {"encoder": "not_a_dict"}}
        matches = collator._find_matching_paths(config, "model.encoder.nested")
        assert len(matches) == 0


class TestGetNestedValue:
    """Test the _get_nested_value method with pattern support."""

    def test_exact_path_found(self, collator, sample_config):
        """Test retrieving value with exact path."""
        value = collator._get_nested_value(sample_config, "model.processor.hidden_dim")
        assert value == 256

    def test_exact_path_not_found(self, collator, sample_config):
        """Test default value when path not found."""
        value = collator._get_nested_value(
            sample_config, "model.nonexistent", default="DEFAULT"
        )
        assert value == "DEFAULT"

    def test_wildcard_pattern_found(self, collator, sample_config):
        """Test retrieving value with wildcard pattern."""
        value = collator._get_nested_value(sample_config, "model.*.hidden_dim")
        assert value == 256  # First match

    def test_wildcard_pattern_not_found(self, collator, sample_config):
        """Test default value when pattern matches nothing."""
        value = collator._get_nested_value(
            sample_config, "model.*.nonexistent", default="N/A"
        )
        assert value == "N/A"

    def test_end_wildcard_pattern(self, collator, sample_config):
        """Test pattern with wildcard at end."""
        value = collator._get_nested_value(sample_config, "model.processor.hidden_*")
        assert value == 256

    def test_end_wildcard_alternate_config(self, collator, alternate_processor_config):
        """Test end wildcard matches different key name."""
        value = collator._get_nested_value(
            alternate_processor_config, "model.processor.hidden_*"
        )
        assert value == 128  # Matches hidden_channels in alternate config

    def test_multiple_matches_returns_first(self, collator, sample_config):
        """Test that first match is returned when multiple matches exist."""
        value = collator._get_nested_value(sample_config, "model.*._target_")
        # Should return first match (order depends on dict iteration)
        # The value should be one of the _target_ values
        assert "autocast" in value
        assert any(
            comp in value for comp in ["encoders", "decoders", "processors", "losses"]
        )

    def test_deep_nested_exact_path(self, collator, sample_config):
        """Test deeply nested exact path."""
        value = collator._get_nested_value(sample_config, "model.encoder.in_channels")
        assert value == 7

    def test_default_value_used(self, collator, sample_config):
        """Test default value parameter."""
        value = collator._get_nested_value(
            sample_config, "missing.path", default="CUSTOM_DEFAULT"
        )
        assert value == "CUSTOM_DEFAULT"


class TestExtractConfigParams:
    """Test the _extract_config_params method."""

    def test_extract_with_exact_paths(self, collator, sample_config):
        """Test extraction with exact paths."""
        collator.config_params = {
            "batch_size": "datamodule.batch_size",
            "learning_rate": "optimizer.learning_rate",
        }
        result = collator._extract_config_params(sample_config)
        assert result["batch_size"] == 64
        assert result["learning_rate"] == 0.0002

    def test_extract_with_wildcard_patterns(self, collator, sample_config):
        """Test extraction with wildcard patterns."""
        collator.config_params = {
            "hidden_size": "model.processor.hidden_*",
            "encoder": "model.encoder._target_",
        }
        result = collator._extract_config_params(sample_config)
        assert result["hidden_size"] == 256
        assert "PermuteConcat" in result["encoder"]  # Simplified

    def test_extract_target_simplification(self, collator, sample_config):
        """Test that _target_ values are simplified."""
        collator.config_params = {
            "processor": "model.processor._target_",
        }
        result = collator._extract_config_params(sample_config)
        # Should be simplified from full path to just class name
        assert result["processor"] == "AViTProcessor"

    def test_extract_missing_params_use_default(self, collator, sample_config):
        """Test that missing params get N/A default."""
        collator.config_params = {
            "missing_field": "model.nonexistent.field",
        }
        result = collator._extract_config_params(sample_config)
        assert result["missing_field"] == "N/A"

    def test_extract_with_alternate_config(self, collator, alternate_processor_config):
        """Test flexible pattern works with alternate config structure."""
        collator.config_params = {
            "hidden_size": "model.processor.hidden_*",  # Flexible pattern
        }
        result = collator._extract_config_params(alternate_processor_config)
        # Should match hidden_channels in alternate config
        assert result["hidden_size"] == 128


class TestDefaultConfigParams:
    """Test the default config parameters."""

    def test_default_params_include_pattern(self, collator):
        """Test that default params include flexible pattern for hidden_size."""
        assert "hidden_size" in collator.config_params
        assert "*" in collator.config_params["hidden_size"]
        assert collator.config_params["hidden_size"] == "model.processor.hidden_*"

    def test_default_params_work_with_sample_config(self, collator, sample_config):
        """Test that default params successfully extract from sample config."""
        result = collator._extract_config_params(sample_config)

        # Check key extractions
        assert result["batch_size"] == 64
        assert result["n_members"] == 10
        assert result["hidden_size"] == 256  # Via pattern matching
        assert result["encoder"] == "PermuteConcat"
        assert result["processor"] == "AViTProcessor"
        assert result["decoder"] == "ChannelsLast"
        assert result["learning_rate"] == 0.0002
        assert result["use_normalization"] is False
        assert result["train_in_latent"] is False


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_pattern_with_literal_asterisk(self, collator):
        """Test that patterns don't match literal asterisks in keys (edge case)."""
        config = {"model": {"key*with*asterisk": "value"}}
        # This is unlikely in practice but testing the behavior
        matches = collator._find_matching_paths(config, "model.*")
        assert len(matches) == 1

    def test_nested_lists_not_traversed(self, collator):
        """Test that lists in config are not traversed."""
        config = {
            "model": {
                "layers": [{"hidden": 128}, {"hidden": 256}],
            }
        }
        # Should not traverse into list
        matches = collator._find_matching_paths(config, "model.layers.hidden")
        assert len(matches) == 0

    def test_none_values_handled(self, collator):
        """Test that None values are handled gracefully."""
        config = {"model": {"processor": None}}
        value = collator._get_nested_value(config, "model.processor.hidden_dim")
        assert value == "N/A"  # Default when can't traverse further

    def test_empty_pattern_parts(self, collator, sample_config):
        """Test behavior with unusual patterns."""
        # Pattern with trailing dot (creates empty part)
        matches = collator._find_matching_paths(sample_config, "model.")
        # Should handle gracefully (implementation dependent)
        assert isinstance(matches, list)
