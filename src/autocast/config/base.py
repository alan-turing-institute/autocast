from typing import Any, Literal

from autoemulate.simulations.base import Simulator
from pydantic import BaseModel, ConfigDict, Field

AutoInt = int | Literal["auto"]


class DataParams(BaseModel):
    """Parameters used to configure the data module/dataset."""

    n_channels: AutoInt = "auto"
    simulator: Simulator | None = None
    split: float = 0.8
    # Add other data params as needed, allowing extra fields for now
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class ModelParams(BaseModel):
    """Parameters for the model architecture."""

    # We allow extra fields because the model config can be complex (encoder, decoder,
    # processor, etc.) and might vary significantly. Ideally these would be sub-models.
    model_config = ConfigDict(extra="allow")

    learning_rate: float = 1e-3
    train_in_latent_space: bool = True
    teacher_forcing_ratio: float = 0.5
    max_rollout_steps: int = 10


class TrainerParams(BaseModel):
    """Parameters for the Lightning Trainer."""

    model_config = ConfigDict(extra="allow")

    max_epochs: int | None = None
    limit_train_batches: int | float | None = None


class LoggingParams(BaseModel):
    """Logging parameters for the autocast run."""

    model_config = ConfigDict(extra="allow")

    log_level: str = "info"


class TrainingConfig(BaseModel):
    """Top-level training configuration that governs the training loop structure.

    This corresponds to the 'training' key in the YAML config.
    """

    n_steps_input: AutoInt = "auto"
    n_steps_output: AutoInt = "auto"
    stride: AutoInt = "auto"
    rollout_stride: AutoInt = "auto"
    autoencoder_checkpoint: str | None = None
    freeze_autoencoder: bool = False


class OptimizerParams(BaseModel):
    """Parameters for the optimizer."""

    model_config = ConfigDict(extra="allow")


class Config(BaseModel):
    """Root configuration object."""

    seed: int = 42
    experiment_name: str = "default_experiment"

    data: DataParams = Field(default_factory=DataParams)
    model: ModelParams = Field(default_factory=ModelParams)
    trainer: TrainerParams = Field(default_factory=TrainerParams)
    logging: LoggingParams = Field(default_factory=LoggingParams)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    optimizer: OptimizerParams | None = None
    output: dict[str, Any] = Field(default_factory=dict)
