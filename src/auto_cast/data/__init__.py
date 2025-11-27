from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import lightning as L
import torch
from torch import Tensor, nn
from torch.utils.data.dataloader import DataLoader

Input = Tensor | DataLoader
RolloutOutput = tuple[Tensor, None] | tuple[Tensor, Tensor]

Batch = dict[str, Any]

# Lightnign modules take in:
# - training step(batch: Any, batch_idx: int) -> Tensor
# - forward(x: Tensor) -> Tensor
# nn.Module take in:
# - forward(x: Tensor) -> Tensor

# Could be a dataclass if we want more structure
# @dataclass
# class Batch:
#     input_fields: Tensor
#     output_fields: Tensor
#     constant_scalars: Tensor
#     constant_fields: Tensor


class Encoder(nn.Module):
    """Base encoder."""

    # Option 1
    def forward(self, x: Tensor) -> Tensor:
        """Forward Pass through the Encoder."""
        msg = "To implement."
        raise NotImplementedError(msg)

    # Option 2
    # def forward(self, x: Batch) -> Tensor:
    #     """Forward Pass through the Encoder."""
    #     msg = "To implement."
    #     raise NotImplementedError(msg)


class Preprocessor(nn.Module):
    """Base Preprocessor."""

    def forward(self, x: Batch) -> Tensor:
        """Forward Pass through the Preprocessor."""
        msg = "To implement."
        raise NotImplementedError(msg)


class Processor(L.LightningModule, ABC):
    """Processor Base Class."""

    teacher_forcing_ratio: float
    stride: int
    max_rolout_steps: int
    preprocessor: Preprocessor

    def __init__(self):
        pass

    # Option 1
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the Processor."""
        msg = "To implement."
        raise NotImplementedError(msg)

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        x = self.preprocessor(batch)
        return self(x)

    # # Option 2
    # def forward(self, x: Batch) -> Tensor:
    #     """Forward pass through the Processor."""
    #     msg = "To implement."
    #     raise NotImplementedError(msg)

    # def training_step(self, batch: Batch, batch_idx: int):
    #     self(batch)

    def configure_optmizers(self):
        pass

    def rollout(self, batch: Batch) -> RolloutOutput:
        """Rollout over multiple time steps."""
        pred_outs = []
        gt_outs = []
        for time_step in range(0, self.max_rolout_steps, self.stride):
            x = self.preprocessor(batch)
            pred_outs.append(self(x))
            gt_outs.append(
                batch["output_fields"]
            )  # Q: this assumes we have output fields
        return torch.stack(pred_outs), torch.stack(gt_outs)


class DiscreteProcessor(Processor, ABC):
    @abstractmethod
    def map(self, x: Batch) -> Tensor:
        ...
        # Map input window of states/times to output window

    def rollout(self, x: Batch) -> RolloutOutput:
        ...
        # Use self.map to generate trajectory


class FlowBasedGenerativeProcessor(DiscreteProcessor):
    def map(self, x: Batch) -> Tensor:
        ...
        # Sample generative model    def loss(self, ...):...
        # Flow matc


class Decoder(nn.Module):
    """Base Decoder"""

    # Q: Should decoder handle all these input types
    def forward(self, x: Batch) -> Tensor: ...


class EncoderDecoder(L.LightningModule):
    encoder: Encoder
    decoder: Decoder
    preprocessor: Preprocessor
    loss_func: nn.Module

    def __init__(self):
        pass

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(self.encoder(x))

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        x = self.preprocessor(batch)
        output = self.decoder(self.encoder(x))
        loss = self.loss_func(output, batch["output_fields"])
        return loss

    def encode_only(self, x: Batch) -> Tensor:
        x = self.preprocessor(x)
        return self.encoder(x)

    def configure_optmizers(self):
        pass


class VAE(EncoderDecoder):
    def forward(self, x: Tensor) -> Tensor:
        mu, log_var = self.encoder(x)
        z = self.reparametrize(mu, log_var)
        x = self.decoder(z)
        return x

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class EncoderProcessorDecoder(L.LightningModule):
    encoder_decoder: EncoderDecoder
    processor: Processor

    def __init__(self):
        pass

    def forward(self, x: DataLoader) -> Tensor:
        return self.encoder_decoder.decoder(
            self.processor(self.encoder_decoder.encoder(x))
        )

    def training_step(batch, batch_idx):
        pass

    def configure_optmizers(self):
        pass


model = EncoderDecoder()  # Anythign that inherits for L.LightningModule
trainer = L.Trainer()
trainer.fit(
    model, train_dataloader
)  # train_dataloader should output a batch of data as an iterable. Is this batch a tensor or dictionary?

model = EncoderProcessorDecoder()
trainer = L.Trainer()
trainer.fit(model, train_dataloader)
