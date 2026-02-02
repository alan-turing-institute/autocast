# Developer Guide

## Overview
AutoCast is a library for spatiotemporal forecasting. This guide covers the internal API and development practices. For information on running scripts and configuring experiments, please see [Scripts and Configuration](SCRIPTS_AND_CONFIGS.md).

## API notes


### Trainer and Model Integration
Example usage with `lightning` Trainer:

```python
model = EncoderDecoder()  # Anything that inherits for L.LightningModule
trainer = L.Trainer()
trainer.fit(model, train_dataloader)  # train_dataloader should output a batch of data from an iterable.

model = EncoderProcessorDecoder()
trainer = L.Trainer()
trainer.fit(model, train_dataloader)
```

### Model API
Subclasses of `LightningModule` from `lightning` aim to have API:
```python
def training_step(self, batch: Batch, batch_idx: int) -> Tensor: ...
def forward(self, x: Tensor) -> Tensor: ...
```

Direct subclasses of `nn.Module` from `torch` aim to have API:
```python
def forward(self, x: Tensor) -> Tensor: ...