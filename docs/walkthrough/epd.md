# Full encoder-processor-decoder

In the final part of the walkthrough, we'll train a full encoder-processor-decoder stack on our data with a nontrivial encoder.
We'll use our previously trained autoencoder to map our data into a latent space and back; the processor will be trained on the latent-space compressed data.

The necessary invocation is an extension of the `epd` command which we saw [at the end of the previous page](./processor.md).
Run this from inside the `autocast` directory:

```
uv run autocast epd \
    --workdir ../full_epd_output \
    ++datamodule.data_path=/path/to/parent_folder/ad_data \
    processor@model.processor=flow_matching \
    ++trainer.max_epochs=10 \
    encoder@model.encoder=dc_deep_256_v2 \
    decoder@model.decoder=dc_deep_256_v2 \
    ++autoencoder_checkpoint=/path/to/parent_folder/ae_output/autoencoder.ckpt
```

As before, we specify the path to the original simulated data (`ad_data`), and for the purposes of this demonstration we use the smaller `flow_matching` processor.

The new parts here are the encoder and decoder specifications.
These have to be set to the same architecture as the autoencoder we trained previously, so that the latent space is compatible.
When we [trained the autoencoder](./autoencoder.md), we simply went with the default architecture, which (as described in that section) is `dc_deep_256_v2`.
We thus set both `encoder` and `decoder` to this.
Finally we provide the path to the trained autoencoder checkpoint.

By default, `autocast` freezes the encoder and decoder weights: they aren't updated in tandem with the processor.
We can see this in the model summary that's printed before training starts:

```
  | Name            | Type                  | Params | Mode
------------------------------------------------------------------
0 | encoder_decoder | EncoderDecoder        | 6.9 M  | eval
1 | processor       | FlowMatchingProcessor | 3.5 M  | train
2 | loss_func       | MSELoss               | 0      | train
3 | val_metrics     | MetricCollection      | 0      | train
4 | test_metrics    | MetricCollection      | 0      | train
------------------------------------------------------------------
3.5 M     Trainable params
6.9 M     Non-trainable params
10.4 M    Total params
```

To change this, you can provide the `++model.freeze_encoder_decoder=false ++model.train_in_latent_space=false` overrides (both must be present).
This will allow the encoder and decoder weights to be updated during training, i.e., the full encoder-processor-decoder stack will be trained end-to-end.
That yields a model summary like this:

```
  | Name            | Type                  | Params | Mode
------------------------------------------------------------------
0 | encoder_decoder | EncoderDecoder        | 6.9 M  | train
1 | processor       | FlowMatchingProcessor | 3.5 M  | train
2 | loss_func       | MSELoss               | 0      | train
3 | val_metrics     | MetricCollection      | 0      | train
4 | test_metrics    | MetricCollection      | 0      | train
------------------------------------------------------------------
10.4 M    Trainable params
0         Non-trainable params
10.4 M    Total params
```

You can also choose `++model.train_in_latent_space=false` but with `++model.freeze_encoder_decoder=true` to calculate the loss in the original data space, but without updating the encoder and decoder weights.
