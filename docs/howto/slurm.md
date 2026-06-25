# Running `autocast` on SLURM

`autocast` supports running experiments on SLURM clusters by adding the `--mode slurm` flag.
This automatically generates a submission Bash script and submits it to the cluster, so you don't have to worry about writing your own submission scripts.

There are also a few [`distributed` configurations](https://github.com/alan-turing-institute/autocast/tree/main/src/autocast/configs/distributed) which allow you to specify the number of nodes/GPUs to use.

In particular, you can add the following command-line arguments:

- `distributed: single_gpu_slurm` to only request one GPU, even if a compute node has more than one GPU
- `distributed: ddp_4gpu_slurm` to request 4 GPUs on a single node
- `distributed: ddp_4gpu_2node_slurm` to request 4 GPUs per node across 2 nodes (8 GPUs in total)

If you want to use more than 2 nodes, you can use `ddp_4gpu_2node_slurm` and additionally specify the following overrides:

```
++trainer.num_nodes=3 ++eval.num_nodes=3 ++hydra.launcher.nodes=3
```

(replace `3` with the number of nodes you want to use).
