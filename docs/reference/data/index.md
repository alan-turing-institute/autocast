# Data

Data loading, preprocessing, and dataset implementations.

## Validation and test trajectories

`SpatioTemporalDataModule` can select whole validation and test trajectories
before constructing windows or distributed sampling. Both options default to
`null`, which uses the complete split:

```yaml
datamodule:
  valid_trajectories: 20
  test_trajectories: [0, 3, 7]
```

A positive integer selects the first N trajectories in the split's stored order.
A list selects the specified zero-based indices in that order. Empty lists,
duplicates, invalid index types and unavailable indices raise an error. The
training split is unaffected. For example, a split with 100 validation
trajectories can use 20 for routine checks and all 100 with
`valid_trajectories: null`.

Windowed and full-trajectory evaluation use the same selection and preserve
original trajectory IDs in trajectory statistics. Normalization continues to
use the supplied training statistics for every split; selecting validation or
test trajectories does not recompute them. These options currently apply to
the `.pt`/HDF5 spatiotemporal datamodule.
