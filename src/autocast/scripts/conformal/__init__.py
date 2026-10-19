"""Calibrate saved ensemble forecasts with conformal prediction and EMOS.

Consumes the prediction files written by the eval's prediction-saving mode
(``eval.dump_rollout_tensors``) and writes one ``eval_conformal/`` folder per
model, laid out as described in :mod:`autocast.scripts.conformal.calibrate`.

Module map
----------
- :mod:`autocast.scripts.conformal.data` -- loading eval-dump prediction
  files, the fixed/balanced calibration-test trajectory splits, and the run
  manifest.
- :mod:`autocast.scripts.conformal.scoring` -- fitting raw/EMOS/conformal and
  computing every score (see that module's docstring for which score comes
  from where).
- :mod:`autocast.scripts.conformal.writers` -- writing the fixed output
  layout's CSV/``.pt``/``.json`` files.
- :mod:`autocast.scripts.conformal.calibrate` -- the ``calibrate`` CLI: all
  four calibration-source x test-source combinations for one model.
- :mod:`autocast.scripts.conformal.sufficiency` -- the ``sufficiency`` CLI:
  the calibration-trajectory-count sweep on the new set.
"""
