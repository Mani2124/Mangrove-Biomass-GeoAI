# Scientific dashboard update

- Replaced one holdout split with five-fold spatial cross-validation.
- Added a fold-specific training-mean baseline and validation pass/fail gate.
- Added bounded GEDI uncertainty weights.
- Refit final models on all accepted samples after validation.
- Added out-of-fold predictions and per-fold metrics.
- Added GEDI distribution, uncertainty and date diagnostics.
- Added Sentinel-2 clear-observation coverage diagnostics.
- Defaulted GEDI AOI and extrapolation filtering to conservative settings.
- Matched inference support to a circular 25 m GEDI footprint.
- Added fractional mangrove area for totals on coarser output grids.
- Added biomass percentiles, total biomass and clearer carbon scope.
- Added client-safe scientific summary download.
- Added explicit acknowledgement before mapping a failed model.
- Added validation/training shot-overlap protection to map validation.
- Removed credentials from the distributable project and added `.env.example`.
