# Scientific biomass workflow

## What the platform estimates

The platform estimates above-ground biomass density (AGBD, Mg/ha) inside a
mangrove mask. GEDI L4A footprints provide reference labels. Sentinel-2,
Sentinel-1 and ALOS-2 provide spatial predictors. Above-ground living carbon is
reported as AGBD multiplied by the documented prototype carbon fraction 0.47.
Roots, deadwood, litter and soil carbon are not included.

## Recommended order

1. Draw or upload an AOI.
2. Process Sentinel-2. Review the valid-observation-count layer before continuing.
3. Optionally process one month of Sentinel-1 and the annual ALOS-2 mosaic.
4. Prepare Global Mangrove Watch or upload a locally validated mangrove mask.
5. Download/read GEDI L4A granules intersecting the AOI.
6. Keep L2/L4 quality observations, exclude degraded and extrapolated records,
   and retain observations inside the AOI.
7. Match every accepted GEDI footprint to satellite predictors using a 25 m
   footprint and require valid predictor coverage.
8. Keep mangrove footprints meeting the selected mangrove-coverage threshold.
9. Compare the three sensor models with spatial cross-validation. The same
   samples and folds are used for every model.
10. Select a model only when it outperforms the spatial mean baseline. Refit the
    selected model on all accepted samples.
11. Predict AGBD on the 25 m grid. Coarser display grids use fractional mangrove
    coverage for area and total calculations.
12. Report validation metrics, coverage, model status, biomass, above-ground
    carbon, assumptions and limitations together.

## Scientific status

- **Failed validation:** spatial R2 is not positive or RMSE does not beat the
  fold-specific training-mean baseline. Maps are demonstration-only.
- **Experimental:** the model passes internal spatial validation but has no
  independent field validation.
- **Locally validated:** use this wording only after comparison with independent
  field plots or another defensible independent reference dataset.

There is no universal R2 threshold that makes a biomass product correct. Model
fitness depends on the use case, biomass range, spatial scale and required
uncertainty.

## Client-safe explanation

The platform learns the relationship between quality-filtered GEDI biomass
footprints and co-located satellite measurements. Geographic blocks are withheld
during validation so nearby observations do not appear in both training and
testing. Validation results and data coverage are shown before map totals. The
final model is refitted using all accepted samples only after the model family
has been evaluated. Carbon values are an above-ground living-carbon proxy, not a
certified carbon-credit quantity.
