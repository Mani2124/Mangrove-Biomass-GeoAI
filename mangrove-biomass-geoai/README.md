> **Research prototype:** Sundarbans multi-sensor mangrove biomass workflow. Experimental model; not independently field-validated or suitable for carbon-credit reporting.

# Forest & Mangrove Biomass Platform

This Streamlit project estimates **mangrove above-ground biomass density (AGBD)** using quality-filtered GEDI L4A footprints as supervised labels and compares three satellite predictor combinations fairly:

1. Sentinel-2
2. Sentinel-2 + Sentinel-1
3. Sentinel-2 + Sentinel-1 + ALOS-2 PALSAR-2

Every model uses the same mangrove-only GEDI observations and the same spatial cross-validation folds. This is a sensor-ablation experiment: it measures whether radar adds predictive value beyond the Sentinel-2 baseline.


## Input data: ALOS-2 and study-area boundary

**Download these separately.** No ALOS-2 tile or AOI boundary is bundled with the public code.

| Input | Local folder (relative to project root) | How to use |
|---|---|---|
| ALOS-2 PALSAR-2 annual mosaic HH/HV GeoTIFF tiles | `local_alos2/` | Download the tiles covering your AOI from the JAXA dataset portal, extract them, and paste the absolute folder path into **Explore & draw → Load ALOS-2**. Click **Load / refresh available tiles**. |
| Study boundary (`.geojson`, `.json` or zipped Shapefile) | `local_aoi/` | Choose the file using **Explore & draw → Upload AOI**, then click **Save AOI and POIs**. Alternatively draw an AOI on the map. |
| GEDI L4A V3 granules (optional local input) | `local_gedi/` | The **GEDI reference** tab can search/download overlapping GEDI files using NASA Earthdata credentials, or you can point it to local `.h5` granules. |

Example Windows folders after extraction:

```text
mangrove-biomass-geoai/
  local_alos2/             # your extracted HH/HV mosaic tiles (not committed)
  local_aoi/               # your boundary file (not committed)
  local_gedi/              # optional GEDI HDF5 files (not committed)
  runs/                    # generated automatically; not committed
  data_cache/              # downloaded mangrove mask cache; not committed
```

**Download sources:** JAXA Earth Observation Research Center (EORC) annual PALSAR-2 mosaic portal: https://www.eorc.jaxa.jp/ALOS/en/dataset/ ; obtain your own AOI boundary from an authoritative source or create/export one in QGIS. The exact JAXA product format and availability depend on the selected annual mosaic.

**What runs automatically after configuration:** the existing Sentinel-2 and Sentinel-1 engines perform their catalogue/download and processing steps using Copernicus credentials; the mangrove-mask workflow can download/cache Global Mangrove Watch; GEDI has an Earthdata search/download option; feature matching, model training, spatial validation and prediction run when you start the corresponding dashboard actions. These steps are **not an unattended one-click pipeline**: choose the AOI, dates and processing actions in the interface. ALOS-2 is optional in the app but must be supplied locally if selected. Downloads require connectivity, accounts where applicable, and available imagery.

**Important:** Do not upload your AOI or imagery to GitHub unless you have permission to redistribute it. The app stores the saved AOI under the selected study's `input/aoi_wgs84.geojson`.

## Scientific workflow

1. Draw or upload a polygon AOI.
2. Process Sentinel-2 (required), Sentinel-1 and ALOS-2.
3. Read one or several GEDI L4A V3 HDF5 granules.
4. Apply GEDI quality filters and AOI filtering.
5. Prepare the mangrove mask:
   - automatic Global Mangrove Watch 2020 v4.0.19 (10 m), or
   - custom GeoTIFF/GeoJSON/GeoPackage/zipped Shapefile.
6. Select a documented mangrove-coverage threshold within each approximate 25 m GEDI footprint (60% is the dashboard default).
7. Sample all satellite predictors at the remaining footprints.
8. Retain only common complete samples available to all three models.
9. Create five spatial cross-validation folds.
10. Train three uncertainty-weighted Random Forest regressors with identical settings.
11. Compare R², MAE, RMSE and bias using out-of-fold GEDI predictions and a spatial mean baseline.
12. Refit each final model on all accepted samples after validation.
13. Predict mangrove-only AGBD with a scientifically labelled selected model.
14. Calculate indicative above-ground carbon as `AGC = AGBD × 0.47` using fractional mangrove area.

## Install and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Use the dashboard tabs from left to right:

1. **Explore & draw** — load ALOS-2 folders and save an AOI.
2. **Process sensors** — process Sentinel-2, Sentinel-1 and ALOS-2.
3. **Results & plots** — inspect every processed raster.
4. **GEDI reference** — select multiple L4A V3 HDF5 granules, filter and export one merged CSV.
5. **Mangrove AI comparison** — prepare mask, build common samples, train/validate three models and create final maps.

## Mangrove mask

The default is Global Mangrove Watch 2020 v4.0.19 (DOI `10.5281/zenodo.12756047`). The official archive is approximately 180 MB and is cached in `data_cache/` after its first download. The dashboard clips and reprojects only intersecting tiles to the Sentinel-2 reference grid using nearest-neighbour resampling.

Use a custom locally validated boundary when it is newer or more accurate. A custom raster must have positive values for mangrove; vector uploads are rasterized internally.

## Fair comparison rules

- Never compare models trained on different GEDI samples.
- Never use a random point split when neighbouring GEDI footprints are spatially autocorrelated.
- All three models use `common_samples_with_spatial_folds.csv`.
- Every out-of-fold prediction comes from a model that did not train on that footprint's spatial block.
- A larger training region may be used when the prediction AOI has few GEDI footprints.
- A transferred model without local GEDI/field validation must be labelled as transferred, not locally validated.

## Main outputs

Within each study folder:

```text
training/mangrove_sample_audit.csv
training/mangrove_common_samples.csv
models/three_model_comparison/model_comparison.csv
models/three_model_comparison/three_model_validation.png
models/three_model_comparison/common_samples_with_spatial_folds.csv
models/three_model_comparison/<model>/out_of_fold_validation.csv
models/three_model_comparison/<model>/spatial_fold_metrics.csv
models/three_model_comparison/<model>/feature_importance.csv
prediction/<model>/mangrove_only/mangrove_AGBD_Mg_ha.tif
prediction/<model>/mangrove_only/mangrove_AGC_Mg_C_ha.tif
prediction/<model>/mangrove_only/mangrove_fraction.tif
prediction/<model>/mangrove_only/mangrove_prediction_summary.json
```

## Interpretation

- Higher R² is better, but R² alone is insufficient.
- Lower MAE and RMSE are better.
- Bias should be close to zero.
- A sensor combination is an improvement only when it performs better on the same held-out footprints.
- GEDI is a remote-sensing reference with uncertainty, not error-free ground truth.
- Field plots or an independent published/local dataset are required for strong external validation.
- Carbon output covers above-ground living biomass only. It excludes roots, deadwood, litter and soil organic carbon.

## Security and GitHub

The project ZIP intentionally contains no `.env` file. Create your own local `.env` from `.env.example`. Never commit credentials, HDF5 granules, downloaded scenes, trained models or study outputs. If a previous `.env` containing a real password was shared or committed, change that password.

## Known limitations

- GMW 2020 may not represent recent mangrove gain/loss.
- The 25 m GEDI footprint is approximated as a uniform circle.
- Cloud masking and acquisition-date harmonisation depend on the existing sensor engines.
- A minimum of 50 common samples is enforced; 300+ well-distributed samples and several spatial blocks are recommended.
- Results are research/prototype estimates and are not certified carbon-credit quantities.
