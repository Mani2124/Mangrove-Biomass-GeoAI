# Data setup (Windows)

1. Download **ALOS-2 PALSAR-2 annual mosaic** HH and HV tiles from the JAXA data portal for the year and area of interest. Extract all files into `local_alos2/`, keeping their filenames.
2. Prepare the study area in WGS84 GeoJSON (`local_aoi/study_area.geojson`) or a ZIP of the Shapefile components (`.shp`, `.shx`, `.dbf`, `.prj`). The application reprojects uploaded vector files to EPSG:4326.
3. Open PowerShell in the project root and run `python -m streamlit run app.py`.
4. In the **Explore & draw** tab, paste the **absolute path** to `local_alos2` in the ALOS-2 sidebar field, then press **Load / refresh available tiles**.
5. Upload the boundary from `local_aoi` using **Upload AOI GeoJSON or zipped Shapefile**, and click **Save AOI and POIs**. Alternatively draw your AOI.
6. In **Process sensors**, select the desired sensors and date range. Sentinel-2 and Sentinel-1 download/processing use the existing Copernicus engine and require valid credentials in your local `.env`. ALOS-2 uses the supplied local tiles.
7. In **GEDI reference**, use the built-in NASA Earthdata search/download (requires credentials), or point to a folder containing GEDI L4A V3 `.h5` granules.
8. In **ML comparison**, obtain the GMW mangrove mask automatically (or supply a custom mask), prepare samples, train/validate models and generate experimental biomass maps.

All downloaded and generated data remain local and are excluded from Git. Some actions are manual dashboard steps; automatic refers to data retrieval and computation **after you start each step**, not an end-to-end unattended run.

If the application reports no ALOS-2 coverage, check that the extracted HH/HV files overlap your AOI, have supported georeferencing, and are readable GeoTIFFs.
