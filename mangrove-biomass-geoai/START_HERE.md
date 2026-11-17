# Start here

Read `SCIENTIFIC_WORKFLOW.md` before interpreting biomass or carbon values.

## Install

```powershell
cd "D:\biomass_platform_modular\modular_project"
D:\Anaconda\python.exe -m pip install -r requirements.txt
D:\Anaconda\python.exe -m streamlit run app.py
```

## Follow this order

1. Save an AOI.
2. Process Sentinel-2, Sentinel-1 and ALOS-2.
3. Select and filter several overlapping GEDI L4A V3 files.
4. Download the merged filtered GEDI CSV.
5. Open **Mangrove AI comparison**.
6. Prepare Global Mangrove Watch or upload a custom mask.
7. Upload the merged GEDI CSV.
8. Build common samples.
9. Train and compare the three models.
10. Create maps from the best validated model.

The output is above-ground biomass density and indicative above-ground carbon. It is not total ecosystem carbon or a certified carbon-credit estimate.
