# SpaceVexa SaaS and three-algorithm scientific prototype

## What was added

- Local account registration and sign-in
- PBKDF2 password hashing
- User-isolated study folders
- Project and processing-job records in SQLite
- Account/activity tab and a three-study demo limit
- Docker configuration
- Separate comparison of three machine-learning algorithms

## The two comparisons are different

1. **Sensor comparison** keeps Random Forest fixed and compares:
   Sentinel-2; Sentinel-2 + Sentinel-1; and Sentinel-2 + Sentinel-1 + ALOS-2.
2. **Algorithm comparison** keeps the common samples, all available predictors,
   spatial blocks and folds fixed, and compares Random Forest, Extra Trees and
   Histogram Gradient Boosting.

The dashboard produces spatial out-of-fold R², RMSE, MAE, bias, fold RMSE
variation, mean-baseline comparison, scatterplots and downloadable CSV files.
No result is fabricated in the package: numerical results are generated after
the user builds a common GEDI table and runs the comparison.

## Run locally

```powershell
copy .env.example .env
D:\Anaconda\python.exe -m pip install -r requirements.txt
D:\Anaconda\python.exe -m streamlit run app.py
```

## Streamlit Community Cloud

The interface can be deployed from GitHub. Do not upload `.env`, downloaded
satellite scenes, HDF5 files, GeoTIFFs, trained models or `saas_data`.

Community Cloud is suitable for viewing results and small demonstrations. SNAP,
large Sentinel downloads and long model jobs should run on a dedicated worker.

## Production migration

The included local authentication and SQLite database are intended for a
controlled prototype. For public multi-user processing, migrate authentication
and project records to Supabase/PostgreSQL, output files to private object
storage, and processing to a worker queue.

## Scientific status

Internal GEDI spatial cross-validation is not independent field validation.
Biomass and carbon outputs remain experimental. Carbon uses an above-ground
conversion factor and is not a certified carbon-credit estimate.
