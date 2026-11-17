"""Previous interface retained for reference; start app.py instead."""
import json
from datetime import date, timedelta
from uuid import uuid4
import streamlit as st
from shapely.geometry import shape
from settings import ROOT, RUNS
from core import jobs
from sensors import sentinel1, sentinel2, alos2
from gedi import labels
from ai import train, predict
from ui import map_panel, results

st.set_page_config(page_title="Biomass platform", layout="wide")
st.title("Forest & mangrove biomass")
st.caption("Research prototype. GEDI is reference data with uncertainty, not exact ground truth.")

def execute(command, log):
    try:
        with st.spinner("Processing. Logs are written to disk; this may take time."):
            jobs.run(*command, log)
        st.success("Process finished. Check reports for actual data availability.")
    except Exception as error:
        st.error(str(error))
    if log.exists():
        with st.expander("Log"):
            st.code(log.read_text(encoding="utf-8",errors="replace")[-20000:])

RUNS.mkdir(exist_ok=True)
studies = sorted(p for p in RUNS.iterdir() if (p/"input/aoi_wgs84.geojson").exists())
selected = st.sidebar.selectbox("Existing study", [None,*studies],
                                format_func=lambda p:p.name if p else "New study")
if selected:
    st.session_state.run = selected
map_tab, sensor_tab, output_tab, ai_tab = st.tabs(["Map / AOI","Sensors","Images & plots","GEDI & AI"])
with map_tab:
    features = map_panel.render()
    if st.button("Save drawn AOI and POIs"):
        polygons = [f for f in features if f["geometry"]["type"] in ("Polygon","MultiPolygon")]
        points = [f for f in features if f["geometry"]["type"] == "Point"]
        if not polygons or any(not shape(f["geometry"]).is_valid for f in polygons):
            st.error("Draw a valid polygon first.")
        else:
            run = RUNS/("study_"+uuid4().hex[:10])
            (run/"input").mkdir(parents=True)
            for name, items in [("aoi_wgs84",polygons),("pois",points)]:
                (run/"input"/f"{name}.geojson").write_text(json.dumps(
                    {"type":"FeatureCollection","features":items}))
            st.session_state.run = run
            st.success(f"Saved {run.name}")
run = st.session_state.get("run")
with sensor_tab:
    if not run:
        st.info("Save or select a study first.")
    else:
        st.write(run.name)
        start = st.date_input("Start",date.today()-timedelta(days=180))
        end = st.date_input("End",date.today())
        scenes = st.number_input("Maximum scenes per Sentinel sensor",1,12,2)
        chosen = st.multiselect("Sensors",["Sentinel-2","Sentinel-1","ALOS-2"],default=["Sentinel-2"])
        year = st.number_input("ALOS-2 downloaded mosaic year",2015,2100,2024)
        st.info(f"Put local HH/HV mosaics in {run/'raw/alos2'}. No automatic ALOS-2 download.")
        if st.button("Run selected sensors",disabled=not chosen):
            if start > end:
                st.error("Start must be before end.")
            else:
                commands = {"Sentinel-2":sentinel2.command(run,start,end,scenes),
                            "Sentinel-1":sentinel1.command(run,start,end,scenes),
                            "ALOS-2":alos2.command(run,year)}
                for name in chosen:
                    execute(commands[name],run/"logs"/f"{name}.log")
with output_tab:
    if run:
        results.render(run)
    else:
        st.info("Select a study first.")
with ai_tab:
    st.subheader("1. GEDI L4B reference labels")
    folder = st.text_input("Local GEDI L4B raster folder")
    if st.button("Prepare labels",disabled=not folder):
        execute(labels.command(folder,ROOT/"training/gedi_l4b"),ROOT/"logs/gedi.log")
    st.subheader("2. Supervised training")
    st.warning("First join satellite statistics to GEDI cell labels using the original training engines. This join is not automated in this screen. Match dates and cell support.")
    csv = st.text_input("Joined training CSV path")
    model = ROOT/"models/custom"
    if st.button("Train Random Forest",disabled=not csv):
        try:
            execute(train.command(csv,model),ROOT/"logs/train.log")
        except Exception as error:
            st.error(str(error))
    st.subheader("3. Biomass prediction")
    st.caption("Only trusted, locally trained joblib models. Inspect validation reports before using predictions.")
    if st.button("Estimate biomass",disabled=not(run and (model/"random_forest_biomass.joblib").exists())):
        execute(predict.command(run,model),run/"logs/predict.log")
    st.caption("Metrics and validation tables: models/custom. No calibrated uncertainty or carbon totals in this version.")
