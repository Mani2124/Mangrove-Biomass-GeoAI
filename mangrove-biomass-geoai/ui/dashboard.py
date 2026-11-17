"""Map-first dashboard. Science lives in sensors/, gedi/, ai/ and engines/."""
import json
import tempfile
import zipfile
from pathlib import Path
from datetime import date,timedelta
from uuid import uuid4
import streamlit as st
from shapely.geometry import shape
from shapely.geometry import mapping
import geopandas as gpd
from settings import ROOT
from core import jobs
from services.job_records import finish_job, recent_jobs, start_job
from services.projects import project_count, register_project, user_runs_root
from core.rasters import inspect_folder,coverage_message
from sensors import sentinel1,sentinel2,alos2
from ui import gedi_panel
from ui import experiment_panel
from ui import study_map, raster_view

PREFERENCES = ROOT/"local_settings.json"

def _safe_extract_zip(archive, destination):
    destination = Path(destination).resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != destination and destination not in target.parents:
            raise ValueError("ZIP contains an unsafe path.")
    archive.extractall(destination)

def execute(command, log, user_id, study_id, job_type):
    job_id = start_job(user_id, study_id, job_type)
    try:
        with st.spinner("Processing selected data. This can take time."):
            jobs.run(*command,log)
        finish_job(job_id, True, "Processing completed")
        st.success("Process ended. Check reports below for available outputs.")
    except Exception as error:
        finish_job(job_id, False, str(error))
        st.error(str(error))
    if log.exists():
        with st.expander("Processing details"):
            with log.open("rb") as source:
                source.seek(max(0,log.stat().st_size-20000))
                st.code(source.read().decode("utf-8",errors="replace"))

def main(user):
    runs = user_runs_root(user["id"])
    st.title("SpaceVexa Forest & Mangrove Explorer")
    st.caption("Load your ALOS-2 tiles → choose a study area → process selected sensors → inspect results")
    runs.mkdir(parents=True, exist_ok=True)
    if "alos_folder" not in st.session_state:
        try:
            st.session_state.alos_folder = json.loads(PREFERENCES.read_text()).get("alos_folder","")
        except (OSError,ValueError):
            st.session_state.alos_folder = ""
    with st.sidebar:
        st.header("1 · Load ALOS-2")
        folder = st.text_input("Folder containing extracted HH / HV files",key="alos_folder",
                              placeholder=r"D:\ALOS2\2024")
        st.caption("Keep the original filenames. Subfolders are included. Extract ZIP files first.")
        if st.button("Load / refresh available tiles",type="primary"):
            try:
                records,errors = inspect_folder(folder)
                st.session_state.records = records
                st.session_state.loaded_folder = str(Path(folder).resolve())
                PREFERENCES.write_text(json.dumps({"alos_folder":folder}))
                if not records:
                    st.warning("No readable HH/HV GeoTIFF files found.")
                for error in errors:
                    st.warning(error)
            except Exception as error:
                st.error(str(error))
        st.header("2 · Study area")
        studies = sorted(p for p in runs.iterdir() if (p/"input/aoi_wgs84.geojson").exists())
        def change_study():
            st.session_state.run = st.session_state.study_choice
        st.selectbox("Saved study",[None,*studies],key="study_choice",on_change=change_study,
                     format_func=lambda p:p.name if p else "Draw a new study")
        if st.button("New study"):
            st.session_state.run = None
            st.session_state.drawings = []
            st.rerun()
    records = st.session_state.get("records",[])
    run = st.session_state.get("run")
    tabs = st.tabs(["Explore & draw","Process sensors","Results & plots","GEDI reference","ML comparison","Account"])
    with tabs[0]:
        st.subheader("Study area: draw or upload")
        uploaded_aoi = st.file_uploader(
            "Upload AOI GeoJSON or zipped Shapefile",
            type=["geojson", "json", "zip"],
            help="Upload a polygon AOI. ZIP must contain .shp, .shx, .dbf and .prj files.",
        )
        if uploaded_aoi is not None:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    source = Path(temp_dir) / uploaded_aoi.name
                    source.write_bytes(uploaded_aoi.getbuffer())
                    read_path = source
                    if source.suffix.lower() == ".zip":
                        extract_dir = Path(temp_dir) / "shapefile"
                        with zipfile.ZipFile(source) as archive:
                            _safe_extract_zip(archive, extract_dir)
                        shp_files = list(extract_dir.rglob("*.shp"))
                        if not shp_files:
                            raise ValueError("ZIP does not contain a .shp file.")
                        read_path = shp_files[0]
                    uploaded_gdf = gpd.read_file(read_path).to_crs(4326)
                    uploaded_features = []
                    for geometry in uploaded_gdf.geometry:
                        if geometry is not None and not geometry.is_empty:
                            uploaded_features.append({
                                "type": "Feature",
                                "properties": {},
                                "geometry": mapping(geometry),
                            })
                    polygons = [f for f in uploaded_features if f["geometry"]["type"] in ("Polygon", "MultiPolygon")]
                    if not polygons:
                        raise ValueError("The uploaded file must contain polygon geometry.")
                    st.session_state["uploaded_aoi"] = polygons
                    st.success(f"Loaded {len(polygons)} polygon(s). You can save this AOI below.")
            except Exception as error:
                st.error(f"Could not read uploaded AOI: {error}")
        if not records:
            st.info("Enter your ALOS-2 folder at the left and click Load. Coverage will be read from the files, not guessed from their names.")
        else:
            st.success(f"{len(records)} HH/HV raster files loaded")
        layer = None
        if records:
            selected = st.selectbox("ALOS-2 layer",records,format_func=lambda r:r["name"])
            db = st.checkbox("Preview calibrated backscatter (dB)",value=False,
                             help="For original JAXA annual-mosaic DN only: 20 log10(DN) − 83. Do not use on files already in dB.")
            try:
                layer = raster_view.prepare(selected["path"],"Gamma0 backscatter (dB)" if db else "Original mosaic DN",db)
            except Exception as error:
                st.error(str(error))
        saved = json.loads((run/"input/aoi_wgs84.geojson").read_text()) if run else None
        if st.session_state.get("uploaded_aoi"):
            uploaded_collection = {"type": "FeatureCollection", "features": st.session_state["uploaded_aoi"]}
            saved = uploaded_collection
        features = study_map.render(records,layer,saved)
        if st.session_state.get("uploaded_aoi"):
            features = st.session_state["uploaded_aoi"] + [f for f in features if f["geometry"]["type"] == "Point"]
        st.caption("Orange outlines = downloaded raster bounds, not mangrove boundaries or guaranteed valid data. Draw a polygon inside the coverage. Markers are saved separately as POIs.")
        if st.button("Save AOI and POIs",type="primary"):
            polygons = [f for f in features if f["geometry"]["type"] in ("Polygon","MultiPolygon")]
            points = [f for f in features if f["geometry"]["type"]=="Point"]
            if not polygons or any(not shape(f["geometry"]).is_valid or shape(f["geometry"]).is_empty for f in polygons):
                st.error("Draw a valid polygon first.")
            else:
                if user["plan"] == "demo" and project_count(user["id"]) >= 3:
                    st.error("The demo plan supports three saved studies.")
                    st.stop()
                run = runs/("study_"+uuid4().hex[:10])
                (run/"input").mkdir(parents=True)
                for name,items in [("aoi_wgs84",polygons),("pois",points)]:
                    (run/"input"/f"{name}.geojson").write_text(json.dumps({"type":"FeatureCollection","features":items}))
                st.session_state.run = run
                register_project(user["id"], run.name)
                st.success(f"Saved {run.name}")
        if layer:
            with st.expander("Raster image, legend and histogram",expanded=True):
                raster_view.plot(layer, key="alos_explore")
    with tabs[1]:
        if not run:
            st.info("Save an AOI on Explore & draw first.")
        else:
            st.write(f"Study: {run.name}")
            start = st.date_input("Sentinel start date",date.today()-timedelta(days=180))
            end = st.date_input("Sentinel end date",date.today())
            scenes = st.number_input("Maximum scenes per Sentinel sensor",1,12,2)
            chosen = st.multiselect("Select sensors to run",["Sentinel-2","Sentinel-1","ALOS-2"],default=["ALOS-2"])
            years = sorted({r["year"] for r in records if r["year"]})
            year = st.number_input("ALOS-2 mosaic year",2015,2100,years[-1] if years else 2024)
            ready = False
            if records:
                polygons = json.loads((run/"input/aoi_wgs84.geojson").read_text())["features"]
                ready,message = coverage_message(polygons,records,year)
                (st.success if ready else st.warning)(message)
                st.caption(f"ALOS source: {st.session_state.loaded_folder}")
            else:
                st.info("Load the ALOS-2 folder before running ALOS-2.")
            if st.button("Run selected sensors",disabled=not chosen):
                if start>end:
                    st.error("Start date must be before end date.")
                elif "ALOS-2" in chosen and not ready:
                    st.error("Load matching HH and HV tiles covering the AOI first.")
                else:
                    if "ALOS-2" in chosen:
                        (run/"input/alos_source.json").write_text(json.dumps({"folder":st.session_state.loaded_folder}))
                    commands = {"Sentinel-2":sentinel2.command(run,start,end,scenes),
                                "Sentinel-1":sentinel1.command(run,start,end,scenes),"ALOS-2":alos2.command(run,year)}
                    for name in chosen:
                        execute(
                            commands[name], run/"logs"/f"{name}.log",
                            user["id"], run.name, name,
                        )
    with tabs[2]:
        files = sorted((run/"processed").rglob("*.tif"))+sorted((run/"prediction").rglob("*.tif")) if run else []
        if not files:
            st.info("Processed rasters will appear here. You can already preview downloaded ALOS-2 on Explore & draw.")
        else:
            path = st.selectbox("Processed raster",files,format_func=lambda p:str(p.relative_to(run)))
            unit = st.text_input("Legend label / units","Raster values (check layer metadata)")
            try:
                output = raster_view.prepare(path,unit)
                raster_view.plot(output, key="processed_results")
            except Exception as error:
                st.error(str(error))
    with tabs[3]:
        gedi_panel.render(run)
    with tabs[4]:
        experiment_panel.render(run)
    with tabs[5]:
        st.subheader("Account and processing activity")
        columns = st.columns(3)
        columns[0].metric("Plan", user["plan"].title())
        columns[1].metric("Saved studies", project_count(user["id"]))
        columns[2].metric("Demo study limit", 3 if user["plan"] == "demo" else "Custom")
        st.caption("Studies and outputs are isolated by account in the application workspace.")
        activity = recent_jobs(user["id"])
        if activity:
            st.dataframe(activity, width="stretch", hide_index=True)
        else:
            st.info("Processing activity will appear here after a sensor job is run.")
