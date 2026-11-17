"""Mangrove mask, fair sensor comparison, and final map panel."""
from __future__ import annotations

import json
import shutil
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import rasterio
import streamlit as st

from settings import ROOT
from ai.build_training_table import build_training_table
from ai.compare_algorithms import ALGORITHMS, compare_algorithms
from ai.compare_models import MODEL_DEFINITIONS, train_comparison
from core.mangrove_mask import (
    GMW_DOI, add_mangrove_coverage, apply_mask_to_prediction,
    build_aligned_mask, download_gmw,
)
from engines.inference_01_predict_biomass import predict_biomass
from ui import raster_view


def _status_box(summary):
    status = summary.get("validation_status", "unknown")
    message = summary.get("status_message", "Validation status is unavailable.")
    if status in {"failed_validation", "unknown"}:
        st.error("Scientific status: FAILED VALIDATION — " + message)
    elif status == "experimental":
        st.warning("Scientific status: EXPERIMENTAL — " + message)
    else:
        st.info("Scientific status: " + status.replace("_", " ").upper() + " — " + message)


def _client_report(run, summary, final=None):
    best = summary.get("best_model_by_rmse", "not available")
    metrics = {}
    if best != "not available":
        metric_file = run / "models" / "three_model_comparison" / best / "metrics.json"
        if metric_file.exists():
            metrics = json.loads(metric_file.read_text(encoding="utf-8"))
    lines = [
        "MANGROVE BIOMASS AND ABOVE-GROUND CARBON — SCIENTIFIC SUMMARY",
        f"Study: {run.name}",
        f"Selected model: {best}",
        f"Scientific status: {summary.get('validation_status', 'unknown')}",
        f"Spatial folds: {summary.get('spatial_folds', '—')}",
        f"Spatial R2: {metrics.get('r2', '—')}",
        f"RMSE (Mg/ha): {metrics.get('rmse_mg_ha', '—')}",
        f"MAE (Mg/ha): {metrics.get('mae_mg_ha', '—')}",
        f"Bias (Mg/ha): {metrics.get('bias_mg_ha', '—')}",
        "Reference: quality-filtered GEDI L4A footprint AGBD.",
        "Predictors: Sentinel-2; optional Sentinel-1 and ALOS-2.",
    ]
    if final:
        lines += [
            f"Fractional mangrove area (ha): {final.get('mangrove_area_ha', '—')}",
            f"Mean AGBD (Mg/ha): {final.get('mean_agbd_mg_ha', '—')}",
            f"Median AGBD (Mg/ha): {final.get('median_agbd_mg_ha', '—')}",
            f"Total above-ground biomass (Mg): {final.get('total_aboveground_biomass_mg', '—')}",
            f"Mean above-ground carbon (Mg C/ha): {final.get('mean_agc_mg_c_ha', '—')}",
            f"Total above-ground carbon (Mg C): {final.get('total_aboveground_carbon_mg_c', '—')}",
            f"Carbon fraction: {final.get('carbon_fraction', 0.47)}",
        ]
    lines += [
        "Limitation: internal GEDI spatial validation is not independent field validation.",
        "Carbon scope: above-ground living carbon only; roots, deadwood, litter and soil carbon are excluded.",
        "This output is not a certified carbon-credit estimate.",
    ]
    return "\n".join(lines)


def _show_s2_coverage(run, mask_file):
    count_file = run / "processed" / "sentinel2" / "S2_valid_observation_count.tif"
    if not count_file.exists() or not mask_file.exists():
        return
    try:
        with rasterio.open(count_file) as counts_src, rasterio.open(mask_file) as mask_src:
            counts = counts_src.read(1)
            mask = mask_src.read(1) > 0
        if counts.shape != mask.shape or not mask.any():
            st.warning("Sentinel-2 valid-count and mangrove mask grids do not align. Rebuild the mask.")
            return
        one = 100.0 * float(np.mean(counts[mask] >= 1))
        three = 100.0 * float(np.mean(counts[mask] >= 3))
        median = float(np.median(counts[mask]))
        cols = st.columns(3)
        cols[0].metric("Mangrove pixels with ≥1 clear S2", f"{one:.1f}%")
        cols[1].metric("Mangrove pixels with ≥3 clear S2", f"{three:.1f}%")
        cols[2].metric("Median clear S2 observations", f"{median:.0f}")
        if one < 80:
            st.error("Sentinel-2 clear-pixel coverage is below 80%. White/NoData gaps may strongly reduce GEDI matching.")
        elif three < 70:
            st.warning("Most mangrove pixels have data, but fewer than 70% have three clear observations.")
        else:
            st.success("Sentinel-2 clear-pixel coverage is suitable for model testing.")
    except Exception as error:
        st.warning(f"Sentinel-2 coverage diagnostics failed: {error}")


def _reference_raster(run: Path):
    folder = run / "processed" / "sentinel2"
    excluded = ("mask", "valid", "cloud", "rgb")
    files = [
        path for path in sorted(folder.glob("*.tif"))
        if not any(token in path.stem.lower() for token in excluded)
    ]
    preferred = [p for p in files if "ndvi" in p.stem.lower()]
    return (preferred or files or [None])[0]


def _save_custom_mask(upload, run: Path):
    target_dir = run / "input" / "mangrove_source"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    target = target_dir / Path(upload.name).name
    target.write_bytes(upload.getbuffer())
    suffix = target.suffix.lower()
    if suffix == ".zip":
        extract = target_dir / "vector"
        extract.mkdir()
        with zipfile.ZipFile(target) as archive:
            resolved_extract = extract.resolve()
            for member in archive.infolist():
                resolved_target = (resolved_extract / member.filename).resolve()
                if resolved_target != resolved_extract and resolved_extract not in resolved_target.parents:
                    raise ValueError("ZIP contains an unsafe path.")
            archive.extractall(extract)
        shp = next(extract.rglob("*.shp"), None)
        if shp is None:
            raise ValueError("The ZIP does not contain a shapefile.")
        return shp, "vector"
    if suffix in {".geojson", ".json", ".gpkg", ".shp"}:
        return target, "vector"
    if suffix in {".tif", ".tiff"}:
        return target, "raster"
    raise ValueError("Use GeoTIFF, GeoJSON, GeoPackage or zipped shapefile.")


def _show_mask(run, mask_file, report):
    cols = st.columns(3)
    cols[0].metric("Mangrove pixels", f"{report.get('mangrove_pixels', 0):,}")
    cols[1].metric("Mangrove area", f"{report.get('mangrove_area_ha', 0):,.1f} ha")
    cols[2].metric("Mask source", "GMW 2020" if "gmw" in report.get("source", "").lower() else "Custom")
    try:
        layer = raster_view.prepare(mask_file, "Mangrove mask (1=mangrove)")
        raster_view.plot(layer, key=f"mangrove_mask_{run.name}")
    except Exception as error:
        st.warning(f"Mask was created but preview failed: {error}")


def render(run):
    st.subheader("Mangrove-only three-model experiment")
    st.caption(
        "The same mangrove GEDI footprints and the same spatial validation split are used for all models. "
        "Only the sensor predictors change."
    )
    if run is None:
        st.info("Save or select a study first.")
        return
    reference = _reference_raster(run)
    if reference is None:
        st.info("Process Sentinel-2 first. Its grid is the reference for the mangrove mask.")
        return
    aoi_file = run / "input" / "aoi_wgs84.geojson"
    mask_file = run / "processed" / "mangrove" / "mangrove_mask.tif"

    st.markdown("### A · Mangrove mask")
    source_choice = st.radio(
        "Mask source",
        ["Global Mangrove Watch 2020 — recommended", "Upload a locally validated mask"],
        horizontal=True,
    )
    custom = None
    if source_choice.startswith("Upload"):
        custom = st.file_uploader(
            "Mangrove mask", type=["tif", "tiff", "geojson", "json", "gpkg", "zip"],
            help="GeoTIFF values must use 1 (or any positive value) for mangrove. ZIP must contain a shapefile.",
            key="mangrove_mask_upload",
        )
    if st.button("Prepare mangrove mask", type="primary", key="prepare_mangrove_mask"):
        try:
            if source_choice.startswith("Global"):
                progress = st.progress(0.0, text="Downloading GMW once; the official archive is about 180 MB.")
                source = download_gmw(ROOT / "data_cache", lambda x: progress.progress(x))
                source_kind = "raster"
            else:
                if custom is None:
                    raise ValueError("Upload a mask first.")
                source, source_kind = _save_custom_mask(custom, run)
            report = build_aligned_mask(reference, aoi_file, source, mask_file, source_kind)
            st.session_state[f"mask_report_{run.name}"] = report
            st.success("Mangrove mask prepared and aligned to Sentinel-2.")
        except Exception as error:
            st.error(str(error))
    report_file = mask_file.parent / "mangrove_mask_report.json"
    if mask_file.exists() and report_file.exists():
        report = json.loads(report_file.read_text(encoding="utf-8"))
        _show_mask(run, mask_file, report)
        _show_s2_coverage(run, mask_file)
        st.caption(f"Default dataset citation: Global Mangrove Watch 2020 v4.0.19, DOI {GMW_DOI}.")

    st.markdown("### B · Create one common mangrove training table")
    st.caption(
        "Use quality-filtered GEDI L4A footprints. Dates here filter GEDI; they do not change "
        "already processed satellite rasters. Confirm the sensor dates shown in their reports."
    )
    uploaded_csv = st.file_uploader("Upload filtered GEDI L4A CSV", type=["csv"], key="comparison_gedi_csv")
    path_text = st.text_input("Or enter its full CSV path", key="comparison_gedi_path")
    c1, c2 = st.columns(2)
    start = c1.date_input("GEDI matching start", date.today() - timedelta(days=365), key="comparison_start")
    end = c2.date_input("GEDI matching end", date.today(), key="comparison_end")
    region = st.text_input("Training-region name", value=run.name, key="comparison_region")
    threshold = st.slider("Minimum mangrove coverage within a GEDI footprint", .50, 1.0, .60, .05)
    common_csv = run / "training" / "mangrove_common_samples.csv"
    audit_csv = run / "training" / "mangrove_sample_audit.csv"
    if st.button("Build common samples for all three models", type="primary", key="build_common_samples"):
        try:
            if not mask_file.exists():
                raise ValueError("Prepare the mangrove mask first.")
            if uploaded_csv is not None:
                source_csv = run / "input" / "gedi_filtered.csv"
                source_csv.write_bytes(uploaded_csv.getbuffer())
            elif path_text.strip():
                source_csv = Path(path_text.strip().strip('"'))
            else:
                raise ValueError("Upload or enter the filtered GEDI CSV.")
            table, build_report = build_training_table(
                source_csv, run,
                ["Sentinel-2", "Sentinel-1", "ALOS-2"],
                region, start, end, minimum_coverage=.80,
            )
            audited, mask_report = add_mangrove_coverage(table, mask_file, threshold)
            common_csv.parent.mkdir(parents=True, exist_ok=True)
            audited.to_csv(audit_csv, index=False)
            ready = audited[audited["sample_ready"] & audited["mangrove_sample"]].copy()
            ready.to_csv(common_csv, index=False)
            report = {**build_report, **mask_report, "common_ready_samples": int(len(ready))}
            (common_csv.parent / "common_sample_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            st.success(f"{len(ready):,} common mangrove samples are ready for all three models.")
        except Exception as error:
            st.error(str(error))
    if common_csv.exists():
        sample_report = json.loads((common_csv.parent / "common_sample_report.json").read_text(encoding="utf-8"))
        cols = st.columns(4)
        cols[0].metric("Input GEDI", f"{sample_report['input_csv_points']:,}")
        cols[1].metric("Mangrove GEDI", f"{sample_report['accepted_mangrove_samples']:,}")
        cols[2].metric("Common complete", f"{sample_report['common_ready_samples']:,}")
        cols[3].metric("Predictors", f"{sample_report['predictor_count']:,}")
        input_gedi = max(int(sample_report.get("input_csv_points", 0)), 1)
        complete = int(sample_report.get("common_ready_samples", 0))
        st.progress(min(complete / input_gedi, 1.0), text=f"Complete training coverage: {100 * complete / input_gedi:.1f}% of input GEDI")
        if complete < 300:
            st.warning("Fewer than 300 complete samples remain. Results may be unstable across spatial folds.")

        try:
            ready = pd.read_csv(common_csv)
            dates = pd.to_datetime(ready.get("observation_time"), utc=True, errors="coerce")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("GEDI median AGBD", f"{ready.agbd_mg_ha.median():.2f} Mg/ha")
            c2.metric("GEDI 95th percentile", f"{ready.agbd_mg_ha.quantile(.95):.2f} Mg/ha")
            c3.metric("GEDI maximum", f"{ready.agbd_mg_ha.max():.2f} Mg/ha")
            c4.metric("Median GEDI SE", f"{ready.agbd_se_mg_ha.median():.2f} Mg/ha")
            if dates.notna().any():
                st.caption(f"GEDI observations used: {dates.min().date()} to {dates.max().date()}")
            st.caption("Models use bounded inverse-uncertainty weights; uncertain GEDI labels are down-weighted, not silently deleted.")
        except Exception as error:
            st.warning(f"Training diagnostics could not be displayed: {error}")

    st.markdown("### C · Train and spatially validate three models")
    block_size = st.number_input("Spatial validation block size (km)", 1.0, 25.0, 5.0, .5)
    n_folds = st.number_input("Spatial cross-validation folds", 3, 10, 5, 1)
    comparison_dir = run / "models" / "three_model_comparison"
    if st.button("Train and compare three models", type="primary", disabled=not common_csv.exists()):
        try:
            with st.spinner("Running spatial cross-validation and refitting final models..."):
                comparison, summary = train_comparison(common_csv, comparison_dir, block_size, n_folds)
            st.success(f"Comparison complete. Lowest eligible spatial RMSE: {summary['best_model_by_rmse']}")
        except Exception as error:
            st.error(str(error))
    comparison_file = comparison_dir / "model_comparison.csv"
    summary_file = comparison_dir / "comparison_summary.json"
    if comparison_file.exists() and summary_file.exists():
        comparison = pd.read_csv(comparison_file)
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        pretty_columns = [
            "model", "validation_count", "spatial_folds", "r2", "mae_mg_ha",
            "rmse_mg_ha", "bias_mg_ha", "fold_std_rmse_mg_ha",
            "baseline_rmse_mg_ha", "beats_mean_baseline", "validation_status",
        ]
        pretty = comparison[[c for c in pretty_columns if c in comparison]].copy()
        st.dataframe(pretty.style.format(precision=3), use_container_width=True)
        st.image(str(comparison_dir / "three_model_validation.png"), caption="Out-of-fold predictions. Each GEDI observation was predicted by a model that did not train on its spatial fold.")
        _status_box(summary)
        st.write("Lowest-RMSE model:", summary["best_model_by_rmse"])
        st.warning(summary["warning"])
        st.download_button("Download model comparison CSV", comparison.to_csv(index=False), "model_comparison.csv", key="comparison_csv_download")

        st.markdown("### C2 · Compare three machine-learning algorithms")
        st.caption(
            "Random Forest, Extra Trees and Histogram Gradient Boosting use the same complete samples, "
            "all 33 available predictors and exactly the same spatial folds. This is an algorithm comparison, "
            "not another sensor comparison."
        )
        algorithm_dir = run / "models" / "algorithm_comparison"
        if st.button("Train and compare three ML algorithms", type="primary", key="compare_algorithms"):
            try:
                with st.spinner("Running the same spatial cross-validation for three algorithms..."):
                    algorithm_table, algorithm_summary = compare_algorithms(
                        common_csv, algorithm_dir, block_size, n_folds
                    )
                st.success(
                    "Algorithm comparison complete. Lowest eligible spatial RMSE: "
                    + algorithm_summary["best_algorithm_label"]
                )
            except Exception as error:
                st.error(str(error))

        algorithm_file = algorithm_dir / "algorithm_comparison.csv"
        algorithm_summary_file = algorithm_dir / "algorithm_summary.json"
        if algorithm_file.exists() and algorithm_summary_file.exists():
            algorithm_table = pd.read_csv(algorithm_file)
            algorithm_summary = json.loads(algorithm_summary_file.read_text(encoding="utf-8"))
            algorithm_columns = [
                "algorithm_label", "sample_count", "predictor_count", "spatial_folds",
                "r2", "mae_mg_ha", "rmse_mg_ha", "bias_mg_ha",
                "fold_std_rmse_mg_ha", "baseline_rmse_mg_ha",
                "beats_mean_baseline", "validation_status", "rank_rmse",
            ]
            display_algorithms = algorithm_table[
                [column for column in algorithm_columns if column in algorithm_table]
            ].copy()
            st.dataframe(display_algorithms.style.format(precision=3), width="stretch")
            validation_plot = algorithm_dir / "algorithm_validation.png"
            if validation_plot.exists():
                st.image(
                    str(validation_plot),
                    caption=(
                        "Out-of-fold spatial predictions. Every GEDI observation is predicted "
                        "by a model that did not train on its spatial fold."
                    ),
                )
            best_row = algorithm_table.iloc[0]
            metrics_columns = st.columns(4)
            metrics_columns[0].metric("Best algorithm", algorithm_summary["best_algorithm_label"])
            metrics_columns[1].metric("Spatial R²", f"{best_row.r2:.3f}")
            metrics_columns[2].metric("RMSE", f"{best_row.rmse_mg_ha:.2f} Mg/ha")
            metrics_columns[3].metric("MAE", f"{best_row.mae_mg_ha:.2f} Mg/ha")
            if best_row.validation_status == "failed_validation":
                st.error("None of the algorithms produced a positive spatial R² while beating the spatial mean baseline.")
            else:
                st.warning(
                    "The selected algorithm passed internal spatial validation only. "
                    "Independent field plots are still required."
                )
            st.download_button(
                "Download algorithm comparison CSV",
                algorithm_table.to_csv(index=False),
                "algorithm_comparison.csv",
                key="algorithm_comparison_download",
            )

        st.markdown("### D · Create the selected mangrove biomass and carbon map")
        model_names = list(MODEL_DEFINITIONS)
        default_index = model_names.index(summary["best_model_by_rmse"])
        selected_model = st.selectbox("Model", model_names, index=default_index)
        selected_metrics_file = comparison_dir / selected_model / "metrics.json"
        selected_metrics = (
            json.loads(selected_metrics_file.read_text(encoding="utf-8"))
            if selected_metrics_file.exists() else {}
        )
        selected_summary = {
            **summary,
            "best_model_by_rmse": selected_model,
            "validation_status": selected_metrics.get("validation_status", "unknown"),
            "status_message": selected_metrics.get("status_message", "Retrain this model with the scientific validation workflow."),
            "map_recommended": bool(selected_metrics.get("map_recommended", False)),
        }
        _status_box(selected_summary)
        if selected_metrics:
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Spatial R²", f"{selected_metrics.get('r2', float('nan')):.3f}")
            q2.metric("Validation RMSE", f"{selected_metrics.get('rmse_mg_ha', float('nan')):.2f} Mg/ha")
            q3.metric("Validation MAE", f"{selected_metrics.get('mae_mg_ha', float('nan')):.2f} Mg/ha")
            q4.metric("Validation bias", f"{selected_metrics.get('bias_mg_ha', float('nan')):.2f} Mg/ha")
            st.caption(
                "RMSE is a model-level validation error indicator, not a pixel-level confidence interval. "
                "Independent field plots are still required for external validation."
            )
        resolution = st.selectbox("Output resolution", [25.0, 50.0, 100.0], index=0, format_func=lambda x: f"{x:.0f} m")
        st.caption("25 m is recommended for calculation. Coarser maps use fractional mangrove area to reduce edge-area bias.")
        # Legacy summaries have no scientific gate and must be retrained.
        failed = not bool(selected_summary.get("map_recommended", False))
        acknowledge = False
        if failed:
            acknowledge = st.checkbox(
                "I understand that this model failed validation and the map is for pipeline demonstration only.",
                key="ack_failed_model",
            )
        if st.button("Create mangrove biomass and carbon maps", type="primary", disabled=failed and not acknowledge):
            try:
                model_dir = comparison_dir / selected_model
                raw_dir = run / "prediction" / selected_model / "raw"
                final_dir = run / "prediction" / selected_model / "mangrove_only"
                with st.spinner("Predicting biomass and applying the mangrove mask..."):
                    prediction = predict_biomass(
                        run, model_dir / "random_forest_l4a.joblib",
                        model_dir / "feature_list_l4a.json", raw_dir,
                        output_resolution_m=resolution,
                        support_size_m=25.0,
                        min_valid_feature_fraction=1.0,
                    )
                    final = apply_mask_to_prediction(prediction["agbd_file"], mask_file, final_dir)
                st.session_state[f"prediction_summary_{run.name}"] = final
                if failed:
                    st.warning("Demonstration map created from a model that failed spatial validation.")
                else:
                    st.success("Experimental mangrove biomass and above-ground carbon maps created.")
            except Exception as error:
                st.error(str(error))
        summary_candidate = (
            run / "prediction" / selected_model / "mangrove_only" /
            "mangrove_prediction_summary.json"
        )
        for candidate in [summary_candidate] if summary_candidate.exists() else []:
            final = json.loads(candidate.read_text(encoding="utf-8"))
            st.markdown("### E · Scientific result summary")
            _status_box(selected_summary)
            cols = st.columns(4)
            cols[0].metric("Mean AGBD", f"{final['mean_agbd_mg_ha']:.2f} Mg/ha" if final['mean_agbd_mg_ha'] is not None else "—")
            cols[1].metric("Median AGBD", f"{final['median_agbd_mg_ha']:.2f} Mg/ha" if final['median_agbd_mg_ha'] is not None else "—")
            cols[2].metric("AGBD 5–95%", f"{final.get('p05_agbd_mg_ha', 0):.1f}–{final.get('p95_agbd_mg_ha', 0):.1f} Mg/ha")
            cols[3].metric("Fractional mangrove area", f"{final['mangrove_area_ha']:,.1f} ha")
            cols = st.columns(4)
            cols[0].metric("Total above-ground biomass", f"{final['total_aboveground_biomass_mg']:,.1f} Mg" if final.get('total_aboveground_biomass_mg') is not None else "—")
            cols[1].metric("Mean AG carbon", f"{final['mean_agc_mg_c_ha']:.2f} Mg C/ha" if final['mean_agc_mg_c_ha'] is not None else "—")
            cols[2].metric("Total AG carbon", f"{final['total_aboveground_carbon_mg_c']:,.1f} Mg C" if final['total_aboveground_carbon_mg_c'] is not None else "—")
            cols[3].metric("Carbon fraction", f"{final.get('carbon_fraction', .47):.2f}")
            st.caption(final["warning"])
            st.caption(final.get("area_method", "Mangrove area method was not recorded."))
            report_text = _client_report(run, selected_summary, final)
            st.download_button(
                "Download scientific client summary",
                report_text,
                file_name=f"{run.name}_scientific_summary.txt",
                mime="text/plain",
                key=f"scientific_summary_{run.name}_{selected_model}",
            )
            break
