"""Dashboard panel for inspecting GEDI L4A Version 3."""

from pathlib import Path

import folium
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

from gedi.l4a_reader import read_l4a, make_geojson
from datetime import date

from gedi.download_l4a import search_gedi, download_gedi

def render(run=None):
    st.subheader("GEDI L4A reference observations")
    
    st.markdown("### Automatically find GEDI for this AOI")

    if run is None:
        st.info("Select or save a study before searching for GEDI.")
    else:
        aoi_path = run / "input" / "aoi_wgs84.geojson"

        if not aoi_path.exists():
            st.info("The selected study does not have a saved AOI.")
        else:
            date_columns = st.columns(2)

            search_start = date_columns[0].date_input(
                "GEDI search start",
                value=date(2019, 4, 1),
                key="gedi_search_start",
            )

            search_end = date_columns[1].date_input(
                "GEDI search end",
                value=date.today(),
                key="gedi_search_end",
            )

            if st.button(
                "Search GEDI for saved AOI",
                type="primary",
                key="search_gedi_aoi",
            ):
                try:
                    with st.spinner(
                        "Searching NASA Earthdata for overlapping GEDI granules..."
                    ):
                        results = search_gedi(
                            aoi_path,
                            search_start,
                            search_end,
                        )

                    st.session_state["gedi_search_results"] = results

                    if results:
                        st.success(
                            f"{len(results)} overlapping GEDI granules found."
                        )
                    else:
                        st.warning(
                            "No GEDI L4A granules overlap this AOI "
                            "and date range."
                        )

                except Exception as error:
                    st.error(str(error))

            results = st.session_state.get(
                "gedi_search_results",
                [],
            )

            if results:
                output_folder = (
                    Path(r"D:\GEDI\L4A_V3")
                    / run.name
                )

                st.write(
                    f"Download folder: `{output_folder}`"
                )

                names = []

                for result in results:
                    try:
                        names.append(
                            result["umm"]["GranuleUR"]
                        )
                    except Exception:
                        names.append(str(result))

                st.dataframe(
                    {"GEDI granule": names},
                    use_container_width=True,
                )

                if st.button(
                    "Download overlapping GEDI files",
                    key="download_gedi_aoi",
                ):
                    try:
                        with st.spinner(
                            "Downloading GEDI from NASA Earthdata..."
                        ):
                            downloaded = download_gedi(
                                results,
                                output_folder,
                            )

                        st.success(
                            f"{len(downloaded)} GEDI files downloaded."
                        )

                        st.session_state[
                            "gedi_l4a_folder"
                        ] = str(output_folder)

                        st.rerun()

                    except Exception as error:
                        st.error(str(error))
    st.caption(
        "Load a downloaded Version 3 HDF5 file. "
        "These are footprint biomass estimates, not a continuous "
        "biomass map or a trained model."
    )

    folder_text = st.text_input(
        "GEDI L4A folder",
        value=r"D:\GEDI\L4A_V3",
        key="gedi_l4a_folder",
    )

    folder = Path(folder_text)

    if not folder.is_dir():
        st.info("Enter the folder containing your GEDI .h5 files.")
        return

    files = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.name.upper().endswith("_V003.H5")
    )

    if not files:
        st.warning("No Version 3 files ending in _V003.h5 found.")
        return

    selected_files = st.multiselect(
        "GEDI files",
        files,
        default=[files[0]],
        format_func=lambda path: path.name,
        key="gedi_l4a_file",
        help="Select several overlapping granules when one orbit provides too few footprints.",
    )

    if not selected_files:
        st.info("Select at least one GEDI file.")
        return

    aoi_file = (
        run / "input" / "aoi_wgs84.geojson"
        if run is not None
        else None
    )

    aoi_available = (
        aoi_file is not None and aoi_file.exists()
    )

    use_aoi = st.checkbox(
        "Keep only footprint centres inside the saved AOI",
        value=bool(aoi_available),
        disabled=not aoi_available,
        key="gedi_use_aoi",
    )

    if use_aoi and not aoi_available:
        st.warning(
            "No saved AOI is available for this study. "
            "Leave this option unchecked or save an AOI first."
        )
        use_aoi = False

    exclude_extrapolated = st.checkbox(
        "Exclude predictions outside GEDI model training limits",
        value=True,
        key="gedi_exclude_extrapolated",
        help=(
            "Recommended for scientific training. It can reduce the available observations."
        ),
    )

    # Hide old results when the input file or filters change.
    aoi_signature = None

    if use_aoi and aoi_file is not None and aoi_file.exists():
        aoi_signature = aoi_file.stat().st_mtime_ns

    signature = (
        tuple(str(path) for path in selected_files),
        tuple(path.stat().st_mtime_ns for path in selected_files),
        str(aoi_file) if use_aoi and aoi_file is not None else None,
        aoi_signature,
        exclude_extrapolated,
    )

    if st.button(
        "Read and filter GEDI",
        key="read_gedi_l4a",
        type="primary",
    ):
        st.session_state.pop("gedi_l4a_result", None)

        try:
            with st.spinner("Reading GEDI beams and applying filters..."):
                tables = []
                reports = []
                for selected in selected_files:
                    one_table, one_report = read_l4a(
                        selected,
                        aoi_file=aoi_file if use_aoi else None,
                        exclude_extrapolated=exclude_extrapolated,
                    )
                    tables.append(one_table)
                    reports.append(one_report)
                import pandas as pd
                table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
                if not table.empty and "shot_number" in table:
                    table = table.drop_duplicates("shot_number").reset_index(drop=True)
                report = {
                    "source_files": [path.name for path in selected_files],
                    "shots_read": sum(item["shots_read"] for item in reports),
                    "passed_selected_quality_filters": sum(item["passed_selected_quality_filters"] for item in reports),
                    "selected_points": len(table),
                    "aoi_filter_applied": use_aoi,
                    "extrapolated_predictions_excluded": exclude_extrapolated,
                }

            st.session_state["gedi_l4a_result"] = {
                "signature": signature,
                "table": table,
                "report": report,
            }
            
            # Share a bounded preview with the AOI drawing map.
            if not table.empty:
                map_points = (
                    table.sample(n=2000, random_state=42)
                    if len(table) > 2000
                    else table
                )

                st.session_state["gedi_map_points"] = (
                    map_points[
                        [
                            "latitude",
                            "longitude",
                            "agbd_mg_ha",
                            "agbd_se_mg_ha",
                        ]
                    ].to_dict("records")
                )

                st.session_state["gedi_map_total"] = len(table)

                st.success(
                    "GEDI points are ready. Open Explore & draw "
                    "and click Refresh processed imagery."
                )
            else:
                st.session_state["gedi_map_points"] = []
                st.session_state["gedi_map_total"] = 0

        except Exception as error:
            st.error(str(error))

    result = st.session_state.get("gedi_l4a_result")

    if result is None or result["signature"] != signature:
        return

    table = result["table"]
    report = result["report"]

    columns = st.columns(3)
    columns[0].metric("Shots read", f"{report['shots_read']:,}")
    columns[1].metric(
        "Passed filters",
        f"{report['passed_selected_quality_filters']:,}",
    )
    columns[2].metric(
        "Selected points",
        f"{report['selected_points']:,}",
    )

    if table.empty:
        st.warning(
            "No observations passed the selected filters and area "
            "selection. This does not mean the area has zero biomass."
        )
        return

    st.write(
        "Observation period:",
        table["observation_time"].min(),
        "to",
        table["observation_time"].max(),
    )

    # Limit only the map display, not the exported dataset.
    displayed = (
        table.sample(n=2000, random_state=42)
        if len(table) > 2000
        else table
    )

    low = float(displayed["agbd_mg_ha"].min())
    high = float(displayed["agbd_mg_ha"].max())

    if high <= low:
        high = low + 1

    colour = LinearColormap(
        ["#ffffcc", "#78c679", "#006837"],
        vmin=low,
        vmax=high,
        caption="GEDI aboveground biomass density (Mg/ha)",
    )

    map_view = folium.Map(
        location=[
            float(displayed["latitude"].mean()),
            float(displayed["longitude"].mean()),
        ],
        zoom_start=5,
        tiles=None,
    )

    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics",
        name="Satellite basemap",
    ).add_to(map_view)

    for row in displayed.itertuples(index=False):
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=3,
            color=colour(row.agbd_mg_ha),
            weight=1,
            fill=True,
            fill_color=colour(row.agbd_mg_ha),
            fill_opacity=0.8,
            tooltip=(
                f"Biomass: {row.agbd_mg_ha:.2f} Mg/ha | "
                f"SE: {row.agbd_se_mg_ha:.2f} Mg/ha | "
                f"{row.observation_time}"
            ),
        ).add_to(map_view)

    map_view.fit_bounds(
        [
            [
                float(displayed["latitude"].min()),
                float(displayed["longitude"].min()),
            ],
            [
                float(displayed["latitude"].max()),
                float(displayed["longitude"].max()),
            ],
        ],
        max_zoom=14,
    )

    colour.add_to(map_view)

    st_folium(
        map_view,
        height=500,
        use_container_width=True,
        key="gedi_l4a_map",
        returned_objects=[],
    )

    st.caption(
        f"Showing {len(displayed):,} of {len(table):,} selected points. "
        "Markers indicate footprint centres; their screen size is "
        "not the actual footprint diameter."
    )

    st.dataframe(
        table.head(100),
        use_container_width=True,
    )

    st.download_button(
        "Download selected points — CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="gedi_l4a_merged_filtered.csv",
        mime="text/csv",
        key="gedi_l4a_csv",
    )

    st.download_button(
        "Download selected points — GeoJSON",
        data=make_geojson(table),
        file_name="gedi_l4a_merged_filtered.geojson",
        mime="application/geo+json",
        key="gedi_l4a_geojson",
    )

    st.caption(
        "Exports contain all selected points, not just the map sample. "
        "Quality-filtered observations are not automatically forest-only "
        "labels or ready-to-train satellite samples."
    )
