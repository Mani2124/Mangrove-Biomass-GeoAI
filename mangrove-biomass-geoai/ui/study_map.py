"""Local tile footprints, raster overlay and editable AOI drawings."""
import folium
from folium.plugins import Draw
import streamlit as st
from streamlit_folium import st_folium
from ui.raster_view import add_overlay
from branca.colormap import LinearColormap

def render(
    records,
    layer,
    saved=None,
    rgb=None,
    gedi_points=None,
    zoom_to_gedi=False,
):
    m = folium.Map(location=[16,80],zoom_start=5,tiles=None)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics",name="Satellite basemap").add_to(m)
    group = folium.FeatureGroup(name="Downloaded tile bounds")
    for record in records:
        west,south,east,north = record["bounds"]
        folium.Rectangle([[south,west],[north,east]],color="#ffb84d",weight=2,
                         fill=False,tooltip=f'{record["component"]} • {record["year"]}').add_to(group)
    group.add_to(m)
    
    if layer:
        add_overlay(m,layer)
        west,south,east,north = layer["bounds"]
        m.fit_bounds([[south,west],[north,east]])
    elif records:
        m.fit_bounds([[min(r["bounds"][1] for r in records),min(r["bounds"][0] for r in records)],
                      [max(r["bounds"][3] for r in records),max(r["bounds"][2] for r in records)]])
    if rgb is not None:
        folium.raster_layers.ImageOverlay(
            image=rgb["image"],
            bounds=rgb["bounds"],
            name="Sentinel-2 true colour",
            opacity=1.0,
            mercator_project=True,
            show=True,
        ).add_to(m)

        m.fit_bounds(rgb["bounds"])
    if saved:
        folium.GeoJson(saved,name="Saved AOI",style_function=lambda _:{"color":"#ff4f78","fillOpacity":0.08}).add_to(m)
    Draw(draw_options={"polyline":False,"circle":False,"circlemarker":False},
         edit_options={"edit":True,"remove":True}).add_to(m)
    if gedi_points:
        values = [
            point["agbd_mg_ha"]
            for point in gedi_points
        ]

        low = min(values)
        high = max(values)

        if high <= low:
            high = low + 1

        colours = LinearColormap(
            ["#ffffcc", "#78c679", "#006837"],
            vmin=low,
            vmax=high,
            caption="GEDI reference biomass (Mg/ha)",
        )

        gedi_layer = folium.FeatureGroup(
            name="GEDI reference footprints",
            show=True,
        )

        for point in gedi_points:
            value = point["agbd_mg_ha"]

            folium.CircleMarker(
                location=[
                    point["latitude"],
                    point["longitude"],
                ],
                radius=4,
                color=colours(value),
                weight=1,
                fill=True,
                fill_color=colours(value),
                fill_opacity=0.9,
                tooltip=(
                    f"Biomass: {value:.2f} Mg/ha | "
                    f"SE: {point['agbd_se_mg_ha']:.2f} Mg/ha"
                ),
            ).add_to(gedi_layer)

        gedi_layer.add_to(m)
        colours.add_to(m)

        if zoom_to_gedi:
            m.fit_bounds(
                [
                    [
                        min(p["latitude"] for p in gedi_points),
                        min(p["longitude"] for p in gedi_points),
                    ],
                    [
                        max(p["latitude"] for p in gedi_points),
                        max(p["longitude"] for p in gedi_points),
                    ],
                ],
                max_zoom=14,
            )
    folium.LayerControl().add_to(m)
    result = st_folium(m,height=540,use_container_width=True,key="coverage_map")
    if result.get("all_drawings") is not None:
        st.session_state["drawings"] = result["all_drawings"]
    return st.session_state.get("drawings",[])
