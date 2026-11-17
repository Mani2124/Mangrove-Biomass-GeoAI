"""AOI polygons and POIs are distinct; a point never replaces a polygon."""
import folium
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

def render():
    m = folium.Map(location=[15,80], zoom_start=4, tiles=None)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics", name="Satellite basemap").add_to(m)
    Draw(draw_options={"polyline":False,"circle":False,"circlemarker":False}).add_to(m)
    data = st_folium(m, height=500, use_container_width=True, key="map")
    if data.get("all_drawings") is not None:
        st.session_state.drawings = data["all_drawings"]
    return st.session_state.get("drawings",[])
