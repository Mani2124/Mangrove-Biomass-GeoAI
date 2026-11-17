"""Shared colour scale for raster overlay, plot and histogram."""
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import folium
from branca.colormap import LinearColormap
import streamlit as st
from core.rasters import preview, display_values

@st.cache_data(max_entries=12)
def load_preview(path, modified):
    return preview(path)

def prepare(path, unit, db=False):
    from pathlib import Path
    path = Path(path)
    data, bounds = load_preview(str(path),path.stat().st_mtime_ns)
    data = display_values(data,db)
    valid = data.compressed()
    valid = valid[np.isfinite(valid)]
    if not len(valid):
        raise ValueError("No valid raster pixels.")
    low, high = np.percentile(valid,[2,98])
    if high <= low:
        high = low+1
    norm = np.clip((data.filled(low)-low)/(high-low),0,1)
    rgba = colormaps["viridis"](norm)
    rgba[...,3] = (~np.ma.getmaskarray(data)).astype(float)*0.8
    return {"data":data, "bounds":bounds, "low":float(low), "high":float(high),
            "rgba":rgba, "unit":unit, "name":path.stem, "values":valid}

def add_overlay(m, layer):
    west,south,east,north = layer["bounds"]
    folium.raster_layers.ImageOverlay(
        layer["rgba"],bounds=[[south,west],[north,east]],
        name=layer["name"],opacity=0.85,mercator_project=True).add_to(m)
    colors = [colormaps["viridis"](x) for x in np.linspace(0,1,8)]
    LinearColormap(colors,vmin=layer["low"],vmax=layer["high"],
                   caption=layer["unit"]+" • 2–98% display stretch").add_to(m)

def plot(layer, *, key):
    values = layer["values"]
    columns = st.columns(3)
    for col, label, value in zip(columns,["Preview mean","Preview minimum","Preview maximum"],
                                [values.mean(),values.min(),values.max()]):
        col.metric(label,f"{value:,.2f}")
    west,south,east,north = layer["bounds"]
    fig, axes = plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    image = axes[0].imshow(layer["data"],extent=[west,east,south,north],origin="upper",
                           cmap="viridis",vmin=layer["low"],vmax=layer["high"])
    axes[0].set(xlabel="Longitude",ylabel="Latitude",title="Raster image")
    fig.colorbar(image,ax=axes[0],label=layer["unit"])
    axes[1].hist(values,bins=50,color="#217b72")
    axes[1].set(xlabel=layer["unit"],ylabel="Preview pixel count",title="Value distribution")
    st.pyplot(fig)
    buffer = io.BytesIO()
    fig.savefig(buffer,format="png",dpi=160)
    plt.close(fig)
    st.download_button(
    "Download plot image",
    buffer.getvalue(),
    "raster_analysis.png",
    "image/png",
    key=f"{key}_download_plot",
)
    st.caption("Downsampled preview statistics, not full-resolution AOI totals. The display stretch does not alter source files.")
