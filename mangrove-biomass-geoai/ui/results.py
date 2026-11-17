"""Preview actual output rasters with bounded memory."""
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import streamlit as st

def render(run):
    files = sorted((run/"processed").rglob("*.tif")) + sorted((run/"prediction").rglob("*.tif"))
    if not files:
        st.info("No raster outputs yet.")
        return
    path = st.selectbox("Layer", files, format_func=lambda p: str(p.relative_to(run)))
    with rasterio.open(path) as src:
        scale = min(1, 800/max(src.width, src.height))
        data = src.read(1, out_shape=(max(1,int(src.height*scale)),max(1,int(src.width*scale))), masked=True)
    values = data.compressed()
    values = values[np.isfinite(values)]
    if not len(values):
        st.warning("No valid pixels.")
        return
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    lo, hi = np.percentile(values,[2,98])
    im = axes[0].imshow(data, vmin=lo, vmax=hi)
    fig.colorbar(im, ax=axes[0])
    axes[1].hist(values,bins=40)
    axes[0].set_title("Raster preview")
    axes[1].set_title("Preview histogram")
    st.pyplot(fig)
    plt.close(fig)
    st.write({"preview_mean":float(values.mean()),"preview_min":float(values.min()),
              "preview_max":float(values.max())})
    st.caption("Downsampled preview statistics; source units. These are not AOI biomass totals.")
    st.code(str(path))
