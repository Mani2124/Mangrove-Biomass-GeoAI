"""Read metadata and small, georeferenced previews; never load full mosaics."""
from pathlib import Path
import re
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from shapely.geometry import box, shape
from shapely.ops import unary_union

def inspect_folder(folder):
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise ValueError("Folder not found. Enter the folder on the computer running this app.")
    records, errors = [], []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".tif", ".tiff"):
            continue
        component = re.search(r"(?:^|_)(HH|HV)(?:_|\.)", path.name.upper())
        if not component:
            continue
        try:
            with rasterio.open(path) as src:
                if src.crs is None:
                    raise ValueError("missing CRS")
                bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
                if bounds[2] - bounds[0] > 180:
                    raise ValueError("antimeridian tile not supported in this viewer")
                year = re.search(r"_(20\d{2}|\d{2})_", path.name)
                year = int(year[1]) if year else None
                if year is not None and year < 100:
                    year += 2000
                records.append({"path":str(path.resolve()), "name":path.name,
                                "component":component[1], "year":year, "bounds":list(bounds)})
        except Exception as error:
            errors.append(f"{path.name}: {error}")
    return records, errors

def coverage_message(features, records, year):
    """Bounding-box coverage is only a preliminary test, not valid-data coverage."""
    aoi = unary_union([shape(f["geometry"]) for f in features])
    for component in ("HH","HV"):
        footprints = [box(*r["bounds"]) for r in records
                      if r["component"] == component and r["year"] == year]
        if not footprints or not unary_union(footprints).covers(aoi):
            return False, f"AOI is not fully inside loaded {year} {component} tile bounds."
    return True, "AOI is inside both HH and HV tile bounds. No-data and quality masks still apply."

def preview(path):
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError("Raster has no CRS.")
        with WarpedVRT(src, crs="EPSG:4326", add_alpha=True) as vrt:
            scale = min(1, 700/max(vrt.width,vrt.height))
            size = (max(1,int(vrt.height*scale)),max(1,int(vrt.width*scale)))
            data = vrt.read(1, out_shape=size, masked=True, resampling=Resampling.nearest)
            alpha = vrt.read(vrt.count, out_shape=size, resampling=Resampling.nearest)
            data = np.ma.masked_where((alpha==0) | ~np.isfinite(data),data)
            return data, tuple(vrt.bounds)

def display_values(data, db=False):
    if db:
        valid = np.ma.masked_less_equal(data.astype(float),0)
        return 20*np.ma.log10(valid)-83
    return data
