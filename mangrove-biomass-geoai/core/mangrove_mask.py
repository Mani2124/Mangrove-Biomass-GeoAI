"""Create and apply a reproducible mangrove-only mask.

Primary source: Global Mangrove Watch 2020 v4.0.19 (10 m).
The global archive is cached locally after the first download. Custom raster
and vector masks are also supported for newer locally validated boundaries.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer
from rasterio.features import geometry_mask, geometry_window, rasterize
from rasterio.warp import reproject, Resampling, transform_bounds
from shapely.geometry import Point, box, mapping, shape
from shapely.ops import transform as shapely_transform

GMW_URL = (
    "https://zenodo.org/records/12756047/files/"
    "gmw_mng_2020_v4019_gtiff.zip?download=1"
)
GMW_DOI = "10.5281/zenodo.12756047"


def download_gmw(cache_dir: Path, progress=None) -> Path:
    """Download/extract the official GMW 2020 raster archive once."""
    cache_dir = Path(cache_dir)
    extract_dir = cache_dir / "gmw_2020_v4019"
    marker = extract_dir / ".complete"
    if marker.exists() and list(extract_dir.rglob("*.tif")):
        return extract_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "gmw_mng_2020_v4019_gtiff.zip"
    if not archive.exists():
        partial = archive.with_suffix(".part")
        with requests.get(GMW_URL, stream=True, timeout=(60, 900)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            done = 0
            with partial.open("wb") as target:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    target.write(chunk)
                    done += len(chunk)
                    if progress and total:
                        progress(min(done / total, 1.0))
        partial.replace(archive)

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    with zipfile.ZipFile(archive) as source:
        source.extractall(extract_dir)
    marker.write_text(json.dumps({"source": GMW_URL, "doi": GMW_DOI}))
    return extract_dir


def _aoi_geometry(aoi_file: Path):
    obj = json.loads(Path(aoi_file).read_text(encoding="utf-8"))
    geoms = [shape(feature["geometry"]) for feature in obj["features"]]
    return gpd.GeoSeries(geoms, crs=4326).union_all()


def _intersecting_rasters(folder: Path, aoi_wgs84) -> list[Path]:
    matches = []
    for path in Path(folder).rglob("*.tif"):
        try:
            with rasterio.open(path) as src:
                bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            if box(*bounds).intersects(aoi_wgs84):
                matches.append(path)
        except Exception:
            continue
    if not matches:
        raise ValueError("No mangrove-mask raster tile overlaps the saved AOI.")
    return matches


def _vector_geometries(path: Path):
    frame = gpd.read_file(path)
    if frame.empty:
        raise ValueError("The custom mangrove vector is empty.")
    if frame.crs is None:
        raise ValueError("The custom mangrove vector has no CRS.")
    return frame


def build_aligned_mask(
    reference_raster: Path,
    aoi_file: Path,
    source: Path,
    output_file: Path,
    source_kind: str = "raster",
) -> dict:
    """Align a categorical mangrove layer to the Sentinel-2 reference grid."""
    reference_raster, source, output_file = map(Path, (reference_raster, source, output_file))
    aoi_wgs84 = _aoi_geometry(aoi_file)
    with rasterio.open(reference_raster) as ref:
        profile = ref.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0, compress="LZW")
        mask = np.zeros((ref.height, ref.width), dtype="uint8")

        if source_kind == "vector":
            frame = _vector_geometries(source).to_crs(ref.crs)
            mask = rasterize(
                [(mapping(geom), 1) for geom in frame.geometry if geom is not None and not geom.is_empty],
                out_shape=mask.shape,
                transform=ref.transform,
                fill=0,
                dtype="uint8",
            )
        else:
            raster_paths = _intersecting_rasters(source, aoi_wgs84) if source.is_dir() else [source]
            for raster_path in raster_paths:
                with rasterio.open(raster_path) as src:
                    temporary = np.zeros(mask.shape, dtype="uint8")
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=temporary,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref.transform,
                        dst_crs=ref.crs,
                        src_nodata=src.nodata,
                        dst_nodata=0,
                        resampling=Resampling.nearest,
                    )
                    mask |= (temporary > 0).astype("uint8")

        transformer = Transformer.from_crs("EPSG:4326", ref.crs, always_xy=True)
        aoi_projected = shapely_transform(transformer.transform, aoi_wgs84)
        inside_aoi = geometry_mask(
            [mapping(aoi_projected)], out_shape=mask.shape,
            transform=ref.transform, invert=True,
        )
        mask = ((mask == 1) & inside_aoi).astype("uint8")
        pixel_area_ha = abs(ref.transform.a * ref.transform.e) / 10000.0

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(mask, 1)
        dst.set_band_description(1, "Mangrove mask: 1=mangrove, 0=excluded")

    report = {
        "mask_file": str(output_file),
        "reference_raster": str(reference_raster),
        "source": str(source),
        "source_kind": source_kind,
        "mangrove_pixels": int(mask.sum()),
        "mangrove_area_ha": float(mask.sum() * pixel_area_ha),
        "pixel_area_ha": float(pixel_area_ha),
    }
    (output_file.parent / "mangrove_mask_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def add_mangrove_coverage(table: pd.DataFrame, mask_file: Path, threshold=0.80):
    """Calculate the fraction of each 25 m GEDI footprint mapped as mangrove."""
    result = table.copy()
    coverages = []
    with rasterio.open(mask_file) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        for row in result.itertuples(index=False):
            x, y = transformer.transform(float(row.longitude), float(row.latitude))
            footprint = Point(x, y).buffer(12.5, resolution=16)
            try:
                window = geometry_window(src, [mapping(footprint)])
                values = src.read(1, window=window, masked=True)
                local_transform = src.window_transform(window)
                inside = geometry_mask(
                    [mapping(footprint)], out_shape=values.shape,
                    transform=local_transform, invert=True,
                )
                usable = inside & ~np.ma.getmaskarray(values)
                coverages.append(
                    float(np.mean(np.asarray(values)[usable] == 1))
                    if usable.any() else 0.0
                )
            except Exception:
                coverages.append(0.0)
    result["mangrove_coverage"] = coverages
    result["mangrove_sample"] = result["mangrove_coverage"] >= float(threshold)
    report = {
        "input_samples": int(len(result)),
        "accepted_mangrove_samples": int(result["mangrove_sample"].sum()),
        "rejected_samples": int((~result["mangrove_sample"]).sum()),
        "minimum_mangrove_coverage": float(threshold),
    }
    return result, report


def apply_mask_to_prediction(agbd_file: Path, mask_file: Path, output_dir: Path, carbon_fraction=0.47):
    """Apply a fractional mangrove mask and calculate area-weighted totals."""
    agbd_file, mask_file, output_dir = map(Path, (agbd_file, mask_file, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(agbd_file) as src:
        agbd = src.read(1).astype("float32")
        valid = np.isfinite(agbd) & (agbd != src.nodata if src.nodata is not None else True)
        profile = src.profile.copy()
        with rasterio.open(mask_file) as mask_src:
            # Average resampling preserves partial mangrove coverage when the
            # prediction grid is coarser than the source mask.
            mangrove_fraction = np.zeros(agbd.shape, dtype="float32")
            reproject(rasterio.band(mask_src, 1), mangrove_fraction,
                      src_transform=mask_src.transform, src_crs=mask_src.crs,
                      dst_transform=src.transform, dst_crs=src.crs,
                      resampling=Resampling.average, dst_nodata=0)
        mangrove_fraction = np.clip(mangrove_fraction, 0.0, 1.0)
        valid &= mangrove_fraction > 0
        nodata = -9999.0
        masked_agbd = np.where(valid, np.maximum(agbd, 0), nodata).astype("float32")
        carbon = np.where(valid, masked_agbd * float(carbon_fraction), nodata).astype("float32")
        profile.update(dtype="float32", nodata=nodata, count=1, compress="LZW")
        agbd_out = output_dir / "mangrove_AGBD_Mg_ha.tif"
        carbon_out = output_dir / "mangrove_AGC_Mg_C_ha.tif"
        fraction_out = output_dir / "mangrove_fraction.tif"
        for path, array, description in (
            (agbd_out, masked_agbd, "Mangrove above-ground biomass density Mg/ha"),
            (carbon_out, carbon, "Mangrove above-ground carbon density Mg C/ha"),
        ):
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(array, 1); dst.set_band_description(1, description)
        fraction_profile = profile.copy()
        fraction_profile["nodata"] = 0.0
        with rasterio.open(fraction_out, "w", **fraction_profile) as dst:
            dst.write(mangrove_fraction.astype("float32"), 1)
            dst.set_band_description(1, "Fractional mangrove coverage 0-1")
        pixel_area_ha = abs(src.transform.a * src.transform.e) / 10000.0

    values = masked_agbd[valid]
    fractions = mangrove_fraction[valid]
    effective_area_ha = float(np.sum(fractions) * pixel_area_ha)
    total_biomass = float(np.sum(values * fractions) * pixel_area_ha) if values.size else None
    weighted_mean = (
        float(np.sum(values * fractions) / np.sum(fractions))
        if values.size and np.sum(fractions) > 0 else None
    )
    summary = {
        "mangrove_pixels": int(valid.sum()),
        "mangrove_area_ha": effective_area_ha,
        "mean_agbd_mg_ha": weighted_mean,
        "median_agbd_mg_ha": float(np.median(values)) if values.size else None,
        "p05_agbd_mg_ha": float(np.quantile(values, .05)) if values.size else None,
        "p95_agbd_mg_ha": float(np.quantile(values, .95)) if values.size else None,
        "total_aboveground_biomass_mg": total_biomass,
        "mean_agc_mg_c_ha": float(weighted_mean * carbon_fraction) if weighted_mean is not None else None,
        "total_aboveground_carbon_mg_c": float(total_biomass * carbon_fraction) if total_biomass is not None else None,
        "carbon_fraction": float(carbon_fraction),
        "area_method": "Fractional mangrove coverage; partial coarse-grid cells are area weighted.",
        "warning": "Above-ground living carbon proxy only; excludes roots, deadwood, litter and soil carbon.",
        "agbd_raster": str(agbd_out), "agc_raster": str(carbon_out),
        "mangrove_fraction_raster": str(fraction_out),
    }
    (output_dir / "mangrove_prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
