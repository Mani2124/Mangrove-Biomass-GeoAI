"""Create a small Sentinel-2 true-colour preview for the map."""

import numpy as np
import rasterio

from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds


def load_rgb(run):
    folder = run / "processed" / "sentinel2"

    # True colour: red, green, blue.
    paths = [
        folder / "S2_Red_median.tif",
        folder / "S2_Green_median.tif",
        folder / "S2_Blue_median.tif",
    ]

    if not all(path.exists() for path in paths):
        return None

    with rasterio.open(paths[0]) as source:
        bounds = transform_bounds(
            source.crs,
            "EPSG:4326",
            *source.bounds,
            densify_pts=21,
        )

    west, south, east, north = bounds

    # A small shared geographic grid keeps the map responsive.
    span = max(east - west, north - south)
    width = max(1, round(700 * (east - west) / span))
    height = max(1, round(700 * (north - south) / span))

    transform = from_bounds(
        west, south, east, north, width, height
    )

    channels = []

    for path in paths:
        with rasterio.open(path) as source:
            with WarpedVRT(
                source,
                crs="EPSG:4326",
                transform=transform,
                width=width,
                height=height,
                dtype="float32",
                nodata=np.nan,
                resampling=Resampling.nearest,
            ) as projected:
                channel = projected.read(1, masked=True)
                channels.append(channel.filled(np.nan))

    rgb = np.stack(channels, axis=-1)
    valid = np.isfinite(rgb).all(axis=-1)

    if not valid.any():
        return None

    # Shared display stretch preserves the relationship between bands.
    low, high = np.percentile(rgb[valid], [2, 98])

    if high <= low:
        high = low + 1

    rgb = np.clip((rgb - low) / (high - low), 0, 1)
    rgb = np.nan_to_num(rgb)

    rgba = np.dstack([rgb, valid.astype(float)])

    return {
        "image": rgba,
        "bounds": [[south, west], [north, east]],
    }