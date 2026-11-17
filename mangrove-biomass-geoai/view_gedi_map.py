"""Show GEDI L4A reference locations on a satellite map."""

import argparse
import html
import webbrowser
from pathlib import Path

import folium
from branca.colormap import LinearColormap
from folium.plugins import Draw, MousePosition

from gedi.l4a_reader import read_l4a


def main():
    parser = argparse.ArgumentParser(
        description="View GEDI L4A footprint locations."
    )

    parser.add_argument(
        "file",
        help="Path to a GEDI L4A Version 3 .h5 file.",
    )

    args = parser.parse_args()
    source = Path(args.file)

    print("Reading GEDI and applying quality filters...")

    table, report = read_l4a(
        file_path=source,
        aoi_file=None,
        exclude_extrapolated=False,
        max_points=500000,
    )

    print(f"Shots read: {report['shots_read']:,}")
    print(f"Selected observations: {len(table):,}")

    if table.empty:
        print("No observations passed the filters.")
        return

    # Keep the browser responsive.
    # This samples only the map display, not the original file.
    maximum_display = 5000

    displayed = (
        table.sample(
            n=maximum_display,
            random_state=42,
        )
        if len(table) > maximum_display
        else table
    )

    low = float(displayed["agbd_mg_ha"].min())
    high = float(displayed["agbd_mg_ha"].max())

    if high <= low:
        high = low + 1

    colours = LinearColormap(
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
        control_scale=True,
        prefer_canvas=True,
    )

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri, Maxar, Earthstar Geographics",
        name="Satellite imagery",
    ).add_to(map_view)

    folium.TileLayer(
        "OpenStreetMap",
        name="Place names and roads",
    ).add_to(map_view)

    points_layer = folium.FeatureGroup(
        name="GEDI footprint centres",
    )

    for row in displayed.itertuples(index=False):
        popup = (
            f"<b>GEDI reference observation</b><br>"
            f"Latitude: {row.latitude:.6f}<br>"
            f"Longitude: {row.longitude:.6f}<br>"
            f"Biomass: {row.agbd_mg_ha:.2f} Mg/ha<br>"
            f"Standard error: {row.agbd_se_mg_ha:.2f} Mg/ha<br>"
            f"Date: {html.escape(str(row.observation_time))}"
        )

        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=3,
            color=colours(row.agbd_mg_ha),
            weight=1,
            fill=True,
            fill_color=colours(row.agbd_mg_ha),
            fill_opacity=0.85,
            tooltip="Click for GEDI details",
            popup=folium.Popup(popup, max_width=320),
        ).add_to(points_layer)

    points_layer.add_to(map_view)
    colours.add_to(map_view)

    # Draw a proposed AOI and export it as GeoJSON.
    Draw(
        export=True,
        filename="gedi_selected_aoi.geojson",
        draw_options={
            "polygon": True,
            "rectangle": True,
            "polyline": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
        },
        edit_options={
            "edit": True,
            "remove": True,
        },
    ).add_to(map_view)

    MousePosition(
        position="bottomleft",
        separator=" | ",
        prefix="Lat / Lon:",
        num_digits=5,
    ).add_to(map_view)

    folium.LayerControl().add_to(map_view)

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

    output = Path(__file__).resolve().parent / "gedi_footprints.html"
    map_view.save(str(output))

    print(f"Map saved: {output}")
    print(
        f"Showing {len(displayed):,} of {len(table):,} observations."
    )
    print("Markers show centres, not actual footprint diameters.")

    webbrowser.open(output.as_uri())


if __name__ == "__main__":
    main()