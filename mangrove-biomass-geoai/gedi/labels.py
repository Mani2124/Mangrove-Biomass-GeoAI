"""Prepare labels from local GEDI L4B rasters, not L4A shots."""
def command(folder, output):
    return "v6_gedi_l4b.py", ["--gedi-dir", folder, "--output-dir", output]
