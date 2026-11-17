"""Configure Sentinel-2 download and optical processing."""
def command(run, start, end, scenes):
    return "v2_process_sentinel2.py", ["--run-dir", run, "--start-date", start,
                                     "--end-date", end, "--max-products", scenes]
