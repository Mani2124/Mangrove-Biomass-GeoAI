"""Configure Sentinel-1 download and SNAP processing."""
def command(run, start, end, scenes):
    return "v3_process_sentinel1.py", ["--run-dir", run, "--start-date", start,
                                     "--end-date", end, "--max-products", scenes]
