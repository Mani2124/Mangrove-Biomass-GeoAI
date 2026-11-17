"""Local annual HH/HV mosaics, not an automatic downloader."""
def command(run, year):
    return "v5_process_alos2.py", ["--run-dir", run, "--year", year]
