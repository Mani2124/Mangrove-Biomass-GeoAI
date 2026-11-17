"""One sequential job; stdout goes to disk rather than memory."""
import subprocess
import sys
from settings import ROOT

def run(script, args, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        code = subprocess.call([sys.executable, "-u", str(ROOT/"engines"/script),
                                *map(str, args)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if code:
        raise RuntimeError(f"Processing failed: see {log}")
