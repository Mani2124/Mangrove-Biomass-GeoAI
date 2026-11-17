import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
DATA_ROOT = Path(os.getenv("SPACEVEXA_DATA_ROOT", str(ROOT / "saas_data"))).resolve()
DATABASE_PATH = Path(os.getenv("SPACEVEXA_DATABASE_PATH", str(DATA_ROOT / "spacevexa.db"))).resolve()
