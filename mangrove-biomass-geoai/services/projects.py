from __future__ import annotations

import re

from services.database import connection, utc_now
from settings import DATA_ROOT


def _safe(value):
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", str(value))
    if not cleaned:
        raise ValueError("Invalid identifier.")
    return cleaned


def user_runs_root(user_id):
    root = (DATA_ROOT / "users" / _safe(user_id) / "runs").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def register_project(owner_id, study_id, name=None):
    now = utc_now()
    with connection() as database:
        database.execute(
            "INSERT OR IGNORE INTO projects(id,owner_id,name,status,created_at,updated_at) VALUES(?,?,?,?,?,?)",
            (study_id, owner_id, name or study_id, "created", now, now),
        )


def project_count(owner_id):
    with connection() as database:
        return int(database.execute("SELECT COUNT(*) FROM projects WHERE owner_id=?", (owner_id,)).fetchone()[0])
