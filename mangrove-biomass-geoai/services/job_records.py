from __future__ import annotations

from uuid import uuid4

from services.database import connection, utc_now


def start_job(owner_id, project_id, job_type):
    job_id = uuid4().hex
    now = utc_now()
    with connection() as database:
        database.execute(
            """INSERT INTO processing_jobs
               (id,owner_id,project_id,job_type,status,progress,message,created_at,started_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (job_id, owner_id, project_id, job_type, "running", 5, "Processing started", now, now),
        )
    return job_id


def finish_job(job_id, succeeded, message):
    with connection() as database:
        database.execute(
            "UPDATE processing_jobs SET status=?,progress=?,message=?,finished_at=? WHERE id=?",
            ("completed" if succeeded else "failed", 100 if succeeded else 0, message, utc_now(), job_id),
        )


def recent_jobs(owner_id, limit=10):
    with connection() as database:
        rows = database.execute(
            """SELECT project_id,job_type,status,progress,message,created_at
               FROM processing_jobs WHERE owner_id=? ORDER BY created_at DESC LIMIT ?""",
            (owner_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]
