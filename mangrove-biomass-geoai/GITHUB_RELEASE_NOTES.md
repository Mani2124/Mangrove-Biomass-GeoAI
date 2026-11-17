# Public repository notes

This package is a cleaned copy of the uploaded research code, not a rewritten or independently verified application.

- Removed the local `.env`, machine-specific `local_settings.json`, local SQLite database, caches and compiled Python files.
- Added `.env.example` with empty placeholders. Verify environment variable names against your local installation before use.
- Replaced the machine-specific README launch commands with portable Windows commands.
- Removed the Docker Compose dependency on a local `.env` file and local ALOS directory. Configure data credentials and ALOS mounts separately if those features are needed in Docker.
- No raw imagery, GEDI files, models or project outputs are bundled.
- Scientific results reported in the presentation have **not** been independently reproduced here.
- This is experimental research, not an independently validated carbon inventory.
- Check third-party dataset redistribution rights before adding data or screenshots.
- No open-source license has been added: select a license only if you intend to grant reuse rights.
