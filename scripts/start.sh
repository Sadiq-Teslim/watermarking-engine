#!/usr/bin/env sh
# All-in-one entrypoint: background RQ worker + foreground web server.
# Used as the image's default CMD so a single Render Web Service runs both.
# (docker-compose / the Render blueprint override this to split web and worker.)
set -e

python -m worker.run &
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
