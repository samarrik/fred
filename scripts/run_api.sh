#!/bin/bash
cd "$(dirname "$0")/.."
export DATABASE_URL="${DATABASE_URL:-postgresql://fred:fred@localhost:5432/fred}"
export SHARED_DATA_PATH="${SHARED_DATA_PATH:-$(pwd)/shared_data}"
export IDENTITIES_PATH="${IDENTITIES_PATH:-$(pwd)/identities}"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
