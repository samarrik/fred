#!/bin/bash
# Load cluster modules first
module load CUDA/12.2.0
module load FFmpeg

cd "$(dirname "$0")/.."

export DATABASE_URL="${DATABASE_URL:-postgresql://fred:fred@localhost:5432/fred}"
export SHARED_DATA_PATH="${SHARED_DATA_PATH:-$(pwd)/shared_data}"
export IDENTITIES_PATH="${IDENTITIES_PATH:-$(pwd)/identities}"
export XNEMO_SIF="${XNEMO_SIF:-$(pwd)/containers/x-nemo/xnemo.sif}"
export XNEMO_WEIGHTS="${XNEMO_WEIGHTS:-$(pwd)/pretrained_weights/xnemo}"
export SEEDVC_SIF="${SEEDVC_SIF:-$(pwd)/containers/seed-vc/seedvc.sif}"

python -m app.workers.job_worker
