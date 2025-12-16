"""
Fred MVP - Video generation with identity reenactment + voice conversion.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import router
from app.db import init_db

# Configuration - paths relative to project root by default
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED_DATA_PATH = os.getenv(
    "SHARED_DATA_PATH", os.path.join(PROJECT_ROOT, "shared_data")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Fred", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve output videos
output_dir = os.path.join(SHARED_DATA_PATH, "output")
os.makedirs(output_dir, exist_ok=True)
app.mount("/output", StaticFiles(directory=output_dir), name="output")


@app.get("/health")
def health():
    return {"status": "ok"}
