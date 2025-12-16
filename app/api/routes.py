"""
MVP API - minimal endpoints.
"""

import os
import uuid
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core import get_identities, get_identity
from app.db import Job, JobStatus, get_db
from app.schemas import JobCreate, JobProgress, JobResponse

router = APIRouter(prefix="/api", tags=["api"])

# Configuration - paths relative to project root by default
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SHARED_DATA_PATH = os.getenv(
    "SHARED_DATA_PATH", os.path.join(PROJECT_ROOT, "shared_data")
)


@router.get("/identities")
def list_identities():
    """Get all available identities to choose from."""
    return get_identities()


@router.get("/identities/{identity_id}")
def get_identity_detail(identity_id: str):
    """Get a single identity with its images."""
    identity = get_identity(identity_id)
    if not identity:
        raise HTTPException(status_code=404, detail="Identity not found")
    return identity


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload user's video. Returns filename to use when creating job."""
    # Generate unique filename
    ext = os.path.splitext(file.filename)[1] or ".mp4"
    filename = f"{uuid.uuid4()}{ext}"

    upload_dir = os.path.join(SHARED_DATA_PATH, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return {"filename": filename}


@router.post("/jobs", response_model=JobResponse)
def create_job(data: JobCreate, db: Session = Depends(get_db)):
    """
    Create a new job.
    User has: picked identity, picked image, uploaded video.
    """
    # Validate identity exists
    identity = get_identity(data.identity_id)
    if not identity:
        raise HTTPException(status_code=400, detail="Invalid identity")

    # Validate image belongs to identity
    if data.identity_image not in identity["images"]:
        raise HTTPException(status_code=400, detail="Invalid image for identity")

    # Validate video was uploaded
    video_path = os.path.join(SHARED_DATA_PATH, "uploads", data.user_video)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=400, detail="Video not found. Upload first.")

    job = Job(
        user_video=data.user_video,
        identity_id=data.identity_id,
        identity_image=data.identity_image,
        status=JobStatus.PENDING,
        progress=0,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@router.get("/jobs/{job_id}/progress", response_model=JobProgress)
def get_progress(job_id: UUID, db: Session = Depends(get_db)):
    """Poll this for progress updates."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobProgress(status=job.status, progress=job.progress)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: UUID, db: Session = Depends(get_db)):
    """Get full job details."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/library", response_model=List[JobResponse])
def get_library(db: Session = Depends(get_db)):
    """Get user's completed jobs (their library)."""
    return (
        db.query(Job)
        .filter(Job.status == JobStatus.COMPLETED)
        .order_by(Job.created_at.desc())
        .all()
    )
