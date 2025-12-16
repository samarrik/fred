"""
Pydantic schemas - minimal for MVP.
"""
from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel

from app.db.models import JobStatus


class JobCreate(BaseModel):
    """Create a job: pick identity + image, provide video filename."""
    identity_id: str
    identity_image: str  # Which image from the identity they chose
    user_video: str  # Filename of uploaded video


class JobProgress(BaseModel):
    """Lightweight progress response for polling."""
    status: JobStatus
    progress: int

    class Config:
        from_attributes = True


class JobResponse(BaseModel):
    """Full job info."""
    id: UUID
    identity_id: str
    identity_image: str
    user_video: str
    status: JobStatus
    progress: int
    output_video: Optional[str]
    error: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True
