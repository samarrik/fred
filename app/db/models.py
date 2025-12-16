"""
Minimal database models for MVP.
"""
import enum
import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Text, Enum
from sqlalchemy.dialects.postgresql import UUID

from app.db.database import Base


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(Base):
    """Job queue - one row per video generation request."""
    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # User's uploaded video
    user_video = Column(String(500), nullable=False)
    
    # Selected identity
    identity_id = Column(String(100), nullable=False)
    identity_image = Column(String(500), nullable=False)  # Which image they picked
    
    # Status
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False)
    progress = Column(Integer, default=0)  # 0-100
    
    # Output
    output_video = Column(String(500), nullable=True)
    error = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
