"""
PostgreSQL database connection using SQLAlchemy.
Uses PostgreSQL's LISTEN/NOTIFY for progress tracking instead of RabbitMQ.
"""
import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://fred:fred@localhost:5432/fred"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency for FastAPI endpoints."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables and triggers."""
    from app.db.models import Job  # noqa: F401
    from sqlalchemy import text
    
    Base.metadata.create_all(bind=engine)
    
    # Create the trigger function for automatic updated_at (ON UPDATE behavior)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE OR REPLACE FUNCTION update_modified_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """))
        
        # Create trigger on jobs table (ignore if exists)
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger WHERE tgname = 'set_job_timestamp'
                ) THEN
                    CREATE TRIGGER set_job_timestamp
                    BEFORE UPDATE ON jobs
                    FOR EACH ROW
                    EXECUTE FUNCTION update_modified_column();
                END IF;
            END $$;
        """))
        conn.commit()

