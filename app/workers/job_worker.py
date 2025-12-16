"""
Job Worker - Processes jobs using X-Nemo and Seed-VC.

Each ML model runs as a separate subprocess to avoid Python import conflicts
(both repos have a `src/` directory with different modules).

Run with: python -m app.workers.job_worker

Environment Variables:
    SHARED_DATA_PATH: Base path for shared data (default: /home/samariva/projects/fred/shared_data)
    XNEMO_REPO_PATH: Path to x-nemo-inference repo (default: /home/samariva/tools/x-nemo-inference)
    XNEMO_WEIGHTS_PATH: Path to X-Nemo pretrained weights (default: XNEMO_REPO_PATH/pretrained_weights)
    SEEDVC_REPO_PATH: Path to seed-vc repo (default: /home/samariva/tools/seed-vc)
    EXECUTION_MODE: "sequential" or "parallel" (default: sequential for GPU safety)
    DEVICE: cuda/cpu (default: cuda)
    PYTHON_EXECUTABLE: Python interpreter to use (default: python)
"""

import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from app.core import get_identity_audio_path, get_identity_image_path
from app.db import Job, JobStatus, SessionLocal
from app.workers.seedvc_runner import SeedVCRunner
from app.workers.xnemo_runner import XNemoRunner

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration - paths relative to project root by default
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SHARED_DATA_PATH = os.getenv(
    "SHARED_DATA_PATH", os.path.join(PROJECT_ROOT, "shared_data")
)
XNEMO_REPO_PATH = os.getenv(
    "XNEMO_REPO_PATH", os.path.join(PROJECT_ROOT, "tools", "x-nemo-inference")
)
XNEMO_WEIGHTS_PATH = os.getenv(
    "XNEMO_WEIGHTS_PATH", os.path.join(XNEMO_REPO_PATH, "pretrained_weights")
)
SEEDVC_REPO_PATH = os.getenv(
    "SEEDVC_REPO_PATH", os.path.join(PROJECT_ROOT, "tools", "seed-vc")
)
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "sequential")  # "sequential" or "parallel"
DEVICE = os.getenv("DEVICE", "cuda")
POLL_INTERVAL = 2


def update_progress(job_id, progress: int, status: JobStatus = None):
    """Update job progress in database."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.progress = progress
            if status:
                job.status = status
            db.commit()
    finally:
        db.close()


def extract_audio(video_path: str, audio_path: str):
    """Extract audio from video using ffmpeg at 22050Hz (Seed-VC sample rate)."""
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "22050",  # Seed-VC expects 22050Hz
        "-ac",
        "1",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")
    logger.info(f"Extracted audio to {audio_path}")


def run_xnemo_task(job_id: str, user_video_path: str, identity_image_path: str) -> str:
    """
    Run X-Nemo: user_video (motion source) + identity_image (face) -> reenacted video.

    X-Nemo takes:
        - source: The motion source video (user's video)
        - reference: The face image to animate (identity's face)
    """
    logger.info(f"[{job_id}] Starting X-Nemo reenactment")
    logger.info(f"[{job_id}]   Motion source: {user_video_path}")
    logger.info(f"[{job_id}]   Face reference: {identity_image_path}")

    output_video = os.path.join(SHARED_DATA_PATH, "temp", f"{job_id}_xnemo.mp4")
    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    runner = XNemoRunner(
        pretrained_weights_path=XNEMO_WEIGHTS_PATH,
        device=DEVICE,
        dtype="fp16",
        verbose=True,
    )

    result = runner.generate(
        source_video=user_video_path,  # User's video provides motion
        identity_image=identity_image_path,  # Identity provides face
        output_path=output_video,
        steps=25,
        guidance_scale=2.5,
    )

    logger.info(f"[{job_id}] X-Nemo completed: {result}")
    return output_video


def run_seedvc_task(job_id: str, user_video_path: str, identity_audio_path: str) -> str:
    """
    Run Seed-VC: audio from user_video + identity_audio -> converted audio.

    Seed-VC takes:
        - source: The audio to convert (from user's video)
        - reference: The target voice sample (identity's voice)
    """
    logger.info(f"[{job_id}] Starting Seed-VC voice conversion")
    logger.info(f"[{job_id}]   Source audio from: {user_video_path}")
    logger.info(f"[{job_id}]   Target voice: {identity_audio_path}")

    # First extract audio from user's video
    user_audio_path = os.path.join(SHARED_DATA_PATH, "temp", f"{job_id}_user_audio.wav")
    extract_audio(user_video_path, user_audio_path)

    output_audio = os.path.join(SHARED_DATA_PATH, "temp", f"{job_id}_converted.wav")
    os.makedirs(os.path.dirname(output_audio), exist_ok=True)

    runner = SeedVCRunner(
        device=DEVICE,
        dtype="fp16",
        compile_model=False,
        verbose=True,
    )

    result = runner.convert(
        source_audio=user_audio_path,  # User's extracted audio
        reference_audio=identity_audio_path,  # Identity's voice sample
        output_path=output_audio,
        diffusion_steps=30,
        intelligibility_cfg_rate=0.7,
        similarity_cfg_rate=0.7,
    )

    logger.info(f"[{job_id}] Seed-VC completed: {result}")
    return output_audio


def combine_video_audio(job_id: str, video_path: str, audio_path: str) -> str:
    """Combine reenacted video with converted audio."""
    logger.info(f"[{job_id}] Combining video and audio")
    logger.info(f"[{job_id}]   Video: {video_path}")
    logger.info(f"[{job_id}]   Audio: {audio_path}")

    output_path = os.path.join(SHARED_DATA_PATH, "output", f"{job_id}.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg combine failed: {result.stderr}")

    logger.info(f"[{job_id}] Combined output: {output_path}")
    return output_path


def cleanup_temp(job_id: str):
    """No-op: keep temp files for debugging.

    We keep all per-job temp artifacts under shared_data/temp so that
    failures can be inspected later (X-Nemo raw video, extracted audio,
    Seed-VC outputs, etc.).
    """
    temp_dir = os.path.join(SHARED_DATA_PATH, "temp")
    logger.info(f"[{job_id}] Keeping temp files in: {temp_dir}")


def validate_paths(
    user_video_path: str, identity_image_path: str, identity_audio_path: str
):
    """Validate all required input paths exist."""
    if not os.path.exists(user_video_path):
        raise FileNotFoundError(f"User video not found: {user_video_path}")
    if not os.path.exists(identity_image_path):
        raise FileNotFoundError(f"Identity image not found: {identity_image_path}")
    if not os.path.exists(identity_audio_path):
        raise FileNotFoundError(f"Identity audio not found: {identity_audio_path}")


def process_job_sequential(job: Job):
    """
    Process job sequentially (safer for GPU memory).

    Order: X-Nemo first, then Seed-VC.
    """
    job_id = str(job.id)
    logger.info(f"Processing job {job_id} (sequential mode)")

    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job.id).first()
        job.status = JobStatus.PROCESSING
        job.progress = 5
        db.commit()

        # Get paths
        user_video_path = os.path.join(SHARED_DATA_PATH, "uploads", job.user_video)
        identity_image_path = get_identity_image_path(
            job.identity_id, job.identity_image
        )
        identity_audio_path = get_identity_audio_path(job.identity_id)

        # Validate inputs exist
        validate_paths(user_video_path, identity_image_path, identity_audio_path)

        # Step 1: X-Nemo (face reenactment) - 5% to 50%
        update_progress(job.id, 10)
        xnemo_result = run_xnemo_task(job_id, user_video_path, identity_image_path)
        update_progress(job.id, 50)

        # Step 2: Seed-VC (voice conversion) - 50% to 85%
        seedvc_result = run_seedvc_task(job_id, user_video_path, identity_audio_path)
        update_progress(job.id, 85)

        # Step 3: Combine video + audio - 85% to 95%
        final_output = combine_video_audio(job_id, xnemo_result, seedvc_result)
        update_progress(job.id, 95)

        # Mark completed
        job = db.query(Job).filter(Job.id == job.id).first()
        job.output_video = f"{job_id}.mp4"
        job.status = JobStatus.COMPLETED
        job.progress = 100
        db.commit()

        cleanup_temp(job_id)
        logger.info(f"Job {job_id} completed successfully!")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        job = db.query(Job).filter(Job.id == job.id).first()
        if job:
            job.status = JobStatus.FAILED
            job.error = str(e)
            db.commit()
    finally:
        db.close()


def process_job_parallel(job: Job):
    """
    Process job with parallel X-Nemo and Seed-VC.

    Note: Requires enough GPU memory for both models simultaneously.
    Since each runs in its own subprocess, they compete for GPU memory.
    """
    job_id = str(job.id)
    logger.info(f"Processing job {job_id} (parallel mode)")

    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job.id).first()
        job.status = JobStatus.PROCESSING
        job.progress = 5
        db.commit()

        # Get paths
        user_video_path = os.path.join(SHARED_DATA_PATH, "uploads", job.user_video)
        identity_image_path = get_identity_image_path(
            job.identity_id, job.identity_image
        )
        identity_audio_path = get_identity_audio_path(job.identity_id)

        # Validate inputs
        validate_paths(user_video_path, identity_image_path, identity_audio_path)

        # Run X-Nemo and Seed-VC in parallel
        update_progress(job.id, 10)

        xnemo_result = None
        seedvc_result = None
        errors = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    run_xnemo_task, job_id, user_video_path, identity_image_path
                ): "xnemo",
                executor.submit(
                    run_seedvc_task, job_id, user_video_path, identity_audio_path
                ): "seedvc",
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    if name == "xnemo":
                        xnemo_result = result
                        update_progress(job.id, 50)
                    else:
                        seedvc_result = result
                        update_progress(job.id, 70)
                except Exception as e:
                    errors.append(f"{name}: {str(e)}")
                    logger.error(f"[{job_id}] {name} failed: {e}", exc_info=True)

        if errors:
            raise RuntimeError("; ".join(errors))

        # Combine video + audio
        update_progress(job.id, 85)
        final_output = combine_video_audio(job_id, xnemo_result, seedvc_result)

        # Mark completed
        job = db.query(Job).filter(Job.id == job.id).first()
        job.output_video = f"{job_id}.mp4"
        job.status = JobStatus.COMPLETED
        job.progress = 100
        db.commit()

        cleanup_temp(job_id)
        logger.info(f"Job {job_id} completed successfully!")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        job = db.query(Job).filter(Job.id == job.id).first()
        if job:
            job.status = JobStatus.FAILED
            job.error = str(e)
            db.commit()
    finally:
        db.close()


def process_job(job: Job):
    """Process a job using configured execution mode."""
    if EXECUTION_MODE == "parallel":
        process_job_parallel(job)
    else:
        process_job_sequential(job)


def verify_setup():
    """Verify that all required components are in place."""
    errors = []

    # Check X-Nemo repo
    xnemo_script = os.path.join(XNEMO_REPO_PATH, "run_xnemo.py")
    if not os.path.exists(xnemo_script):
        errors.append(f"X-Nemo runner script not found: {xnemo_script}")

    # Check X-Nemo weights directory
    if not os.path.exists(XNEMO_WEIGHTS_PATH):
        errors.append(f"X-Nemo weights not found: {XNEMO_WEIGHTS_PATH}")

    # Check Seed-VC repo
    seedvc_script = os.path.join(SEEDVC_REPO_PATH, "run_seedvc.py")
    if not os.path.exists(seedvc_script):
        errors.append(f"Seed-VC runner script not found: {seedvc_script}")

    # Check shared data directories
    for subdir in ["uploads", "temp", "output"]:
        path = os.path.join(SHARED_DATA_PATH, subdir)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created directory: {path}")

    if errors:
        for error in errors:
            logger.error(error)
        return False

    return True


def main():
    """Main loop - poll for pending jobs."""
    logger.info("=" * 60)
    logger.info("Job Worker Starting")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info(f"  SHARED_DATA_PATH: {SHARED_DATA_PATH}")
    logger.info(f"  XNEMO_REPO_PATH: {XNEMO_REPO_PATH}")
    logger.info(f"  XNEMO_WEIGHTS_PATH: {XNEMO_WEIGHTS_PATH}")
    logger.info(f"  SEEDVC_REPO_PATH: {SEEDVC_REPO_PATH}")
    logger.info(f"  EXECUTION_MODE: {EXECUTION_MODE}")
    logger.info(f"  DEVICE: {DEVICE}")
    logger.info("=" * 60)

    # Verify setup
    if not verify_setup():
        logger.error("Setup verification failed! Fix the errors above and restart.")
        sys.exit(1)

    logger.info("Setup verified. Worker ready, polling for jobs...")

    while True:
        db = SessionLocal()
        try:
            job = (
                db.query(Job)
                .filter(Job.status == JobStatus.PENDING)
                .order_by(Job.created_at)
                .first()
            )
            if job:
                process_job(job)
            else:
                time.sleep(POLL_INTERVAL)
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            time.sleep(POLL_INTERVAL)
        finally:
            db.close()


if __name__ == "__main__":
    main()
