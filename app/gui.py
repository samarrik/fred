#!/usr/bin/env python3
"""
Fred Gradio GUI - Simple interface for face reenactment + voice conversion.

Run with: python app/gui.py
"""

import os
import subprocess
import sys
import uuid

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

# Configuration - paths relative to project root by default
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED_DATA_PATH = os.getenv(
    "SHARED_DATA_PATH", os.path.join(PROJECT_ROOT, "shared_data")
)
IDENTITIES_PATH = os.getenv("IDENTITIES_PATH", os.path.join(PROJECT_ROOT, "identities"))
XNEMO_REPO_PATH = os.getenv(
    "XNEMO_REPO_PATH", os.path.join(PROJECT_ROOT, "tools", "x-nemo-inference")
)
XNEMO_WEIGHTS_PATH = os.getenv(
    "XNEMO_WEIGHTS_PATH", os.path.join(XNEMO_REPO_PATH, "pretrained_weights")
)
SEEDVC_REPO_PATH = os.getenv(
    "SEEDVC_REPO_PATH", os.path.join(PROJECT_ROOT, "tools", "seed-vc")
)
DEVICE = os.getenv("DEVICE", "cuda")

# Import identity management
from app.core.identities import (
    IDENTITIES,
    get_identity_audio_path,
    get_identity_image_path,
)


def get_identity_choices():
    """Get list of identity choices for dropdown."""
    choices = list(IDENTITIES.keys())
    print(f"Identity choices: {choices}")  # Debug
    return choices


def get_image_choices(identity_id: str):
    """Get list of image choices for an identity."""
    if not identity_id or identity_id not in IDENTITIES:
        print(f"No images for identity: {identity_id}")  # Debug
        return []
    images = IDENTITIES[identity_id]["images"]
    print(f"Images for {identity_id}: {images}")  # Debug
    return images


def get_identity_preview(identity_id: str):
    """Get preview image for an identity."""
    if not identity_id or identity_id not in IDENTITIES:
        return None
    images = IDENTITIES[identity_id]["images"]
    if images:
        path = get_identity_image_path(identity_id, images[0])
        if os.path.exists(path):
            return path
    return None


def update_image_dropdown(identity_id: str):
    """Update image dropdown when identity changes."""
    print(f"Identity selected: {identity_id}")  # Debug
    choices = get_image_choices(identity_id)
    preview = get_identity_preview(identity_id)
    print(f"Updating dropdown with choices: {choices}, preview: {preview}")  # Debug
    return (
        gr.update(choices=choices, value=choices[0] if choices else None),
        preview,
    )


def update_image_preview(identity_id: str, image_name: str):
    """Update image preview when selection changes."""
    if not identity_id or not image_name:
        return None
    path = get_identity_image_path(identity_id, image_name)
    if os.path.exists(path):
        return path
    return None


def extract_audio(video_path: str, audio_path: str):
    """Extract audio from video using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "22050",
        "-ac",
        "1",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")


def combine_video_audio(video_path: str, audio_path: str, output_path: str):
    """Combine video and audio using ffmpeg with web-compatible codecs.

    We don't touch FPS at all – we just re-encode the X-Nemo video to a
    browser‑friendly H.264 variant and mux in the converted audio.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
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


def process_video(
    identity_id: str,
    image_name: str,
    video_file,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Process video through the pipeline.

    Args:
        identity_id: Selected identity ID
        image_name: Selected image filename
        video_file: Uploaded video file path
        progress: Gradio progress tracker
    """
    if not identity_id:
        raise gr.Error("Please select an identity")
    if not image_name:
        raise gr.Error("Please select a face image")
    if video_file is None:
        raise gr.Error("Please upload a video")

    # Validate identity assets exist
    identity_image_path = get_identity_image_path(identity_id, image_name)
    identity_audio_path = get_identity_audio_path(identity_id)

    if not os.path.exists(identity_image_path):
        raise gr.Error(f"Identity image not found: {identity_image_path}")
    if not os.path.exists(identity_audio_path):
        raise gr.Error(f"Identity audio not found: {identity_audio_path}")

    # Create job ID and paths
    job_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join(SHARED_DATA_PATH, "temp")
    output_dir = os.path.join(SHARED_DATA_PATH, "output")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    user_video_path = video_file
    xnemo_output = os.path.join(temp_dir, f"{job_id}_xnemo.mp4")
    user_audio = os.path.join(temp_dir, f"{job_id}_user_audio.wav")
    converted_audio = os.path.join(temp_dir, f"{job_id}_converted.wav")
    final_output = os.path.join(output_dir, f"{job_id}_final.mp4")

    try:
        # Step 1: Run X-Nemo (Face Reenactment)
        progress(0.1, desc="Running X-Nemo (face reenactment)...")

        from app.workers.xnemo_runner import XNemoRunner

        xnemo_runner = XNemoRunner(
            pretrained_weights_path=XNEMO_WEIGHTS_PATH,
            device=DEVICE,
            dtype="fp16",
            verbose=True,
        )

        xnemo_runner.generate(
            source_video=user_video_path,
            identity_image=identity_image_path,
            output_path=xnemo_output,
            steps=25,
            guidance_scale=2.5,
        )

        progress(0.5, desc="X-Nemo complete. Extracting audio...")

        # Step 2: Extract audio from user video
        extract_audio(user_video_path, user_audio)

        progress(0.55, desc="Running Seed-VC (voice conversion)...")

        # Step 3: Run Seed-VC (Voice Conversion)
        from app.workers.seedvc_runner import SeedVCRunner

        seedvc_runner = SeedVCRunner(
            device=DEVICE,
            dtype="fp16",
            compile_model=False,
            verbose=True,
        )

        seedvc_runner.convert(
            source_audio=user_audio,
            reference_audio=identity_audio_path,
            output_path=converted_audio,
            diffusion_steps=30,
            intelligibility_cfg_rate=0.7,
            similarity_cfg_rate=0.7,
        )

        progress(0.9, desc="Seed-VC complete. Combining video and audio...")

        # Step 4: Combine video + audio
        combine_video_audio(xnemo_output, converted_audio, final_output)

        progress(1.0, desc="Done!")

        # NOTE: We intentionally keep temp files (X-Nemo output + intermediate audio)
        # under shared_data/temp for debugging and inspection:
        #   - xnemo_output
        #   - user_audio
        #   - converted_audio
        # They are not deleted automatically.

        return final_output

    except Exception as e:
        # Keep temp files on error as well for debugging.
        raise gr.Error(f"Processing failed: {str(e)}")


def create_gui():
    """Create the Gradio interface."""

    with gr.Blocks(
        title="Fred - Face Reenactment + Voice Conversion",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
        # 🎭 Fred - Face Reenactment + Voice Conversion
        
        Transform your video with a different identity's face and voice!
        
        1. **Select an identity** - Choose whose face and voice to use
        2. **Pick a face image** - Select which face image to animate
        3. **Upload your video** - This provides the motion and speech content
        4. **Generate!** - Click to create the transformed video
        """
        )

        with gr.Row():
            # Left column: Inputs
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Settings")

                # Identity selection
                identity_choices = get_identity_choices()
                default_identity = identity_choices[0] if identity_choices else None
                identity_dropdown = gr.Dropdown(
                    choices=identity_choices,
                    value=default_identity,
                    label="Select Identity",
                    info="Choose the identity (face + voice) to use",
                )

                # Image selection (updated based on identity)
                default_images = (
                    get_image_choices(default_identity) if default_identity else []
                )
                image_dropdown = gr.Dropdown(
                    choices=default_images,
                    value=default_images[0] if default_images else None,
                    label="Select Face Image",
                    info="Choose which face image to animate",
                )

                # Preview of selected identity image
                default_preview = (
                    get_identity_preview(default_identity) if default_identity else None
                )
                identity_preview = gr.Image(
                    label="Identity Preview",
                    type="filepath",
                    value=default_preview,
                    interactive=False,
                    height=200,
                )

                # Video upload
                video_input = gr.Video(
                    label="Upload Your Video",
                    sources=["upload"],
                )

                # Generate button
                generate_btn = gr.Button(
                    "🎬 Generate Video",
                    variant="primary",
                    size="lg",
                )

            # Right column: Output
            with gr.Column(scale=1):
                gr.Markdown("### 🎥 Result")

                video_output = gr.Video(
                    label="Generated Video",
                    interactive=False,
                    autoplay=True,
                )

                gr.Markdown(
                    """
                ### ℹ️ How it works
                
                1. **X-Nemo** analyzes your video's facial movements and transfers them to the identity's face
                2. **Seed-VC** converts your voice to match the identity's voice  
                3. The results are combined into the final video
                
                ⏱️ Processing typically takes 2-5 minutes depending on video length.
                """
                )

        # Event handlers
        identity_dropdown.change(
            fn=update_image_dropdown,
            inputs=[identity_dropdown],
            outputs=[image_dropdown, identity_preview],
        )

        image_dropdown.change(
            fn=update_image_preview,
            inputs=[identity_dropdown, image_dropdown],
            outputs=[identity_preview],
        )

        generate_btn.click(
            fn=process_video,
            inputs=[identity_dropdown, image_dropdown, video_input],
            outputs=[video_output],
        )

    return demo


def main():
    """Run the Gradio app."""
    print("=" * 60)
    print("Fred GUI Starting")
    print("=" * 60)
    print(f"SHARED_DATA_PATH: {SHARED_DATA_PATH}")
    print(f"IDENTITIES_PATH: {IDENTITIES_PATH}")
    print(f"XNEMO_REPO_PATH: {XNEMO_REPO_PATH}")
    print(f"SEEDVC_REPO_PATH: {SEEDVC_REPO_PATH}")
    print(f"DEVICE: {DEVICE}")
    print("=" * 60)

    # Create required directories
    for subdir in ["uploads", "temp", "output"]:
        os.makedirs(os.path.join(SHARED_DATA_PATH, subdir), exist_ok=True)

    # Verify setup
    errors = []

    xnemo_script = os.path.join(XNEMO_REPO_PATH, "run_xnemo.py")
    if not os.path.exists(xnemo_script):
        errors.append(f"X-Nemo runner not found: {xnemo_script}")

    seedvc_script = os.path.join(SEEDVC_REPO_PATH, "run_seedvc.py")
    if not os.path.exists(seedvc_script):
        errors.append(f"Seed-VC runner not found: {seedvc_script}")

    if not os.path.exists(XNEMO_WEIGHTS_PATH):
        errors.append(f"X-Nemo weights not found: {XNEMO_WEIGHTS_PATH}")

    if errors:
        print("\n⚠️  Setup warnings:")
        for error in errors:
            print(f"  - {error}")
        print("\nThe GUI will start, but processing may fail.")
        print("Run 'python scripts/verify_setup.py' for full verification.\n")

    # Launch GUI
    demo = create_gui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
