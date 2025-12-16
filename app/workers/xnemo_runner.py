"""
X-Nemo Runner - Wrapper for X-Nemo video generation.

Uses subprocess to avoid Python import conflicts between ML repos.
"""

import json
import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# Configuration - paths relative to project root by default
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
XNEMO_REPO_PATH = os.getenv(
    "XNEMO_REPO_PATH", os.path.join(PROJECT_ROOT, "tools", "x-nemo-inference")
)
XNEMO_WEIGHTS_PATH = os.getenv(
    "XNEMO_WEIGHTS_PATH", os.path.join(XNEMO_REPO_PATH, "pretrained_weights")
)
PYTHON_EXECUTABLE = os.getenv("PYTHON_EXECUTABLE", "python")


class XNemoRunner:
    """
    Wrapper for X-Nemo video generation.

    Runs X-Nemo as a subprocess to avoid import conflicts.
    """

    def __init__(
        self,
        pretrained_weights_path: str = None,
        device: str = "cuda",
        dtype: str = "fp16",
        verbose: bool = True,
    ):
        """
        Initialize X-Nemo runner.

        Args:
            pretrained_weights_path: Path to pretrained weights directory.
            device: Device to use ("cuda" or "cpu")
            dtype: Data type ("fp16", "bf16", or "fp32")
            verbose: Print progress messages
        """
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.pretrained_weights_path = pretrained_weights_path or XNEMO_WEIGHTS_PATH
        self.script_path = os.path.join(XNEMO_REPO_PATH, "run_xnemo.py")

        if not os.path.exists(self.script_path):
            raise FileNotFoundError(
                f"X-Nemo runner script not found: {self.script_path}"
            )

    def generate(
        self,
        source_video: str,
        identity_image: str,
        output_path: str,
        width: int = 512,
        height: int = 512,
        steps: int = 25,
        guidance_scale: float = 2.5,
        seed: int = 42,
        max_frames: Optional[int] = None,
        timeout: int = 1800,  # 30 minute timeout
    ) -> str:
        """
        Generate a reenacted video.

        Args:
            source_video: Path to user's video (motion source)
            identity_image: Path to identity's face image
            output_path: Path for output video
            width: Output width (default 512)
            height: Output height (default 512)
            steps: Denoising steps (default 25)
            guidance_scale: Guidance scale (default 2.5)
            fps: Output FPS (default 25)
            seed: Random seed (default 42)
            max_frames: Max frames to process (None = all)
            timeout: Timeout in seconds (default 30 minutes)

        Returns:
            Path to generated video
        """
        # Build command
        cmd = [
            PYTHON_EXECUTABLE,
            self.script_path,
            "--source",
            source_video,
            "--reference",
            identity_image,
            "--output",
            output_path,
            "--pretrained-model",
            os.path.join(self.pretrained_weights_path, "sd-image-variations-diffusers"),
            "--vae-path",
            os.path.join(
                self.pretrained_weights_path, "stable-video-diffusion-img2vid-xt/vae"
            ),
            "--denoising-unet",
            os.path.join(self.pretrained_weights_path, "xnemo_denoising_unet.pth"),
            "--temporal-module",
            os.path.join(self.pretrained_weights_path, "xnemo_temporal_module.pth"),
            "--width",
            str(width),
            "--height",
            str(height),
            "--steps",
            str(steps),
            "--guidance",
            str(guidance_scale),
            "--seed",
            str(seed),
            "--device",
            self.device,
            "--dtype",
            self.dtype,
        ]

        if max_frames is not None:
            cmd.extend(["--max-frames", str(max_frames)])

        if not self.verbose:
            cmd.append("--quiet")

        logger.info(f"Running X-Nemo: {' '.join(cmd)}")

        # Run subprocess
        try:
            result = subprocess.run(
                cmd,
                cwd=XNEMO_REPO_PATH,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                logger.error(f"X-Nemo stderr: {result.stderr}")
                raise RuntimeError(
                    f"X-Nemo failed with code {result.returncode}: {result.stderr}"
                )

            # Parse result from output
            stdout = result.stdout
            if self.verbose:
                print(stdout)

            # Look for JSON result marker
            if "__RESULT_JSON__:" in stdout:
                json_str = stdout.split("__RESULT_JSON__:")[-1].strip()
                # Handle potential extra output after JSON
                if "\n" in json_str:
                    json_str = json_str.split("\n")[0]
                result_data = json.loads(json_str)
                if result_data.get("success"):
                    return result_data.get("output", output_path)

            # Fallback: assume success if file exists
            if os.path.exists(output_path):
                return output_path

            raise RuntimeError(f"X-Nemo completed but output not found: {output_path}")

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"X-Nemo timed out after {timeout} seconds")


def get_xnemo_runner(
    pretrained_weights_path: str = None,
    device: str = "cuda",
    dtype: str = "fp16",
    verbose: bool = True,
) -> XNemoRunner:
    """
    Create X-Nemo runner instance.
    """
    return XNemoRunner(
        pretrained_weights_path=pretrained_weights_path,
        device=device,
        dtype=dtype,
        verbose=verbose,
    )


def run_xnemo(
    source_video: str,
    identity_image: str,
    output_path: str,
    pretrained_weights_path: str = None,
    **kwargs,
) -> str:
    """
    Convenience function to run X-Nemo reenactment.

    Args:
        source_video: Path to user's video (motion source)
        identity_image: Path to identity's face image
        output_path: Path for output video
        pretrained_weights_path: Optional path to weights
        **kwargs: Additional generation parameters

    Returns:
        Path to generated video
    """
    runner = get_xnemo_runner(pretrained_weights_path=pretrained_weights_path)
    return runner.generate(
        source_video=source_video,
        identity_image=identity_image,
        output_path=output_path,
        **kwargs,
    )
