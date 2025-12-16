"""
Seed-VC Runner - Wrapper for Seed-VC voice conversion.

Uses subprocess to avoid Python import conflicts between ML repos.
"""

import json
import logging
import os
import subprocess

logger = logging.getLogger(__name__)

# Configuration - paths relative to project root by default
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SEEDVC_REPO_PATH = os.getenv(
    "SEEDVC_REPO_PATH", os.path.join(PROJECT_ROOT, "tools", "seed-vc")
)
PYTHON_EXECUTABLE = os.getenv("PYTHON_EXECUTABLE", "python")


class SeedVCRunner:
    """
    Wrapper for Seed-VC voice conversion.

    Runs Seed-VC as a subprocess to avoid import conflicts.
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "fp16",
        compile_model: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize Seed-VC runner.

        Args:
            device: Device to use ("cuda", "cpu", or "mps")
            dtype: Data type ("fp16" or "fp32")
            compile_model: Use torch.compile for faster inference
            verbose: Print progress messages
        """
        self.device = device
        self.dtype = dtype
        self.compile_model = compile_model
        self.verbose = verbose
        self.script_path = os.path.join(SEEDVC_REPO_PATH, "run_seedvc.py")

        if not os.path.exists(self.script_path):
            raise FileNotFoundError(
                f"Seed-VC runner script not found: {self.script_path}"
            )

    def convert(
        self,
        source_audio: str,
        reference_audio: str,
        output_path: str,
        diffusion_steps: int = 30,
        length_adjust: float = 1.0,
        intelligibility_cfg_rate: float = 0.7,
        similarity_cfg_rate: float = 0.7,
        convert_style: bool = False,
        seed: int = 42,
        timeout: int = 1800,  # 30 minute timeout
    ) -> str:
        """
        Convert voice from source to match reference.

        Args:
            source_audio: Path to source audio (voice to convert)
            reference_audio: Path to reference audio (target voice, 1-25s)
            output_path: Path for output audio
            diffusion_steps: Number of diffusion steps (default 30)
            length_adjust: Length adjustment factor (default 1.0)
            intelligibility_cfg_rate: Speech clarity (0.0-1.0)
            similarity_cfg_rate: Voice similarity (0.0-1.0)
            convert_style: Enable accent/emotion conversion
            seed: Random seed (default 42)
            timeout: Timeout in seconds (default 30 minutes)

        Returns:
            Path to converted audio
        """
        # Build command
        cmd = [
            PYTHON_EXECUTABLE,
            self.script_path,
            "--source",
            source_audio,
            "--reference",
            reference_audio,
            "--output",
            output_path,
            "--diffusion-steps",
            str(diffusion_steps),
            "--length-adjust",
            str(length_adjust),
            "--intelligibility-cfg-rate",
            str(intelligibility_cfg_rate),
            "--similarity-cfg-rate",
            str(similarity_cfg_rate),
            "--seed",
            str(seed),
            "--device",
            self.device,
            "--dtype",
            self.dtype,
        ]

        if convert_style:
            cmd.append("--convert-style")

        if self.compile_model:
            cmd.append("--compile")

        if not self.verbose:
            cmd.append("--quiet")

        logger.info(f"Running Seed-VC: {' '.join(cmd)}")

        # Run subprocess
        try:
            result = subprocess.run(
                cmd,
                cwd=SEEDVC_REPO_PATH,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                logger.error(f"Seed-VC stderr: {result.stderr}")
                raise RuntimeError(
                    f"Seed-VC failed with code {result.returncode}: {result.stderr}"
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

            raise RuntimeError(f"Seed-VC completed but output not found: {output_path}")

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Seed-VC timed out after {timeout} seconds")


def get_seedvc_runner(
    device: str = "cuda",
    dtype: str = "fp16",
    compile_model: bool = False,
    verbose: bool = True,
) -> SeedVCRunner:
    """
    Create Seed-VC runner instance.
    """
    return SeedVCRunner(
        device=device,
        dtype=dtype,
        compile_model=compile_model,
        verbose=verbose,
    )


def run_seedvc(
    source_audio: str,
    reference_audio: str,
    output_path: str,
    **kwargs,
) -> str:
    """
    Convenience function to run Seed-VC voice conversion.

    Args:
        source_audio: Path to source audio (voice to convert)
        reference_audio: Path to reference audio (target voice)
        output_path: Path for output audio
        **kwargs: Additional conversion parameters

    Returns:
        Path to converted audio
    """
    runner = get_seedvc_runner()
    return runner.convert(
        source_audio=source_audio,
        reference_audio=reference_audio,
        output_path=output_path,
        **kwargs,
    )
