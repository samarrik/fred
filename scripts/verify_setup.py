#!/usr/bin/env python3
"""
Verify that all components of the Fred pipeline are correctly set up.

Run with: python scripts/verify_setup.py
"""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def check_path(path: str, description: str) -> bool:
    """Check if a path exists."""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {path}")
    return exists


def main():
    print("=" * 60)
    print("Fred Pipeline Setup Verification")
    print("=" * 60)

    errors = []

    # Configuration - use relative paths by default
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
    IDENTITIES_PATH = os.getenv(
        "IDENTITIES_PATH", os.path.join(PROJECT_ROOT, "identities")
    )

    print(f"\nProject root: {PROJECT_ROOT}")

    print("\n[1] X-Nemo Repository")
    print("-" * 40)

    xnemo_checks = [
        (XNEMO_REPO_PATH, "Repo root"),
        (os.path.join(XNEMO_REPO_PATH, "xnemo_api.py"), "Main API"),
        (os.path.join(XNEMO_REPO_PATH, "run_xnemo.py"), "Runner script"),
        (
            os.path.join(XNEMO_REPO_PATH, "blaze_face_short_range.tflite"),
            "Face detector",
        ),
        (
            os.path.join(
                XNEMO_REPO_PATH, "configs/inference/inference_xnemo_stage2.yaml"
            ),
            "Inference config",
        ),
    ]

    for path, desc in xnemo_checks:
        if not check_path(path, desc):
            errors.append(f"X-Nemo: {desc} not found")

    print("\n[2] X-Nemo Pretrained Weights")
    print("-" * 40)

    weights_checks = [
        (XNEMO_WEIGHTS_PATH, "Weights directory"),
        (
            os.path.join(XNEMO_WEIGHTS_PATH, "sd-image-variations-diffusers"),
            "SD image variations",
        ),
        (
            os.path.join(XNEMO_WEIGHTS_PATH, "stable-video-diffusion-img2vid-xt/vae"),
            "SVD VAE",
        ),
        (
            os.path.join(XNEMO_WEIGHTS_PATH, "xnemo_denoising_unet.pth"),
            "Denoising UNet",
        ),
        (
            os.path.join(XNEMO_WEIGHTS_PATH, "xnemo_reference_unet.pth"),
            "Reference UNet",
        ),
        (
            os.path.join(XNEMO_WEIGHTS_PATH, "xnemo_motion_encoder.pth"),
            "Motion encoder",
        ),
        (
            os.path.join(XNEMO_WEIGHTS_PATH, "xnemo_temporal_module.pth"),
            "Temporal module",
        ),
    ]

    for path, desc in weights_checks:
        if not check_path(path, desc):
            errors.append(f"X-Nemo weights: {desc} not found")

    print("\n[3] Seed-VC Repository")
    print("-" * 40)

    seedvc_checks = [
        (SEEDVC_REPO_PATH, "Repo root"),
        (os.path.join(SEEDVC_REPO_PATH, "seed_vc_api.py"), "Main API"),
        (os.path.join(SEEDVC_REPO_PATH, "run_seedvc.py"), "Runner script"),
        (os.path.join(SEEDVC_REPO_PATH, "configs/vc_wrapper.yaml"), "Model config"),
        (os.path.join(SEEDVC_REPO_PATH, "src/v2/vc_wrapper.py"), "VC wrapper module"),
    ]

    for path, desc in seedvc_checks:
        if not check_path(path, desc):
            errors.append(f"Seed-VC: {desc} not found")

    print("\n[4] Shared Data Directories")
    print("-" * 40)

    for subdir in ["uploads", "temp", "output"]:
        path = os.path.join(SHARED_DATA_PATH, subdir)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"  + Created: {path}")
        else:
            check_path(path, subdir)

    print("\n[5] Identity Assets")
    print("-" * 40)

    check_path(IDENTITIES_PATH, "Identities directory")

    # Check identity structure
    from app.core.identities import IDENTITIES

    if not IDENTITIES:
        print("  ⚠ No identities found!")
        print("    Add folders with face images + voice audio to:")
        print(f"    {IDENTITIES_PATH}/")
    else:
        for identity_id, data in IDENTITIES.items():
            identity_dir = os.path.join(IDENTITIES_PATH, identity_id)
            check_path(identity_dir, f"Identity: {data['name']}")

            # Check images
            for img in data["images"]:
                img_path = os.path.join(identity_dir, img)
                check_path(img_path, f"  Image: {img}")

            # Check audio
            audio_path = os.path.join(identity_dir, data["audio"])
            check_path(audio_path, f"  Audio: {data['audio']}")

    print("\n[6] Python Imports")
    print("-" * 40)

    try:
        print("  ✓ XNemoRunner import OK")
    except Exception as e:
        print(f"  ✗ XNemoRunner import failed: {e}")
        errors.append(f"Import error: XNemoRunner - {e}")

    try:
        print("  ✓ SeedVCRunner import OK")
    except Exception as e:
        print(f"  ✗ SeedVCRunner import failed: {e}")
        errors.append(f"Import error: SeedVCRunner - {e}")

    try:
        print("  ✓ Database models import OK")
    except Exception as e:
        print(f"  ✗ Database models import failed: {e}")
        errors.append(f"Import error: Database - {e}")

    print("\n" + "=" * 60)

    if errors:
        print(f"VERIFICATION FAILED - {len(errors)} error(s) found:")
        print("-" * 40)
        for error in errors:
            print(f"  • {error}")
        print("\nPlease fix the above issues before running the pipeline.")
        print("\nHints:")
        print(
            "  - Clone X-Nemo: git clone https://github.com/samarrik/x-nemo-inference tools/x-nemo-inference"
        )
        print(
            "  - Clone Seed-VC: git clone https://github.com/samarrik/seed-vc tools/seed-vc"
        )
        print("  - Or set XNEMO_REPO_PATH and SEEDVC_REPO_PATH environment variables")
        return 1
    else:
        print("✓ VERIFICATION PASSED - All checks OK!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
