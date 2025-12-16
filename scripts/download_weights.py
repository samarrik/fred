#!/usr/bin/env python3
"""
Download all required pretrained weights for the Fred pipeline.

Usage:
    python scripts/download_weights.py

This will download:
    - X-NeMo weights (denoising unet, reference unet, motion encoder, temporal module)
    - Stable Diffusion Image Variations
    - Stable Video Diffusion VAE
"""

import argparse
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def download_weights(xnemo_repo_path: str, force: bool = False):
    """Download all required weights."""

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import snapshot_download

    weights_dir = os.path.join(xnemo_repo_path, "pretrained_weights")
    os.makedirs(weights_dir, exist_ok=True)

    print("=" * 60)
    print("Downloading Pretrained Weights")
    print("=" * 60)
    print(f"Target directory: {weights_dir}")
    print()

    # 1. Download SD Image Variations
    sd_variations_path = os.path.join(weights_dir, "sd-image-variations-diffusers")
    if not os.path.exists(sd_variations_path) or force:
        print("[1/4] Downloading SD Image Variations from lambdalabs...")
        snapshot_download(
            repo_id="lambdalabs/sd-image-variations-diffusers",
            local_dir=sd_variations_path,
            local_dir_use_symlinks=False,
        )
        print("      ✓ Done")
    else:
        print("[1/4] SD Image Variations already exists, skipping...")

    # 2. Download Stable Video Diffusion (for VAE)
    svd_path = os.path.join(weights_dir, "stable-video-diffusion-img2vid-xt")
    if not os.path.exists(svd_path) or force:
        print("[2/4] Downloading Stable Video Diffusion from stabilityai...")
        snapshot_download(
            repo_id="stabilityai/stable-video-diffusion-img2vid-xt",
            local_dir=svd_path,
            local_dir_use_symlinks=False,
        )
        print("      ✓ Done")
    else:
        print("[2/4] Stable Video Diffusion already exists, skipping...")

    # 3. Download X-NeMo weights from the X-NeMo repo or a mirror
    xnemo_weights = [
        "xnemo_denoising_unet.pth",
        "xnemo_reference_unet.pth",
        "xnemo_motion_encoder.pth",
        "xnemo_temporal_module.pth",
    ]

    print("[3/4] Checking X-NeMo weights...")
    missing_xnemo = []
    for weight_file in xnemo_weights:
        weight_path = os.path.join(weights_dir, weight_file)
        if not os.path.exists(weight_path):
            missing_xnemo.append(weight_file)
        else:
            print(f"      ✓ {weight_file} exists")

    if missing_xnemo:
        print()
        print("      ⚠ Missing X-NeMo weights:")
        for f in missing_xnemo:
            print(f"        - {f}")
        print()
        print(
            "      X-NeMo weights must be downloaded manually from the X-NeMo release."
        )
        print("      See: https://github.com/bytedance/x-nemo-inference")
        print()
        print("      Expected location:")
        print(f"        {weights_dir}/")
        for f in xnemo_weights:
            print(f"          - {f}")

    # 4. Summary
    print()
    print("[4/4] Verifying installation...")
    all_good = True

    checks = [
        (sd_variations_path, "SD Image Variations"),
        (os.path.join(svd_path, "vae"), "SVD VAE"),
    ]
    for weight_file in xnemo_weights:
        checks.append((os.path.join(weights_dir, weight_file), weight_file))

    for path, name in checks:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"      {status} {name}")
        if not exists:
            all_good = False

    print()
    print("=" * 60)
    if all_good:
        print("✓ All weights downloaded successfully!")
    else:
        print("⚠ Some weights are missing. See above for details.")
    print("=" * 60)

    return all_good


def main():
    parser = argparse.ArgumentParser(description="Download pretrained weights")
    parser.add_argument(
        "--xnemo-path",
        default=os.path.join(PROJECT_ROOT, "tools", "x-nemo-inference"),
        help="Path to X-NeMo repository",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files exist",
    )
    args = parser.parse_args()

    # Check if XNEMO_REPO_PATH env var is set
    xnemo_path = os.getenv("XNEMO_REPO_PATH", args.xnemo_path)

    if not os.path.exists(xnemo_path):
        print(f"Error: X-NeMo repository not found at {xnemo_path}")
        print()
        print("Please clone it first:")
        print(f"  git clone https://github.com/samarrik/x-nemo-inference {xnemo_path}")
        sys.exit(1)

    success = download_weights(xnemo_path, args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
