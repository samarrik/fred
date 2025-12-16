"""
Identity management for the Fred pipeline.

Each identity needs:
- One or more face images (.jpg, .jpeg, .png) - for X-Nemo face reenactment
- One voice audio file (.wav) - for Seed-VC voice conversion

Directory structure:
    identities/
    ├── persona_a/
    │   ├── face1.jpg           # Any image files work
    │   ├── face2.png
    │   └── voice.wav           # Single audio file for voice
    └── persona_b/
        ├── photo.jpg
        └── voice.wav
"""

import os
from typing import Optional

# Base path for identity assets - relative to project root by default
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
IDENTITIES_PATH = os.getenv("IDENTITIES_PATH", os.path.join(PROJECT_ROOT, "identities"))

# Supported file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}


def _discover_identities() -> dict:
    """
    Auto-discover identities from the identities folder.

    Each subfolder is an identity. Images and audio are auto-detected.
    """
    identities = {}

    if not os.path.exists(IDENTITIES_PATH):
        return identities

    for folder_name in os.listdir(IDENTITIES_PATH):
        folder_path = os.path.join(IDENTITIES_PATH, folder_name)

        if not os.path.isdir(folder_path):
            continue

        # Skip hidden folders
        if folder_name.startswith("."):
            continue

        # Find all images and audio files
        images = []
        audio = None

        for filename in sorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, filename)
            if not os.path.isfile(file_path):
                continue

            ext = os.path.splitext(filename)[1].lower()

            if ext in IMAGE_EXTENSIONS:
                images.append(filename)
            elif ext in AUDIO_EXTENSIONS and audio is None:
                # Use first audio file found
                audio = filename

        # Only add identity if it has at least one image and audio
        if images and audio:
            # Create display name from folder name
            display_name = folder_name.replace("_", " ").replace("-", " ").title()

            identities[folder_name] = {
                "name": display_name,
                "images": images,
                "audio": audio,
            }

    return identities


# Discover identities on module load
IDENTITIES = _discover_identities()


def refresh_identities():
    """Re-scan the identities folder. Call after adding new identities."""
    global IDENTITIES
    IDENTITIES = _discover_identities()
    return IDENTITIES


def get_identities() -> dict:
    """Return all available identities."""
    return {
        id_: {
            "id": id_,
            "name": data["name"],
            "images": data["images"],
        }
        for id_, data in IDENTITIES.items()
    }


def get_identity(identity_id: str) -> Optional[dict]:
    """Get a single identity by ID."""
    if identity_id not in IDENTITIES:
        return None
    data = IDENTITIES[identity_id]
    return {
        "id": identity_id,
        "name": data["name"],
        "images": data["images"],
        "audio": data["audio"],
    }


def get_identity_image_path(identity_id: str, image_name: str) -> str:
    """Get full path to an identity's image."""
    return os.path.join(IDENTITIES_PATH, identity_id, image_name)


def get_identity_audio_path(identity_id: str) -> Optional[str]:
    """Get full path to an identity's reference audio."""
    data = IDENTITIES.get(identity_id)
    if not data:
        return None
    return os.path.join(IDENTITIES_PATH, identity_id, data["audio"])


def list_identity_status():
    """Print status of all identities (useful for debugging)."""
    print(f"Identities path: {IDENTITIES_PATH}")
    print(f"Found {len(IDENTITIES)} identities:\n")

    for id_, data in IDENTITIES.items():
        print(f"  {id_}/")
        print(f"    Name: {data['name']}")
        print(f"    Images: {', '.join(data['images'])}")
        print(f"    Audio: {data['audio']}")
        print()


if __name__ == "__main__":
    # Run this to check your identity setup
    list_identity_status()
