# Fred - Face Reenactment + Voice Conversion Pipeline

Generate videos with identity reenactment (X-Nemo) and voice conversion (Seed-VC).

## How It Works

1. User selects an **identity** (face + voice)
2. User uploads their **video** (provides motion + speech content)
3. System runs **X-Nemo** (face reenactment) + **Seed-VC** (voice conversion)
4. Output: Video with the identity's face animated by user's motion, speaking with identity's voice

## Quick Start

### 1. Prerequisites

- **Conda** (Miniconda or Anaconda)
- **CUDA-capable GPU** (16GB+ VRAM recommended)
- **FFmpeg** - Load via module system on cluster: `module load FFmpeg`
- **Git** for cloning repositories

### 2. Clone This Repository

```bash
git clone <this-repo-url> fred
cd fred
```

### 3. Clone ML Tool Repositories

Clone the X-Nemo and Seed-VC repos into a `tools/` directory:

```bash
# Create tools directory (inside fred or anywhere you prefer)
mkdir -p tools

# X-Nemo - Face reenactment
git clone https://github.com/samarrik/x-nemo-inference tools/x-nemo-inference

# Seed-VC - Voice conversion
git clone https://github.com/samarrik/seed-vc tools/seed-vc
```

### 4. Create Conda Environment

```bash
# Create environment with Python 3.10
conda create -n fred python=3.10 -y
conda activate fred

# Install PyTorch with CUDA (adjust CUDA version as needed)
# For CUDA 12.1:
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# For CUDA 11.8:
# conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### 5. Install Dependencies

```bash
cd fred
conda activate fred

# Install all ML dependencies
pip install -r requirements-ml.txt

# Install Fred package
pip install -e .
```

### 6. Download Pretrained Weights

#### One-Command Download

```bash
# Download HuggingFace models (SD Image Variations + Stable Video Diffusion)
python scripts/download_weights.py
```

This downloads ~20GB of model weights to `tools/x-nemo-inference/pretrained_weights/`.

#### X-NeMo Custom Weights (Manual)

The X-NeMo specific weights (`xnemo_*.pth`) must be downloaded separately from the X-NeMo release:

1. Visit the [X-NeMo repository](https://github.com/bytedance/x-nemo-inference)
2. Download the following files to `tools/x-nemo-inference/pretrained_weights/`:
   - `xnemo_denoising_unet.pth`
   - `xnemo_reference_unet.pth`
   - `xnemo_motion_encoder.pth`
   - `xnemo_temporal_module.pth`

#### Expected Structure

```
tools/x-nemo-inference/pretrained_weights/
├── sd-image-variations-diffusers/    # Downloaded automatically
├── stable-video-diffusion-img2vid-xt/ # Downloaded automatically
│   └── vae/
├── xnemo_denoising_unet.pth          # Manual download
├── xnemo_reference_unet.pth          # Manual download
├── xnemo_motion_encoder.pth          # Manual download
└── xnemo_temporal_module.pth         # Manual download
```

#### Seed-VC Weights

Seed-VC weights are **downloaded automatically** from HuggingFace on first run.

### 7. Add Identity Assets

Create folders in `identities/` with face images and a voice sample:

```
identities/
├── Person One/
│   ├── photo1.jpg           # Face images (.jpg, .png, .webp)
│   ├── photo2.jpg
│   └── voice.wav            # Voice sample (.wav, .mp3, .flac) - 5-25 seconds
└── Person Two/
    ├── headshot.png
    └── speaking.mp3
```

**Requirements:**
- **Face images**: Clear frontal face, good lighting, 512x512+ resolution
- **Voice audio**: 5-25 seconds of clear speech, minimal background noise

**Verify identities are discovered:**
```bash
python -c "from app.core.identities import list_identity_status; list_identity_status()"
```

### 8. Configure Paths (if needed)

If you placed the tools in a different location, set environment variables:

```bash
export XNEMO_REPO_PATH=/path/to/x-nemo-inference
export SEEDVC_REPO_PATH=/path/to/seed-vc
export XNEMO_WEIGHTS_PATH=/path/to/x-nemo-inference/pretrained_weights
```

Or create a `.env` file in the fred directory.

### 9. Verify Setup

```bash
python scripts/verify_setup.py
```

### 10. Run the Pipeline

**Note:** On cluster systems, load FFmpeg module first:
```bash
module load FFmpeg
```

#### Option A: Gradio GUI (Recommended)

```bash
conda activate fred
python app/gui.py
```

Open **http://localhost:7860** in your browser.

#### Option B: API + Worker (for production)

```bash
# Load FFmpeg module (if on cluster)
module load FFmpeg

# Terminal 1: Database (optional, uses SQLite by default)
# ./scripts/start_postgres.sh

# Terminal 2: API server
conda activate fred
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 3: Background worker
conda activate fred
python -m app.workers.job_worker
```

API available at http://localhost:8000

---

## Quick Start Summary

```bash
# Clone everything
git clone <this-repo> fred && cd fred
mkdir -p tools
git clone https://github.com/samarrik/x-nemo-inference tools/x-nemo-inference
git clone https://github.com/samarrik/seed-vc tools/seed-vc

# Setup conda environment
conda create -n fred python=3.10 -y && conda activate fred
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install all dependencies
pip install -r requirements-ml.txt
pip install -e .

# Download weights (see step 6)
python scripts/download_weights.py

# Add identities to identities/ folder (see step 7)

# Load FFmpeg module (if on cluster)
module load FFmpeg

# Run
python app/gui.py
```

---

## Gradio GUI Usage

1. **Select Identity**: Choose from discovered identities
2. **Select Face Image**: Pick which face image to animate
3. **Upload Video**: Your video provides motion + speech content
4. **Generate**: Click to process (2-5 minutes typical)
5. **Download**: Save the generated video

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/identities` | GET | List available identities |
| `/api/identities/{id}` | GET | Get identity details |
| `/api/upload` | POST | Upload user video |
| `/api/jobs` | POST | Create processing job |
| `/api/jobs/{id}/progress` | GET | Poll job progress |
| `/api/jobs/{id}` | GET | Get job details |
| `/api/library` | GET | List completed jobs |
| `/health` | GET | Health check |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `XNEMO_REPO_PATH` | `./tools/x-nemo-inference` | Path to X-Nemo repo |
| `XNEMO_WEIGHTS_PATH` | `{XNEMO_REPO_PATH}/pretrained_weights` | X-Nemo weights |
| `SEEDVC_REPO_PATH` | `./tools/seed-vc` | Path to Seed-VC repo |
| `IDENTITIES_PATH` | `./identities` | Identity assets folder |
| `SHARED_DATA_PATH` | `./shared_data` | Uploads, temp, output |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `EXECUTION_MODE` | `sequential` | `sequential` or `parallel` |

---

## Project Structure

```
fred/
├── app/
│   ├── api/routes.py          # REST API endpoints
│   ├── core/identities.py     # Identity management (auto-discovery)
│   ├── db/                    # Database models
│   ├── workers/
│   │   ├── job_worker.py      # Background job processor
│   │   ├── xnemo_runner.py    # X-Nemo subprocess wrapper
│   │   └── seedvc_runner.py   # Seed-VC subprocess wrapper
│   ├── gui.py                 # Gradio GUI
│   └── main.py                # FastAPI app
├── tools/                     # ML tool repos (you clone these)
│   ├── x-nemo-inference/
│   └── seed-vc/
├── identities/                # Your identity assets
│   └── {Name}/
│       ├── *.jpg/png          # Face images
│       └── *.wav/mp3          # Voice sample
├── shared_data/
│   ├── uploads/               # User uploaded videos
│   ├── temp/                  # Processing temp files
│   └── output/                # Generated videos
├── scripts/
│   ├── verify_setup.py        # Check your setup
│   └── run_gui.sh
└── pyproject.toml
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface                          │
│              (Gradio GUI or REST API)                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Processing                               │
│                                                             │
│  ┌─────────────────┐         ┌─────────────────┐           │
│  │    X-Nemo       │         │    Seed-VC      │           │
│  │  (subprocess)   │         │  (subprocess)   │           │
│  │                 │         │                 │           │
│  │ User Video ──►  │         │ User Audio ──►  │           │
│  │ + Identity Face │         │ + Identity Voice│           │
│  │     ▼           │         │     ▼           │           │
│  │ Reenacted Video │         │ Converted Audio │           │
│  └────────┬────────┘         └────────┬────────┘           │
│           │                           │                     │
│           └───────────┬───────────────┘                     │
│                       ▼                                     │
│              FFmpeg Combine                                 │
│                       ▼                                     │
│              Final Output Video                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### "No module named torch"
Activate the conda environment:
```bash
conda activate fred
```

### "CUDA not available"
Check PyTorch CUDA:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```
Reinstall PyTorch with correct CUDA version if needed.

### "X-Nemo/Seed-VC runner script not found"
Check that repos are cloned and paths are correct:
```bash
ls tools/x-nemo-inference/run_xnemo.py
ls tools/seed-vc/run_seedvc.py
```

If in different location, set `XNEMO_REPO_PATH` and `SEEDVC_REPO_PATH`.

### "No identities found"
Each identity folder needs BOTH images AND audio:
```bash
python -c "from app.core.identities import list_identity_status; list_identity_status()"
```

### CUDA out of memory
- Default `EXECUTION_MODE=sequential` runs models one at a time
- Reduce video length or use `--max-frames`
- Need 16GB+ VRAM for comfortable processing

### FFmpeg not found

**On cluster (module system):**
```bash
# Load FFmpeg module before running
module load FFmpeg
```

**Local installation:**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Conda
conda install ffmpeg -c conda-forge
```

---

## License

Apache 2.0 for code. Model weights have their own licenses - check X-NeMo and Seed-VC repos.
