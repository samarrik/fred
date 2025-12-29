# Fred - Face Reenactment + Voice Conversion Pipeline

Generate videos with identity reenactment (X-Nemo) and voice conversion (Seed-VC).

## How It Works

1. User selects an **identity** (face + voice)
2. User uploads their **video** (provides motion + speech content)
3. System runs **X-Nemo** (face reenactment) + **Seed-VC** (voice conversion)
4. Output: Video with the identity's face animated by user's motion, speaking with identity's voice

## Installation

### Prerequisites

- **Conda** (Miniconda or Anaconda)
- **CUDA-capable GPU** (16GB+ VRAM recommended)
- **Git** for cloning repositories

### Step 1: Clone Repositories

```bash
# Clone main repository
git clone <this-repo-url> fred
cd fred

# Clone ML tool repositories
mkdir -p tools
git clone https://github.com/samarrik/x-nemo-inference tools/x-nemo-inference
git clone https://github.com/samarrik/seed-vc tools/seed-vc
```

### Step 2: Setup Python Environment

```bash
# Create conda environment with Python 3.10
conda create -n fred python=3.10 -y
conda activate fred

# Install PyTorch with CUDA (adjust CUDA version as needed)
# For CUDA 12.1:
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# For CUDA 11.8:
# conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install all Python dependencies (includes ML packages)
pip install -r requirements-ml.txt

# Install Fred package
pip install -e .
```

### Step 3: Install FFmpeg

**On cluster (module system):**
```bash
module load FFmpeg
```

**Local installation:**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Conda (if not using system FFmpeg)
conda install ffmpeg -c conda-forge
```

### Step 4: Download Pretrained Weights

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

### Step 5: Add Identity Assets

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

### Step 6: Configure Paths (Optional)

If you placed the tools in a different location, set environment variables:

```bash
export XNEMO_REPO_PATH=/path/to/x-nemo-inference
export SEEDVC_REPO_PATH=/path/to/seed-vc
export XNEMO_WEIGHTS_PATH=/path/to/x-nemo-inference/pretrained_weights
```

Or create a `.env` file in the fred directory.

### Step 7: Verify Setup

```bash
python scripts/verify_setup.py
```

## Running the Pipeline

### Option A: Gradio GUI (Recommended)

```bash
# Activate environment
conda activate fred

# Load FFmpeg module (if on cluster)
module load FFmpeg  # Skip if running locally with system FFmpeg

# Run GUI
python app/gui.py
```

Open **http://localhost:7860** in your browser.

### Option B: API + Worker (for production)

```bash
# Terminal 1: API server
conda activate fred
module load FFmpeg  # Skip if running locally
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Background worker
conda activate fred
module load FFmpeg  # Skip if running locally
python -m app.workers.job_worker
```

API available at http://localhost:8000

---

## Quick Start (All-in-One)

Complete setup script:

```bash
# 1. Clone repositories
git clone <this-repo> fred && cd fred
mkdir -p tools
git clone https://github.com/samarrik/x-nemo-inference tools/x-nemo-inference
git clone https://github.com/samarrik/seed-vc tools/seed-vc

# 2. Setup Python environment
conda create -n fred python=3.10 -y
conda activate fred
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements-ml.txt
pip install -e .

# 3. Install FFmpeg
# On cluster:
module load FFmpeg
# Or locally:
# sudo apt install ffmpeg  # Ubuntu/Debian
# brew install ffmpeg      # macOS

# 4. Download weights
python scripts/download_weights.py

# 5. Add identities to identities/ folder (see Step 5 above)

# 6. Run
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

**On cluster:**
```bash
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

**Verify FFmpeg is working:**
```bash
ffmpeg -version
```

---

## License

Apache 2.0 for code. Model weights have their own licenses - check X-NeMo and Seed-VC repos.
