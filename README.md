# VHS Digitization & Restoration

AI-powered pipeline for capturing and restoring VHS tapes.
Produces clean, upscaled, natural-looking output from raw captures.

**Hardware target:** NVIDIA RTX 3060 Ti (8 GB VRAM) · Ubuntu 22.04

---

## Quick start (fresh Ubuntu 22.04 VM)

```bash
# 1. Install GitHub CLI and authenticate
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] \
  https://cli.github.com/packages stable main" \
  | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update && sudo apt install -y gh
gh auth login

# 2. Clone
gh repo clone michalmie/vhs-restore
cd vhs-restore

# 3. Install all dependencies
bash pipeline/setup_ubuntu.sh

# 4. Verify
bash pipeline/verify.sh
```

> If the NVIDIA driver is missing, the script installs it and asks you to reboot — just reboot and re-run step 3.

**Something went wrong?** Full clean reset:
```bash
bash pipeline/setup_ubuntu.sh --clean
bash pipeline/setup_ubuntu.sh
```

---

## Repository layout

```
vhs-restore/
├── README.md                   This file — navigation and quick start
│
├── RECOMMENDATIONS.md          Full research findings: hardware tiers,
│                               capture settings, tool comparison,
│                               VapourSynth script templates
│
└── pipeline/
    ├── README.md               Complete usage guide — all options,
    │                           tuning guide, quality gates, workflows
    ├── restore.py              Main pipeline script (deinterlace →
    │                           denoise → upscale → grain, 4 quality gates)
    ├── trim.py                 Auto-trim trailing blank screen from capture
    ├── setup_ubuntu.sh         Full dependency installer for Ubuntu 22.04
    │                             bash setup_ubuntu.sh          # install
    │                             bash setup_ubuntu.sh --clean  # wipe + reinstall
    ├── verify.sh               Post-install dependency check
    └── requirements.txt        Python dependencies reference
```

---

## Pipeline at a glance

```
captured.mkv
    │
    ├─ [trim.py]          remove trailing blank screen (optional)
    │
    └─ [restore.py]
           Stage 1 · VapourSynth
             auto-detect field order
             interlaced → QTGMC deinterlace + KNLMeansCL denoise
             progressive → KNLMeansCL denoise only (OBS MP4)
             Gate 1: deinterlace check · Gate 2: BRISQUE denoise check
           Stage 2 · Real-ESRGAN
             AI upscaling on GPU (tiled, 8 GB VRAM safe)
             Gate 3: VMAF vs bicubic baseline
           Stage 3 · ffmpeg
             film grain synthesis + final encode (FFV1 / ProRes)
             Gate 4: NIQE naturalness check
             → output.mkv + output.quality_report.json
```

---

## Common commands

```bash
source ~/vhs-env/bin/activate

# Trim blank end from capture
python pipeline/trim.py captured.mkv

# Test restore (30-second clip)
python pipeline/restore.py captured.mkv output.mkv --test

# Full restore
python pipeline/restore.py captured.mkv output.mkv

# Maximum quality (overnight)
python pipeline/restore.py captured.mkv output.mkv \
  --qtgmc-preset Placebo --realesrgan-scale 4 --grain 3

# Quick preview
python pipeline/restore.py captured.mkv preview.mkv \
  --qtgmc-preset Fast --realesrgan-scale 2 --codec prores --test
```

See [`pipeline/README.md`](pipeline/README.md) for all options, quality gate tuning, and processing time estimates.
