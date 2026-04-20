# VHS Restoration Pipeline

AI-powered post-processing pipeline for digitised VHS footage.
Converts a raw capture (MKV/MP4) into a clean, upscaled, natural-looking output.

**Hardware target:** NVIDIA RTX 3060 Ti (8 GB VRAM) · Ubuntu 22.04

---

## Pipeline overview

```
Input (MKV / MP4)
    │
    ▼
[trim.py]  ←  optional: remove trailing blank screen
    │
    ▼
[restore.py]
    ├─ Stage 1 · VapourSynth
    │      auto-detect field order
    │      ├─ interlaced → QTGMC deinterlace + KNLMeansCL denoise + levels
    │      └─ progressive → KNLMeansCL denoise + levels  (OBS MP4 path)
    │      Quality gate 1: deinterlace check (idet)      ← skipped if progressive
    │      Quality gate 2: denoise check (BRISQUE)
    │
    ├─ Stage 2 · Real-ESRGAN
    │      AI upscaling on GPU, tiled for 8 GB VRAM
    │      Quality gate 3: VMAF vs bicubic baseline
    │
    └─ Stage 3 · ffmpeg
           film grain synthesis + final encode
           Quality gate 4: naturalness check (NIQE)
           → output.mkv  +  output.quality_report.json
```

---

## 1. Installation (Ubuntu 22.04 — fresh VM)

### 1.1 Install GitHub CLI and authenticate

```bash
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] \
  https://cli.github.com/packages stable main" \
  | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update && sudo apt install -y gh
gh auth login
```

When prompted: choose **GitHub.com → HTTPS → Login with a web browser**, paste the one-time code shown.

### 1.2 Clone the repository

```bash
gh repo clone michalmie/vhs-restore
cd vhs-restore
```

### 1.3 Run the setup script

```bash
bash pipeline/setup_ubuntu.sh
```

The script installs everything in order and skips steps that are already complete — safe to re-run.

> **NVIDIA driver not installed?** The script detects this, installs driver 535, and exits with a reboot prompt. After rebooting, re-run the same command.

### 1.4 Verify the installation

```bash
bash pipeline/verify.sh
```

Expected output:

```
GPU         ✓ NVIDIA GeForce RTX 3060 Ti, driver 535.x, 8192 MiB
ffmpeg      ✓ ffmpeg with libvmaf: present
vspipe      ✓ VapourSynth rXX
Python      ✓ 3.12.x  venv: active
VapourSynth ✓ vapoursynth rXX
            ✓ ffms2: loaded
            ✓ knlmeanscl: loaded
            ✓ mvtools: loaded
            ✓ havsfunc (QTGMC): imported
PyTorch     ✓ 2.x.x  CUDA: NVIDIA GeForce RTX 3060 Ti (8 GB)
            ✓ piq (BRISQUE/NIQE): imported
Real-ESRGAN ✓ inference_realesrgan.py found
            ✓ Model: realesr-general-x4v3.pth
Pipeline    ✓ restore.py --help: OK
```

### 1.5 Clean reset (if something went wrong)

If a previous setup attempt left the system in a bad state, wipe everything and start fresh:

```bash
bash pipeline/setup_ubuntu.sh --clean
bash pipeline/setup_ubuntu.sh
```

This removes the venv, VapourSynth, all plugins, and Real-ESRGAN, then reinstalls from scratch. No manual commands needed.

### 1.6 What the setup script installs

| Step | What | Notes |
|------|------|-------|
| 1 | System packages | build tools, ffmpeg, Python 3.12, OpenCL, FFTW3, Boost |
| 2 | NVIDIA driver + CUDA | RTX 3060 Ti needs driver ≥ 525 |
| 3 | ffmpeg with libvmaf | uses jellyfin-ffmpeg7 if system ffmpeg lacks libvmaf |
| 4 | Python 3.12 venv | created at `~/vhs-env` |
| 5 | VapourSynth core | built from source (R74+, requires Python ≥ 3.12) |
| 6 | VapourSynth plugins | havsfunc (QTGMC), ffms2, MVTools, KNLMeansCL — built from source |
| 7 | PyTorch + Real-ESRGAN | cu118 for CUDA 11.x, cu121 for CUDA 12.x; model weights downloaded |
| 8 | Pipeline Python deps | `piq` (BRISQUE/NIQE quality metrics) |

### 1.7 Manual plugin notes

If a plugin build fails, you can place the `.so` file manually in `~/.local/lib/vapoursynth/`:

| Plugin | Source |
|--------|--------|
| KNLMeansCL | https://github.com/Khanattila/KNLMeansCL |
| MVTools | https://github.com/dubhater/vapoursynth-mvtools |
| ffms2 | system: `/usr/lib/x86_64-linux-gnu/libffms2.so` |

```bash
mkdir -p ~/.local/lib/vapoursynth
cp downloaded_plugin.so ~/.local/lib/vapoursynth/
```

---

## 2. Trim trailing blank screen (optional pre-step)

VHS recordings often end with a blank screen when the recorder stops. Trim before restoring — it is faster and cheaper to trim the raw capture.

```bash
source ~/vhs-env/bin/activate

# Auto-detect and trim (saves input_trimmed.mkv alongside the original)
python pipeline/trim.py captured.mkv

# Preview only — see detected cut point without writing anything
python pipeline/trim.py captured.mkv --preview

# Explicit output path
python pipeline/trim.py captured.mkv clean.mkv

# Stricter detection (catches faint grey screens, not just pure black)
python pipeline/trim.py captured.mkv --threshold 0.05

# Adjust minimum blank run length (default 0.5s)
python pipeline/trim.py captured.mkv --min-duration 1.0
```

The script only trims if blank frames are found at the **end** of the file; mid-video fade-to-blacks are left untouched.

---

## 3. Restore a video

### 3.1 Activate the environment

```bash
source ~/vhs-env/bin/activate
```

### 3.2 Quick-start — recommended first run

Always tune on a short clip before committing to a full tape:

```bash
# Step 1: test on a 30-second clip from minute 5
python pipeline/restore.py captured.mkv output.mkv --test

# Step 2: review the quality report
cat output.quality_report.json

# Step 3: if all gates pass, run the full tape
python pipeline/restore.py captured.mkv output.mkv
```

### 3.3 Field order auto-detection

The pipeline automatically detects whether the source is interlaced or already progressive **before processing starts**:

```
Field order detected: progressive
  → Skipping QTGMC (source already progressive — likely OBS MP4)

Field order detected: tt
  → Running QTGMC deinterlace (top-field-first)
```

| Source | Detected as | QTGMC | Gate 1 |
|--------|-------------|-------|--------|
| MKV captured with ffmpeg (raw interlaced) | `tt` or `bb` | runs | checked |
| MP4 from OBS (OBS deinterlaces by default) | `progressive` | skipped | skipped |
| Source with ambiguous metadata | `unknown` | runs (safe default) | checked |

To force a specific behaviour:

```bash
# Force skip QTGMC (progressive source)
python pipeline/restore.py input.mp4 output.mkv --skip-deinterlace

# Verify what your file reports before running
ffprobe -v error -select_streams v:0 \
  -show_entries stream=field_order \
  -of default=noprint_wrappers=1:nokey=1 input.mp4
```

---

## 4. All options

```
python pipeline/restore.py input output [options]
```

### Deinterlacing

| Option | Default | Description |
|--------|---------|-------------|
| `--qtgmc-preset` | `Slower` | Quality preset: `Draft` / `Fast` / `Medium` / `Slow` / `Slower` / `Placebo` |
| `--skip-deinterlace` | auto | Skip QTGMC (auto-set if source is progressive) |

QTGMC presets — speed vs quality tradeoff:

| Preset | Speed | Quality | Use when |
|--------|-------|---------|----------|
| `Draft` | fastest | lowest | quick test only |
| `Fast` | fast | acceptable | rough preview |
| `Slower` | slow | very good | **default, recommended** |
| `Placebo` | slowest | best | maximum quality, overnight runs |

### Denoising

| Option | Default | Description |
|--------|---------|-------------|
| `--knlm-h` | `1.2` | Denoise strength: `0.8`=light · `1.2`=balanced · `2.0`=aggressive |
| `--knlm-d` | `1` | Temporal radius — frames looked ahead/behind for matching |
| `--gpu` | `0` | GPU device ID (0 = first GPU) |

Rule of thumb: if Gate 2 (BRISQUE) fails, reduce `--knlm-h` by 0.4 and re-test.

### Color correction

| Option | Default | Description |
|--------|---------|-------------|
| `--levels-min-in` | `16` | Black point — standard for VHS (video levels) |
| `--levels-max-in` | `235` | White point — standard for VHS (video levels) |

Only change these if your tape was captured at full range (0–255):

```bash
# Full-range capture (rare)
python pipeline/restore.py input.mkv output.mkv --levels-min-in 0 --levels-max-in 255
```

### Upscaling (Real-ESRGAN)

| Option | Default | Description |
|--------|---------|-------------|
| `--realesrgan-model` | `realesr-general-x4v3` | AI model (see table below) |
| `--realesrgan-scale` | `2` | Scale factor: `2` (720p→1440p) or `4` (720p→2880p) |
| `--realesrgan-tile` | `256` | Tile size for 8 GB VRAM; `0` = disable tiling (needs >8 GB) |
| `--realesrgan-dir` | `~/Real-ESRGAN` | Path to cloned Real-ESRGAN repository |

Model selection:

| Model | Best for | Notes |
|-------|---------|-------|
| `realesr-general-x4v3` | VHS footage — **recommended** | Handles real-world degradation best |
| `RealESRGAN_x4plus` | General video | Good fallback if x4v3 VMAF gate fails |
| `RealESRGAN_x4plus-anime` | Animated / cartoon content only | Do not use on live-action VHS |

### Output

| Option | Default | Description |
|--------|---------|-------------|
| `--grain` | `4` | Film grain strength: `1`=subtle · `4`=natural · `8`=heavy |
| `--codec` | `ffv1` | Output codec: `ffv1` (lossless archival) or `prores` (post-production) |
| `--keep-intermediates` | off | Keep work files in `.vhs_work_<name>/` for inspection |

### Quality gate thresholds

| Option | Default | Gate | Meaning |
|--------|---------|------|---------|
| `--gate-progressive-pct` | `0.95` | Gate 1 | Min % progressive frames after deinterlace |
| `--gate-brisque-delta` | `5.0` | Gate 2 | Max allowed BRISQUE score increase after denoise |
| `--gate-vmaf` | `65.0` | Gate 3 | Min VMAF score vs bicubic upscale |
| `--gate-niqe` | `6.0` | Gate 4 | Max NIQE score on final output (lower = more natural) |

Gates print `✓ PASSED`, `✗ FAILED`, or `~ SKIPPED` (when the tool is not available).
A failed gate does not stop the pipeline — it logs a specific tuning suggestion and continues.

### Test mode

| Option | Default | Description |
|--------|---------|-------------|
| `--test` | off | Process a 30-second clip only (fast tuning iterations) |
| `--test-sample` | off | Auto-pick clip from the middle of the video (probes duration; overrides `--test-start`) |
| `--test-start` | `00:05:00` | Where in the file the test clip starts (ignored when `--test-sample` is set) |
| `--test-duration` | `30` | Test clip length in seconds |

### Verbosity

| Option | Description |
|--------|-------------|
| `-v` / `--verbose` | Show all subprocess commands and debug output |

---

## 5. Quality report

Every run saves a JSON report alongside the output file:

```
output.mkv
output.quality_report.json
```

Example report:

```json
{
  "input": "/home/user/captured.mkv",
  "output": "/home/user/output.mkv",
  "config": { "qtgmc_preset": "Slower", "knlm_h": 1.2, "..." : "..." },
  "gates": [
    {
      "gate": "deinterlace",
      "progressive_pct": 0.981,
      "tff_frames": 10,
      "bff_frames": 0,
      "threshold": 0.95,
      "passed": true
    },
    {
      "gate": "denoise",
      "brisque_before": 38.4,
      "brisque_after": 36.1,
      "delta": -2.3,
      "threshold_delta": 5.0,
      "passed": true
    },
    {
      "gate": "upscale",
      "vmaf_vs_bicubic": 72.4,
      "threshold": 65.0,
      "passed": true
    },
    {
      "gate": "final",
      "niqe_score": 4.12,
      "threshold": 6.0,
      "passed": true
    }
  ],
  "summary": { "gates_passed": 4, "gates_total": 4 }
}
```

---

## 6. Tuning guide

### Recommended workflow

```bash
# 1. Extract a representative test clip (static + motion + faces + dark scene)
python pipeline/restore.py input.mkv out_test.mkv --test --test-start 00:10:00

# 2. Tune one parameter at a time, re-test after each change
python pipeline/restore.py input.mkv out_test.mkv --test --knlm-h 0.8
python pipeline/restore.py input.mkv out_test.mkv --test --knlm-h 2.0

# 3. When all gates pass consistently, run the full tape
python pipeline/restore.py input.mkv output_final.mkv
```

### What each failed gate tells you

| Gate | Failure meaning | Fix |
|------|----------------|-----|
| Gate 1: deinterlace | Combing artifacts remain | Try `--qtgmc-preset Placebo`; check field order with ffprobe |
| Gate 2: denoise | Over-smoothing (plastic look) | Reduce `--knlm-h` by 0.4 |
| Gate 3: upscale | AI added no real improvement | Try `--realesrgan-model RealESRGAN_x4plus`; or reduce `--realesrgan-scale 2` |
| Gate 4: final | Output looks artificial | Reduce `--grain`; lower `--knlm-h`; try lighter QTGMC preset |

### Parameter cheat sheet

| Parameter | Conservative | Balanced (default) | Aggressive |
|-----------|-------------|-------------------|------------|
| `--knlm-h` | `0.8` | `1.2` | `2.0` |
| `--qtgmc-preset` | `Fast` | `Slower` | `Placebo` |
| `--realesrgan-scale` | `2` | `2` | `4` |
| `--grain` | `2` | `4` | `8` |

### Processing time estimates (RTX 3060 Ti)

| Stage | Time per hour of footage |
|-------|------------------------|
| Stage 1 (QTGMC + KNLMeansCL) | ~20–40 min |
| Stage 2 (Real-ESRGAN 2x) | ~15–25 min |
| Stage 2 (Real-ESRGAN 4x) | ~60–90 min |
| Stage 3 (grain + encode) | ~5 min |

---

## 7. Common workflows

### Raw MKV capture (interlaced, from ffmpeg/vrecord)

```bash
# Trim blank end, then restore
python pipeline/trim.py captured.mkv
python pipeline/restore.py captured_trimmed.mkv restored.mkv
```

### OBS MP4 capture (progressive, already deinterlaced by OBS)

```bash
# Field order is auto-detected — QTGMC is skipped automatically
python pipeline/restore.py obs_recording.mp4 restored.mkv

# Or force it explicitly to skip the detection probe
python pipeline/restore.py obs_recording.mp4 restored.mkv --skip-deinterlace
```

### Maximum quality (overnight run)

```bash
python pipeline/restore.py captured.mkv restored_hq.mkv \
  --qtgmc-preset Placebo \
  --knlm-h 1.5 \
  --realesrgan-model realesr-general-x4v3 \
  --realesrgan-scale 4 \
  --grain 3 \
  --codec ffv1
```

### Quick preview / fast turnaround

```bash
python pipeline/restore.py captured.mkv preview.mkv \
  --qtgmc-preset Fast \
  --knlm-h 0.8 \
  --realesrgan-scale 2 \
  --codec prores \
  --test
```

### Keep intermediates for inspection

```bash
python pipeline/restore.py captured.mkv output.mkv \
  --keep-intermediates -v

# Work files saved in .vhs_work_output/
#   s1_vs.mkv        — after deinterlace + denoise
#   s2_upscaled.mkv  — after Real-ESRGAN
#   frames_in/       — extracted PNG frames (pre-upscale)
#   frames_out/      — upscaled PNG frames
#   g2_before.png    — sample frame for BRISQUE gate
#   g3_bicubic.mkv   — bicubic reference for VMAF gate
```

---

## 8. Files reference

```
pipeline/
├── restore.py        Main pipeline script
├── trim.py           Trailing blank-screen removal
├── setup_ubuntu.sh   Full dependency installer (Ubuntu 22.04)
│                       bash setup_ubuntu.sh           # install / resume
│                       bash setup_ubuntu.sh --clean   # wipe and reinstall
├── verify.sh         Post-install verification
├── requirements.txt  pip dependencies + notes on external tools
└── README.md         This file
```
