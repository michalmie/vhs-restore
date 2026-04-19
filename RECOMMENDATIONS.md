# VHS Capture & AI Restoration — Open Source Pipeline 2026

## Part 1: Raw VHS Capture

### Capture Software

| Tool | Best For | Platform |
|------|----------|----------|
| **OBS Studio** | Easiest lossless setup | Win/Mac/Linux |
| **ffmpeg** | Scriptable pipelines, full control | All |
| **vrecord** | macOS + Blackmagic hardware | Mac/Linux |

### Capture Hardware Tiers

- **Budget (~$30-60):** Elgato Video Capture USB — clean output, easy setup
- **Mid-range (~$150):** Hauppauge USB-Live2 — adds Proc Amp controls (brightness/color adjustment hardware-level)
- **Pro (~$300+):** Blackmagic DeckLink Mini Recorder — S-Video + composite, professional quality
- **Archival gold-standard (~$500):** Domesday Duplicator — captures raw RF signal for future re-decoding via vhs-decode (https://github.com/oyvindln/vhs-decode)

### Capture Format: FFV1 10-bit 4:2:2 in MKV (lossless, open standard)

**OBS Studio settings:**
```
Output: Custom FFmpeg → MKV
Video codec: FFV1
Pixel format: yuv422p10le
Audio codec: FLAC
Resolution: 720x480 (NTSC) or 720x576 (PAL)
FPS: 59.94 (NTSC) or 50 (PAL)
Color Space: Rec. 601
```

**ffmpeg capture command (Windows/Linux):**
```bash
ffmpeg -f dshow -i video="USB Device" -f dshow -i audio="USB Device" \
  -c:v ffv1 -level 3 -pix_fmt yuv422p10le \
  -flags +ilme+ildct -top 1 -vf "setfield=tff" \
  -c:a flac -ar 48000 \
  captured_vhs.mkv
```

**Key capture rules:**
- Preserve interlacing — do NOT deinterlace during capture (do it in post)
- Field order: TFF (Top-Field-First) for NTSC VHS
- Color space: Rec. 601 for VHS
- Audio: 48kHz FLAC minimum

---

## Part 2: AI Post-Processing Pipeline

### Processing Order (critical — wrong order ruins quality)

```
1. Deinterlace  →  2. Denoise  →  3. Artifact removal  →  4. Color correction  →  5. Upscale  →  6. Subtle grain
```

---

## Quality Gates & Parameter Tuning

### Why Blind Metrics (No Reference Needed)

VHS has no clean "original" to compare against, so reference-based metrics (PSNR, SSIM) are useless here. Use **no-reference metrics** that score perceived quality from the output alone.

| Metric | Tool | What It Measures | Good For |
|--------|------|-----------------|----------|
| **VMAF** | ffmpeg `-lavfi libvmaf` | Perceptual quality (Netflix standard) | Overall output quality vs. bicubic baseline |
| **NIQE** | Python `piq` / `iqa-pytorch` | Naturalness (lower = better) | Detecting over-processing / plastic look |
| **BRISQUE** | Python `brisque` / `iqa-pytorch` | Blind image quality (lower = better) | Noise, blur, ringing artifacts |
| **SSIM** | ffmpeg `-lavfi ssim` | Structural similarity | Useful when comparing two versions of same clip |

### Stage-by-Stage Quality Gates

#### Gate 1: After Deinterlacing
Check for combing artifacts (horizontal zig-zag on moving edges = wrong field order or failed deinterlace):
```bash
# Visual check — scrub through output in mpv
mpv --deinterlace=no deinterlaced.mkv

# Detect combing with ffmpeg
ffmpeg -i deinterlaced.mkv -vf idet -f null - 2>&1 | grep -E "Single|Multi|Progressive"
# Target: >95% Progressive frames reported
```

**Tune:** If combing persists, flip field order (`-top 0`) or try `preset="Placebo"` in QTGMC.

#### Gate 2: After Denoising
Check you're not over-smoothing (destroying texture = plastic look):
```bash
# BRISQUE score — lower is better, typical good range: 20-40
python3 -c "
import cv2, brisque
frame = cv2.imread('sample_frame.png')
print(brisque.score(frame))
"

# Visual: extract a sample frame before/after and compare sharpness
ffmpeg -i denoised.mkv -vf "select=eq(n\,100)" -frames:v 1 frame_denoised.png
ffmpeg -i deinterlaced.mkv -vf "select=eq(n\,100)" -frames:v 1 frame_before.png
```

**Tune KNLMeansCL `h` parameter:**
- `h=0.8` — light denoising (preserves texture, may leave noise)
- `h=1.2` — balanced (good default for VHS)
- `h=2.0` — aggressive (risk of plastic look, check BRISQUE)

**Rule:** If BRISQUE score *increases* after denoising, you are over-smoothing — reduce `h`.

#### Gate 3: After Upscaling
Compare upscaled output vs. a simple bicubic upscale as baseline:
```bash
# Generate bicubic reference
ffmpeg -i denoised.mkv -vf scale=1440:1080:flags=bicubic bicubic_ref.mkv

# VMAF score: upscaled vs. bicubic (higher = better, target >70)
ffmpeg -i upscaled.mkv -i bicubic_ref.mkv \
  -lavfi "[0:v][1:v]libvmaf=log_fmt=json:log_path=vmaf.json" \
  -f null -

cat vmaf.json | python3 -c "import json,sys; d=json.load(sys.stdin); print('VMAF:', d['pooled_metrics']['vmaf']['mean'])"
```

**Tune Real-ESRGAN model:**
- `RealESRGAN_x4plus` — general, handles mixed content
- `realesr-general-x4v3` — better on real-world degradation (try this first for VHS)
- `RealESRGAN_x4plus-anime` — only for animated/cartoon content

**Rule:** VMAF >70 vs. bicubic = meaningful improvement. If <55, try a different model or reduce scale factor (2x instead of 4x).

#### Gate 4: After Full Pipeline
Final perceptual quality check:
```bash
# NIQE score (lower = more natural, target <4.0 for good restoration)
python3 -c "
import iqa
import torchvision.io as io
video = io.read_video('final_restored.mkv')[0].float() / 255.0
score = iqa.niqe(video)
print('NIQE:', score.item())
"

# Side-by-side visual comparison with mpv
mpv --lavfi-complex="[vid1][vid2]hstack[vo]" \
  --external-file=final_restored.mkv captured_vhs.mkv
```

### Efficient Tuning Workflow (Test Before Full Encode)

Never run full-tape tests when tuning — extract a representative 30-second clip covering: static shot, motion, faces, dark scene.

```bash
# Extract 30-second test clip from minute 5
ffmpeg -ss 00:05:00 -t 30 -i captured_vhs.mkv -c copy test_clip.mkv

# Run your VapourSynth script on just the test clip
vspipe script.vpy - --start 0 --end 750 | ffmpeg -f yuv4mpegpipe -i pipe: test_output.mkv
```

Then score and compare before committing to the full tape.

### Parameter Tuning Cheat Sheet

| Parameter | Conservative | Balanced (default) | Aggressive |
|-----------|-------------|-------------------|------------|
| KNLMeansCL `h` | 0.8 | 1.2 | 2.0 |
| BM3D `sigma` | 1.5 | 3.0 | 6.0 |
| QTGMC `preset` | "Fast" | "Slower" | "Placebo" |
| Real-ESRGAN scale | 2x | 2x–4x | 4x |
| grain `var` | 0.2 | 0.4 | 0.8 |

### Install Quality Metric Tools

```bash
pip install brisque iqa-pytorch piq opencv-python torch torchvision

# ffmpeg must be built with --enable-libvmaf (most package managers include it)
ffmpeg -filters | grep vmaf
```

### Step-by-Step Tools

#### 1. Deinterlacing — QTGMC (best in class)
- Part of VapourSynth ecosystem (https://github.com/vapoursynth/vapoursynth)
- `preset="Slower"`, `FPSDivisor=2` for single-rate output

#### 2. Denoising — KNLMeansCL (primary on GPU) + BM3D (CPU fallback)
- **KNLMeansCL** (recommended for NVIDIA): GPU-accelerated via CUDA, use `device_type="gpu"` — fast and high quality on RTX
- **VapourSynth-BM3D** (CPU): https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D — highest quality but slow; use for final pass if time allows
- `sigma=3.0` is a good starting point for VHS noise level

#### 3. VHS Artifact Removal

| Artifact | Tool |
|----------|------|
| Dot crawl (NTSC) | Checkmate filter |
| Chroma noise/bleed | Cnr2, Bifrost |
| Head switching noise | BM3D + TComb |
| Dropouts | BasicVSR++ interpolation |

#### 4. Color Correction
- DeOldify (https://github.com/jantic/DeOldify) — AI-based color restoration for faded VHS (also colorizes B&W)
- VapourSynth `core.std.Levels(min_in=16, max_in=235)` for basic levels fix
- DaVinci Resolve (free version) for manual grading + export LUT

#### 5. AI Upscaling

| Tool | Speed | Quality | Best For |
|------|-------|---------|----------|
| Real-ESRGAN (https://github.com/xinntao/Real-ESRGAN) | Fast (GPU) | Very good | General use, best speed/quality on RTX |
| RealBasicVSR (https://github.com/ckkelvinchan/RealBasicVSR) | Slow | Best | Noisy/degraded video, highest quality |
| NNEDI3 (https://github.com/dubhater/vapoursynth-nnedi3) | Slowest | Sharpest edges | CPU-only — skip with a GPU available |

**RealBasicVSR** is the top quality pick for VHS — built-in pre-cleaning stage designed for real-world degradation.
**Real-ESRGAN** is the practical pick on RTX 3060 Ti — much faster, still excellent results.

**RTX 3060 Ti (8GB VRAM) specific flags:**

```bash
# Real-ESRGAN: --tile 256 keeps VRAM under 8GB safely
python inference_realesrgan.py -n RealESRGAN_x4plus -i input/ -o output/ -s 2 --tile 256

# RealBasicVSR: add tiling to avoid OOM on 8GB
python inference_realbasicvsr.py --input denoised.mp4 --output upscaled.mp4 --tile 256 256
```

#### 6. Film Grain Synthesis (prevents "plastic" look after AI processing)
- DeepFilmGrain: https://github.com/InterDigitalInc/DeepFilmGrain
- Or VapourSynth: `core.grain.Add(clip, var=0.5, constant=False)`

---

## GUI Options (no-script approach)

| Tool | GitHub | Notes |
|------|--------|-------|
| **VapourBox** | https://github.com/StuartCameronCode/VapourBox | Modern drag-and-drop, 200+ plugins, VHS-focused |
| **Hybrid** | https://github.com/Selur/hybrid-vapoursynth-addon | Most features, wraps VapourSynth/Avisynth, updated 2026 |
| **Advanced Tape Restorer v4.1** | https://github.com/idocinthebox/Advanced-Tape-Restorer | All-in-one: BM3D + QTGMC + Real-ESRGAN + BasicVSR++, MIT licensed |

**Advanced Tape Restorer** is the best single tool if you want everything integrated.

---

## Complete VapourSynth Script Template (RTX 3060 Ti optimized)

```python
import vapoursynth as vs
from vapoursynth import core
import havsfunc as haf

source = core.ffms2.Source("captured_vhs.mkv")

# 1. Deinterlace (QTGMC)
deint = core.qtgmc.QTGMC(source, preset="Slower", FPSDivisor=2)

# 2. Denoise — KNLMeansCL on GPU (CUDA, fast on RTX)
denoised = core.knlm.KNLMeansCL(deint, h=1.2, d=1, device_type="gpu", device_id=0)

# 3. Fix levels (VHS black/white point)
leveled = core.std.Levels(denoised, min_in=16, max_in=235, min_out=0, max_out=255)

# 4. Upscale via Real-ESRGAN (run externally, pipe back) or vs-mlrt on GPU
#    See ffmpeg pipe command below — Real-ESRGAN handles upscaling outside VapourSynth

# 5. Add subtle grain (natural look, avoids plastic AI appearance)
final = core.grain.Add(leveled, var=0.4, constant=False)

final.set_output()
```

```bash
# Full pipeline: VapourSynth (deinterlace + denoise) → Real-ESRGAN (upscale on GPU) → FFV1 archive
vspipe script.vpy - -f yuv4mpegpipe | \
  python inference_realesrgan.py -n RealESRGAN_x4plus --tile 256 -i pipe: -o pipe: | \
  ffmpeg -f yuv4mpegpipe -i pipe: \
    -c:v ffv1 -level 3 -pix_fmt yuv422p10le \
    -c:a copy final_restored.mkv
```

> Note: If direct piping to Real-ESRGAN is not supported, export intermediate as lossless first, then run Real-ESRGAN, then re-encode.

---

## Key Repository Index

| Repo | Purpose |
|------|---------|
| https://github.com/oyvindln/vhs-decode | RF capture & decoding (archival gold standard) |
| https://github.com/Hank-tha-Cowdog/vhs-vapourizer | Full VapourSynth pipeline script |
| https://github.com/idocinthebox/Advanced-Tape-Restorer | All-in-one AI restoration GUI |
| https://github.com/xinntao/Real-ESRGAN | Video upscaling |
| https://github.com/ckkelvinchan/RealBasicVSR | Video super-resolution for degraded footage |
| https://github.com/ckkelvinchan/BasicVSR_PlusPlus | BasicVSR++ backbone |
| https://github.com/JingyunLiang/SwinIR | Transformer-based restoration |
| https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D | Denoising filter |
| https://github.com/StuartCameronCode/VapourBox | Restoration GUI |
| https://github.com/Selur/hybrid-vapoursynth-addon | VapourSynth/Avisynth GUI |
| https://github.com/staxrip/staxrip | Encoding GUI |
| https://github.com/jantic/DeOldify | Color restoration / colorization |
| https://github.com/InterDigitalInc/DeepFilmGrain | Grain synthesis |
| https://github.com/amiaopensource/vrecord | Professional capture tool |

---

## Quick Decision Guide

| Goal | Recommendation |
|------|---------------|
| Just starting out | OBS + Elgato, then VapourBox GUI |
| Best quality capture | ffmpeg + Blackmagic hardware |
| True archival | Domesday Duplicator + vhs-decode |
| Best all-in-one AI restoration | Advanced Tape Restorer v4.1 |
| Maximum quality pipeline | QTGMC → V-BM3D → RealBasicVSR → DeepFilmGrain via VapourSynth |

Processing time on RTX 3060 Ti: ~15-25 minutes per hour of VHS footage (GPU-accelerated denoising + Real-ESRGAN). RealBasicVSR is slower — estimate ~60-90 minutes per hour.

## Hardware Note — RTX 3060 Ti (8GB VRAM)

- Use `--tile 256` for Real-ESRGAN and RealBasicVSR to stay within 8GB VRAM
- Always use `device_type="gpu"` in KNLMeansCL — CUDA path is significantly faster than OpenCL on NVIDIA
- Skip NNEDI3 — it is CPU-only and much slower than GPU alternatives
- NVENC (NVIDIA hardware encoder) can be used for fast intermediate encodes, but use FFV1 for final archival output
