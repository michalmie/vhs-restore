# VHS Digitization & Restoration Project

## Project Purpose

This directory is a workspace for digitizing VHS tapes using open-source tools and restoring them with AI post-processing to achieve the best possible quality — natural-looking, clear, de-noised output.

## Scope

- **Phase 1:** Raw VHS capture (lossless, preserve all original data)
- **Phase 2:** AI-based post-processing (deinterlace, denoise, artifact removal, color correction, upscale, grain synthesis)

## Key Documents

- `RECOMMENDATIONS.md` — Full research findings: hardware, software, capture settings, AI restoration pipeline, VapourSynth script templates, and repository index.
- `pipeline/restore.py` — Fully automated Python pipeline (deinterlace → denoise → upscale → grain + 4 quality gates)
- `pipeline/requirements.txt` — Python deps + setup notes for VapourSynth, Real-ESRGAN, ffmpeg

## Technology Stack

- **Capture:** OBS Studio or ffmpeg → FFV1 10-bit 4:2:2 MKV (lossless)
- **Post-processing framework:** VapourSynth (scripting) or Advanced Tape Restorer (GUI)
- **GPU:** NVIDIA RTX 3060 Ti (8GB VRAM) — use CUDA paths, `--tile 256` for upscaling models
- **Deinterlacing:** QTGMC
- **Denoising:** KNLMeansCL on GPU (primary), BM3D (CPU, optional quality pass)
- **Upscaling:** Real-ESRGAN with `--tile 256` (speed), RealBasicVSR (best quality)
- **Grain synthesis:** DeepFilmGrain or VapourSynth grain.Add

## Notes

- Always capture interlaced (deinterlace in post, not during capture)
- Processing pipeline order matters: deinterlace → denoise → artifacts → color → upscale → grain
- Target format for archival output: FFV1 10-bit 4:2:2 MKV
- Research current as of April 2026
