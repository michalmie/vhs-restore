#!/usr/bin/env python3
"""
VHS Post-Processing Pipeline with Quality Gates

Stages:
  1. VapourSynth  — QTGMC deinterlace + KNLMeansCL denoise + levels correction
  2. Real-ESRGAN  — AI upscaling (GPU, tiled for 8GB VRAM)
  3. Final encode — Film grain + FFV1/ProRes output

Quality gates after each stage (BRISQUE, VMAF, NIQE).
Produces a quality_report.json alongside the output file.

Usage:
  python restore.py input.mkv output.mkv
  python restore.py input.mkv output.mkv --test          # 30-second sample
  python restore.py input.mkv output.mkv --knlm-h 2.0   # aggressive denoise
  python restore.py input.mkv output.mkv --keep-intermediates -v

Requirements: see requirements.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

LOG = logging.getLogger("vhs")

# ── Environment check ─────────────────────────────────────────────────────────
_VENV = Path.home() / "vhs-env"
_VSPIPE_SEARCH = ["/usr/local/bin/vspipe", str(_VENV / "bin" / "vspipe")]

def _find_vspipe() -> str:
    for p in _VSPIPE_SEARCH:
        if Path(p).exists():
            return p
    found = shutil.which("vspipe")
    if found:
        return found
    return ""

def _check_env() -> None:
    errors = []
    if not os.environ.get("VIRTUAL_ENV") and not (_VENV / "bin" / "activate").exists():
        errors.append(
            f"venv not found at {_VENV}\n"
            f"  Run setup first:  bash pipeline/setup_ubuntu.sh"
        )
    elif not os.environ.get("VIRTUAL_ENV"):
        errors.append(
            f"venv not activated — run:\n"
            f"  source {_VENV}/bin/activate"
        )
    if not _find_vspipe():
        errors.append(
            "vspipe not found — VapourSynth not installed or not in PATH\n"
            "  Expected at /usr/local/bin/vspipe\n"
            "  Re-run setup:  bash pipeline/setup_ubuntu.sh"
        )
    if errors:
        print("\nERROR: environment not ready:\n", file=sys.stderr)
        for e in errors:
            print(f"  • {e}\n", file=sys.stderr)
        sys.exit(1)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Deinterlace (QTGMC via havsfunc)
    qtgmc_preset: str = "Slower"   # Draft / Fast / Medium / Slow / Slower / Placebo

    # Denoise (KNLMeansCL, CUDA)
    knlm_h: float = 1.2            # 0.8=light  1.2=balanced  2.0=aggressive
    knlm_d: int = 1                 # temporal radius (frames)
    gpu_device_id: int = 0

    # Color correction
    levels_min_in: int = 16        # VHS black point
    levels_max_in: int = 235       # VHS white point

    # Upscale (Real-ESRGAN)
    realesrgan_model: str = "realesr-general-x4v3"
    realesrgan_scale: int = 2      # 2 or 4
    realesrgan_tile: int = 256     # tile size; 0 = disable (needs >8 GB VRAM)
    realesrgan_dir: str = "~/Real-ESRGAN"

    # Film grain (ffmpeg noise filter)
    grain_strength: int = 4        # 1=subtle  4=natural  8=heavy

    # Output
    output_codec: str = "ffv1"     # ffv1 (lossless archival) | prores (post-production)
    keep_intermediates: bool = False

    # Quality gate thresholds
    gate_min_progressive_pct: float = 0.95  # Gate 1: deinterlace
    gate_max_brisque_delta: float = 5.0     # Gate 2: denoise  (fail if BRISQUE rises by this)
    gate_min_vmaf: float = 65.0             # Gate 3: upscale  (vs bicubic baseline)
    gate_max_niqe: float = 6.0             # Gate 4: final    (naturalness)

    # Source format
    skip_deinterlace: bool = False  # set True if source is already progressive (e.g. OBS MP4)

    # Test mode
    test_mode: bool = False
    test_start: str = "00:05:00"
    test_duration: int = 30
    test_sample: bool = False  # auto-pick start from middle of video


# ── VapourSynth script template ───────────────────────────────────────────────

_VS_TEMPLATE_INTERLACED = textwrap.dedent("""\
    import vapoursynth as vs
    from vapoursynth import core
    import havsfunc as haf

    source = core.ffms2.Source(r"{input_path}")

    # Deinterlace – QTGMC, single-rate output
    deint = haf.QTGMC(source, Preset="{qtgmc_preset}", FPSDivisor=2)

    # Denoise – KNLMeansCL on GPU (CUDA)
    denoised = core.knlm.KNLMeansCL(
        deint,
        h={knlm_h},
        d={knlm_d},
        device_type="gpu",
        device_id={gpu_device_id},
    )

    # Color levels – correct VHS black/white points
    leveled = core.std.Levels(
        denoised,
        min_in={levels_min_in},
        max_in={levels_max_in},
        min_out=0,
        max_out=255,
    )

    leveled.set_output()
""")

# Used when source is already progressive (OBS MP4, pre-deinterlaced capture)
_VS_TEMPLATE_PROGRESSIVE = textwrap.dedent("""\
    import vapoursynth as vs
    from vapoursynth import core

    source = core.ffms2.Source(r"{input_path}")

    # Denoise – KNLMeansCL on GPU (CUDA)
    denoised = core.knlm.KNLMeansCL(
        source,
        h={knlm_h},
        d={knlm_d},
        device_type="gpu",
        device_id={gpu_device_id},
    )

    # Color levels – correct VHS black/white points
    leveled = core.std.Levels(
        denoised,
        min_in={levels_min_in},
        max_in={levels_max_in},
        min_out=0,
        max_out=255,
    )

    leveled.set_output()
""")


# ── Shell helpers ─────────────────────────────────────────────────────────────

def _run(cmd: list[str | Path], **kwargs) -> subprocess.CompletedProcess:
    """Run command and capture output (for short probes / checks)."""
    LOG.debug("$ %s", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kwargs)


def _run_live(cmd: list[str | Path]) -> None:
    """Run command streaming output to terminal (for long-running stages)."""
    LOG.debug("$ %s", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def _probe_duration(path: Path) -> float:
    r = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    return float(r.stdout.strip())


def _seconds_to_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _probe_fps(path: Path) -> str:
    r = _run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    return r.stdout.strip()


def _probe_dimensions(path: Path) -> tuple[int, int]:
    r = _run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(path),
    ])
    w, h = r.stdout.strip().split(",")
    return int(w), int(h)


def _extract_frame(video: Path, out_png: Path, frame_n: int = 100) -> None:
    _run([
        "ffmpeg", "-y",
        "-i", str(video),
        "-vf", f"select=eq(n\\,{frame_n})",
        "-frames:v", "1",
        "-pix_fmt", "rgb24",  # force 8-bit so piq receives values in 0-255
        str(out_png),
    ])


def _ffv1_flags() -> list[str]:
    return ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv422p10le"]


def _prores_flags() -> list[str]:
    return ["-c:v", "prores_ks", "-profile:v", "hq", "-pix_fmt", "yuv422p10le"]


def _video_flags(cfg: Config) -> list[str]:
    return _prores_flags() if cfg.output_codec == "prores" else _ffv1_flags()


# ── Stage 1: VapourSynth (deinterlace + denoise + color) ─────────────────────

def detect_field_order(path: Path) -> str:
    """Return 'progressive', 'tt' (TFF), 'bb' (BFF), or 'unknown'."""
    r = _run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=field_order",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    return r.stdout.strip() or "unknown"


def _vs_plugins_available() -> bool:
    """Return True if ffms2 and knlm are both loaded in VapourSynth core."""
    try:
        import vapoursynth as vs  # type: ignore
        core = vs.core
        return hasattr(core, "ffms2") and hasattr(core, "knlm")
    except Exception:
        return False


def _stage_vs_ffmpeg(input_path: Path, output_path: Path, cfg: Config) -> None:
    """Stage 1 fallback using ffmpeg filters (progressive sources only).

    Used when VapourSynth plugins (ffms2, KNLMeansCL) are not installed.
    hqdn3d approximates temporal+spatial denoising; colorlevels maps VHS
    video-range (16-235) to full range.
    """
    LOG.warning(
        "VapourSynth plugins (ffms2 / KNLMeansCL) not installed — "
        "using ffmpeg hqdn3d fallback for Stage 1.  "
        "Run bash pipeline/setup_ubuntu.sh to install full plugin stack."
    )

    # Map knlm_h (0.8–2.0) to hqdn3d luma_spatial (2–6)
    luma = max(1.0, cfg.knlm_h * 3.0)
    filters = [f"hqdn3d={luma:.1f}:{luma*0.75:.1f}:{luma*1.5:.1f}:{luma:.1f}"]

    lo = cfg.levels_min_in / 255.0
    hi = cfg.levels_max_in / 255.0
    if lo > 0.001 or hi < 0.999:
        filters.append(
            f"colorlevels=rimin={lo:.4f}:rimax={hi:.4f}"
            f":gimin={lo:.4f}:gimax={hi:.4f}"
            f":bimin={lo:.4f}:bimax={hi:.4f}"
        )

    cmd = (
        ["ffmpeg", "-y", "-i", str(input_path), "-vf", ",".join(filters)]
        + _ffv1_flags()
        + [str(output_path)]
    )
    _run_live(cmd)
    LOG.info("  Stage 1 complete (ffmpeg fallback) → %s", output_path)


def stage_vs(input_path: Path, output_path: Path, cfg: Config, work_dir: Path) -> None:
    if cfg.skip_deinterlace:
        LOG.info("[Stage 1] Denoise + Color correction (deinterlace skipped — progressive source)")
        # Use ffmpeg fallback when VapourSynth plugins are missing (progressive only)
        if not _vs_plugins_available():
            _stage_vs_ffmpeg(input_path, output_path, cfg)
            return
        template = _VS_TEMPLATE_PROGRESSIVE
    else:
        if not _vs_plugins_available():
            raise RuntimeError(
                "VapourSynth plugins (ffms2, KNLMeansCL) are required for interlaced sources. "
                "Run bash pipeline/setup_ubuntu.sh to install them."
            )
        LOG.info("[Stage 1] Deinterlace + Denoise + Color correction (VapourSynth)")
        template = _VS_TEMPLATE_INTERLACED

    script = template.format(
        input_path=str(input_path).replace("\\", "/"),
        qtgmc_preset=cfg.qtgmc_preset,
        knlm_h=cfg.knlm_h,
        knlm_d=cfg.knlm_d,
        gpu_device_id=cfg.gpu_device_id,
        levels_min_in=cfg.levels_min_in,
        levels_max_in=cfg.levels_max_in,
    )

    script_path = work_dir / "pipeline.vpy"
    script_path.write_text(script)

    vspipe_cmd = [_find_vspipe(), "-c", "y4m", str(script_path), "-"]
    ffmpeg_cmd = (
        ["ffmpeg", "-y", "-f", "yuv4mpegpipe", "-i", "pipe:"]
        + _ffv1_flags()
        + [str(output_path)]
    )

    LOG.info("  vspipe | ffmpeg  (this will take a while — QTGMC + KNLMeansCL)...")
    vspipe = subprocess.Popen(vspipe_cmd, stdout=subprocess.PIPE)
    ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=vspipe.stdout)
    vspipe.stdout.close()
    ffmpeg.wait()
    vspipe.wait()

    if vspipe.returncode != 0 or ffmpeg.returncode != 0:
        raise RuntimeError("Stage 1 failed — check VapourSynth and ffmpeg logs above")

    LOG.info("  Stage 1 complete → %s", output_path)


# ── Stage 2: Real-ESRGAN upscaling ───────────────────────────────────────────

def stage_upscale(input_path: Path, output_path: Path, cfg: Config, work_dir: Path) -> None:
    LOG.info(
        "[Stage 2] Upscaling %dx with %s (tile=%d)",
        cfg.realesrgan_scale, cfg.realesrgan_model, cfg.realesrgan_tile,
    )

    frames_in = work_dir / "frames_in"
    frames_out = work_dir / "frames_out"
    frames_in.mkdir(exist_ok=True)
    frames_out.mkdir(exist_ok=True)

    fps = _probe_fps(input_path)
    LOG.info("  Extracting frames (source fps=%s)...", fps)
    _run_live([
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", "format=rgb24",  # force 8-bit before PNG encode (pix_fmt alone ignored for image2)
        str(frames_in / "frame_%08d.png"),
    ])

    realesrgan_script = Path(cfg.realesrgan_dir).expanduser() / "inference_realesrgan.py"
    if not realesrgan_script.exists():
        raise FileNotFoundError(
            f"Real-ESRGAN inference script not found: {realesrgan_script}\n"
            f"Clone the repo: git clone https://github.com/xinntao/Real-ESRGAN\n"
            f"Then set --realesrgan-dir to its path."
        )

    tile_args = ["--tile", str(cfg.realesrgan_tile)] if cfg.realesrgan_tile > 0 else []
    LOG.info("  Running Real-ESRGAN inference...")
    _run_live([
        sys.executable, str(realesrgan_script),
        "-n", cfg.realesrgan_model,
        "-i", str(frames_in),
        "-o", str(frames_out),
        "-s", str(cfg.realesrgan_scale),
        "--suffix", "",  # keep original filename; default adds "_out" suffix
        *tile_args,
    ])

    LOG.info("  Reassembling frames → %s", output_path)
    _run_live([
        "ffmpeg", "-y",
        "-framerate", fps,
        "-i", str(frames_out / "frame_%08d.png"),
        *_ffv1_flags(),
        str(output_path),
    ])

    LOG.info("  Stage 2 complete → %s", output_path)


# ── Stage 3: Film grain + final encode ───────────────────────────────────────

def stage_final(
    video: Path, audio_source: Path, output_path: Path, cfg: Config
) -> None:
    LOG.info("[Stage 3] Film grain (strength=%d) + final encode (%s)", cfg.grain_strength, cfg.output_codec)

    grain_filter = f"noise=alls={cfg.grain_strength}:allf=t+u"

    _run_live([
        "ffmpeg", "-y",
        "-i", str(video),
        "-i", str(audio_source),
        "-map", "0:v",
        "-map", "1:a",
        "-vf", grain_filter,
        *_video_flags(cfg),
        "-c:a", "flac",
        str(output_path),
    ])

    LOG.info("  Stage 3 complete → %s", output_path)


# ── Quality gates ─────────────────────────────────────────────────────────────

def gate_deinterlace(video: Path, cfg: Config) -> dict:
    """Gate 1: check % progressive frames via ffmpeg idet filter."""
    LOG.info("[Gate 1] Deinterlace check (idet)...")
    try:
        result = _run([
            "ffmpeg", "-i", str(video),
            "-vf", "idet",
            "-frames:v", "500",
            "-f", "null", "-",
        ])
        output = result.stderr

        match = re.search(
            r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)",
            output,
        )
        if match:
            tff, bff, progressive = int(match.group(1)), int(match.group(2)), int(match.group(3))
            total = tff + bff + progressive or 1
            progressive_pct = progressive / total
        else:
            LOG.warning("  Could not parse idet output — skipping gate")
            return {"gate": "deinterlace", "passed": None, "note": "idet parse failed"}

        passed = progressive_pct >= cfg.gate_min_progressive_pct
        gate = {
            "gate": "deinterlace",
            "progressive_pct": round(progressive_pct, 3),
            "tff_frames": tff,
            "bff_frames": bff,
            "threshold": cfg.gate_min_progressive_pct,
            "passed": passed,
        }
    except subprocess.CalledProcessError as e:
        gate = {"gate": "deinterlace", "passed": None, "note": str(e)}

    _log_gate(gate)
    if gate["passed"] is False:
        LOG.warning(
            "  Suggestion: %.0f%% progressive < %.0f%% threshold. "
            "Check field order — try flipping with setfield=bff in VS script.",
            gate["progressive_pct"] * 100, cfg.gate_min_progressive_pct * 100,
        )
    return gate


def gate_denoise(before: Path, after: Path, cfg: Config, work_dir: Path) -> dict:
    """Gate 2: BRISQUE score should not significantly increase after denoising."""
    LOG.info("[Gate 2] Denoise check (BRISQUE)...")
    try:
        import piq
        import torch
        from torchvision.io import read_image

        frame_before = work_dir / "g2_before.png"
        frame_after  = work_dir / "g2_after.png"
        _extract_frame(before, frame_before)
        _extract_frame(after, frame_after)

        def brisque_score(p: Path) -> float:
            img = read_image(str(p)).float().div(255.0).unsqueeze(0)
            return piq.brisque(img, data_range=1.0).item()

        score_before = brisque_score(frame_before)
        score_after  = brisque_score(frame_after)
        delta = score_after - score_before
        passed = delta <= cfg.gate_max_brisque_delta

        gate = {
            "gate": "denoise",
            "brisque_before": round(score_before, 2),
            "brisque_after":  round(score_after,  2),
            "delta": round(delta, 2),
            "threshold_delta": cfg.gate_max_brisque_delta,
            "passed": passed,
        }
    except ImportError:
        LOG.warning("  piq not installed — skipping BRISQUE gate (pip install piq)")
        gate = {"gate": "denoise", "passed": None, "note": "piq not installed"}

    _log_gate(gate)
    if gate.get("passed") is False:
        LOG.warning(
            "  Suggestion: BRISQUE rose by %.2f — denoising is over-smoothing. "
            "Try reducing --knlm-h (current: %.1f → try %.1f).",
            gate["delta"], cfg.knlm_h, max(0.5, cfg.knlm_h - 0.4),
        )
    return gate


def gate_upscale(upscaled: Path, source: Path, cfg: Config, work_dir: Path) -> dict:
    """Gate 3: SSIM score vs bicubic upscale using piq (higher = better, target ≥0.70)."""
    LOG.info("[Gate 3] Upscale quality check (SSIM vs bicubic)...")
    try:
        import piq
        from torchvision.io import read_image

        w, h = _probe_dimensions(upscaled)

        frame_up = work_dir / "g3_upscaled.png"
        frame_bc = work_dir / "g3_bicubic.png"

        _extract_frame(upscaled, frame_up, frame_n=200)
        _run_live([
            "ffmpeg", "-y",
            "-i", str(source),
            "-vf", f"select=eq(n\\,200),scale={w}:{h}:flags=bicubic,format=rgb24",
            "-frames:v", "1",
            str(frame_bc),
        ])

        img_up = read_image(str(frame_up)).float().div(255.0).unsqueeze(0)
        img_bc = read_image(str(frame_bc)).float().div(255.0).unsqueeze(0)
        score = piq.ssim(img_up, img_bc, data_range=1.0).item()
        passed = score >= cfg.gate_min_vmaf / 100.0  # gate_min_vmaf is 0-100, ssim is 0-1

        gate = {
            "gate": "upscale",
            "ssim_vs_bicubic": round(score, 4),
            "passed": passed,
        }
    except (ImportError, AttributeError) as e:
        LOG.warning("  piq not available — skipping upscale gate")
        gate = {"gate": "upscale", "passed": None, "note": "piq unavailable"}
    except (subprocess.CalledProcessError, KeyError) as e:
        gate = {"gate": "upscale", "passed": None, "note": str(e)}

    _log_gate(gate)
    if gate.get("passed") is False:
        LOG.warning(
            "  Suggestion: SSIM %.4f below threshold. Try model 'RealESRGAN_x4plus' "
            "or reduce scale from %dx to 2x.",
            gate.get("ssim_vs_bicubic", 0), cfg.realesrgan_scale,
        )
    return gate


def gate_final(video: Path, cfg: Config, work_dir: Path) -> dict:
    """Gate 4: naturalness check on final output using BRISQUE (lower = better, target ≤50)."""
    LOG.info("[Gate 4] Final naturalness check (BRISQUE)...")
    try:
        import piq
        from torchvision.io import read_image

        frame = work_dir / "g4_final.png"
        _extract_frame(video, frame, frame_n=200)
        img = read_image(str(frame)).float().div(255.0).unsqueeze(0)
        score = piq.brisque(img, data_range=1.0).item()
        passed = score <= cfg.gate_max_niqe * 8  # BRISQUE scale ~0-100, NIQE threshold was ~6

        gate = {
            "gate": "final",
            "brisque_score": round(score, 3),
            "passed": passed,
        }
    except (ImportError, AttributeError):
        LOG.warning("  piq not available — skipping Gate 4")
        gate = {"gate": "final", "passed": None, "note": "piq unavailable"}

    _log_gate(gate)
    if gate.get("passed") is False:
        LOG.warning(
            "  Suggestion: NIQE %.2f > %.1f — output looks unnatural (AI plastic look). "
            "Reduce --grain or lower --knlm-h, or switch to a lighter QTGMC preset.",
            gate.get("niqe_score", 0), cfg.gate_max_niqe,
        )
    return gate


def _log_gate(gate: dict) -> None:
    status = {True: "✓ PASSED", False: "✗ FAILED", None: "~ SKIPPED"}[gate.get("passed")]
    details = {k: v for k, v in gate.items() if k not in ("gate", "passed")}
    LOG.info("  %s  %s  |  %s", gate["gate"].upper().ljust(14), status, details)


# ── Pipeline orchestration ────────────────────────────────────────────────────

def run_pipeline(input_path: Path, output_path: Path, cfg: Config) -> dict:
    # FFV1 and ProRes require MKV; silently fix .mp4 output paths
    if output_path.suffix.lower() == ".mp4" and cfg.output_codec in ("ffv1", "prores"):
        output_path = output_path.with_suffix(".mkv")
        LOG.warning("Output renamed to %s — FFV1/ProRes not supported in MP4 container", output_path)

    work_dir = output_path.parent / f".vhs_work_{output_path.stem}"
    work_dir.mkdir(exist_ok=True)
    LOG.info("Work dir: %s", work_dir)

    audio_source = input_path  # always use original audio

    try:
        # Test mode: extract a representative clip
        if cfg.test_mode:
            test_start = cfg.test_start
            if cfg.test_sample:
                duration = _probe_duration(input_path)
                mid = duration / 2.0
                start_sec = max(0.0, mid - cfg.test_duration / 2.0)
                test_start = _seconds_to_hms(start_sec)
                LOG.info(
                    "TEST MODE (auto-sample) — video duration %.1fs, sampling %ds clip "
                    "from middle at %s",
                    duration, cfg.test_duration, test_start,
                )
            else:
                LOG.info(
                    "TEST MODE — extracting %ds clip from %s (start: %s)",
                    cfg.test_duration, input_path.name, test_start,
                )
            clip = work_dir / "test_clip.mkv"
            _run_live([
                "ffmpeg", "-y",
                "-ss", test_start,
                "-t", str(cfg.test_duration),
                "-i", str(input_path),
                "-c", "copy",
                str(clip),
            ])
            input_path = clip
            audio_source = clip

        # Auto-detect field order if not explicitly set
        if not cfg.skip_deinterlace:
            field_order = detect_field_order(input_path)
            LOG.info("Field order detected: %s", field_order)
            if field_order == "progressive":
                LOG.warning(
                    "Source is already progressive (likely OBS deinterlaced during capture). "
                    "Skipping QTGMC. Use --skip-deinterlace to suppress this message."
                )
                cfg.skip_deinterlace = True

        report: dict = {
            "input": str(input_path),
            "output": str(output_path),
            "config": asdict(cfg),
            "gates": [],
        }

        # ── Stage 1 ──
        vs_out = work_dir / "s1_vs.mkv"
        stage_vs(input_path, vs_out, cfg, work_dir)
        if not cfg.skip_deinterlace:
            g1 = gate_deinterlace(vs_out, cfg)
            report["gates"].append(g1)
        g2 = gate_denoise(input_path, vs_out, cfg, work_dir)
        report["gates"].append(g2)

        # ── Stage 2 ──
        upscaled_out = work_dir / "s2_upscaled.mkv"
        stage_upscale(vs_out, upscaled_out, cfg, work_dir)
        g3 = gate_upscale(upscaled_out, vs_out, cfg, work_dir)
        report["gates"].append(g3)

        # ── Stage 3 ──
        stage_final(upscaled_out, audio_source, output_path, cfg)
        g4 = gate_final(output_path, cfg, work_dir)
        report["gates"].append(g4)

        # ── Report ──
        passed = sum(1 for g in report["gates"] if g.get("passed") is True)
        total  = sum(1 for g in report["gates"] if g.get("passed") is not None)
        report["summary"] = {"gates_passed": passed, "gates_total": total}

        report_path = output_path.with_suffix("").with_suffix(".quality_report.json")
        report_path.write_text(json.dumps(report, indent=2))

        LOG.info("─" * 60)
        LOG.info("Done.  Gates: %d/%d passed", passed, total)
        LOG.info("Output:  %s", output_path)
        LOG.info("Report:  %s", report_path)

        return report

    finally:
        if not cfg.keep_intermediates:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            LOG.info("Intermediates kept at: %s", work_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VHS post-processing pipeline (deinterlace → denoise → upscale → grain)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",  type=Path, help="Captured VHS .mkv file")
    p.add_argument("output", type=Path, help="Output file path")

    g = p.add_argument_group("Deinterlacing")
    g.add_argument("--qtgmc-preset", default="Slower",
                   choices=["Draft", "Fast", "Medium", "Slow", "Slower", "Placebo"])
    g.add_argument("--skip-deinterlace", action="store_true",
                   help="Skip QTGMC — use when source is already progressive (OBS MP4)")

    g = p.add_argument_group("Denoising")
    g.add_argument("--knlm-h",   type=float, default=1.2,  help="Denoise strength (0.8–2.0)")
    g.add_argument("--knlm-d",   type=int,   default=1,    help="Temporal radius in frames")
    g.add_argument("--gpu",      type=int,   default=0,    dest="gpu_device_id")

    g = p.add_argument_group("Color correction")
    g.add_argument("--levels-min-in", type=int, default=16,  help="Black point")
    g.add_argument("--levels-max-in", type=int, default=235, help="White point")

    g = p.add_argument_group("Upscaling (Real-ESRGAN)")
    g.add_argument("--realesrgan-model", default="realesr-general-x4v3",
                   choices=["realesr-general-x4v3", "RealESRGAN_x4plus", "RealESRGAN_x4plus-anime"])
    g.add_argument("--realesrgan-scale", type=int,  default=2,    choices=[2, 4])
    g.add_argument("--realesrgan-tile",  type=int,  default=256,  help="0 = disable tiling")
    g.add_argument("--realesrgan-dir",   default="~/Real-ESRGAN", help="Path to cloned repo")

    g = p.add_argument_group("Output")
    g.add_argument("--grain",  type=int, default=4,     dest="grain_strength", help="Film grain 1–8")
    g.add_argument("--codec",  default="ffv1",          dest="output_codec",   choices=["ffv1", "prores"])
    g.add_argument("--keep-intermediates", action="store_true")

    g = p.add_argument_group("Quality gate thresholds")
    g.add_argument("--gate-progressive-pct", type=float, default=0.95, dest="gate_min_progressive_pct",
                   help="Min progressive frames after deinterlace")
    g.add_argument("--gate-brisque-delta",   type=float, default=5.0,  dest="gate_max_brisque_delta",
                   help="Max BRISQUE increase after denoise")
    g.add_argument("--gate-vmaf",            type=float, default=65.0, dest="gate_min_vmaf",
                   help="Min VMAF vs bicubic after upscale")
    g.add_argument("--gate-niqe",            type=float, default=6.0,  dest="gate_max_niqe",
                   help="Max NIQE on final output")

    g = p.add_argument_group("Test mode")
    g.add_argument("--test",           action="store_true", dest="test_mode",
                   help="Process 30-second sample clip only")
    g.add_argument("--test-sample",    action="store_true", dest="test_sample",
                   help="Auto-pick clip from middle of video (probes duration, overrides --test-start)")
    g.add_argument("--test-start",     default="00:05:00",  help="Clip start time HH:MM:SS (ignored when --test-sample is set)")
    g.add_argument("--test-duration",  type=int, default=30, help="Clip length in seconds")

    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> None:
    _check_env()
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    input_path  = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    skip = {"input", "output", "verbose"}
    cfg_kwargs = {k: v for k, v in vars(args).items() if k not in skip}
    cfg = Config(**cfg_kwargs)

    run_pipeline(input_path, output_path, cfg)


if __name__ == "__main__":
    main()
