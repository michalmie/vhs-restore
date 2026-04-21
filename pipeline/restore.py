#!/usr/bin/env python3
"""VHS digitization post-processing pipeline."""

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
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

LOG = logging.getLogger("vhs")
_VERBOSE = False  # set to True via --verbose; controls whether subprocess output is shown
_CONSOLE = Console()

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
    output_codec: str = "ffv1"     # ffv1 | prores | h264 | h265
    output_crf: int = 18           # quality for h264/h265 (0=lossless, 51=worst)
    output_res: str = "native"     # native | 720p | 1080p | 4k | WxH
    keep_intermediates: bool = False

    # Quality gate thresholds
    gate_min_progressive_pct: float = 0.95  # Gate 1: deinterlace
    gate_max_brisque_delta: float = 5.0     # Gate 2: denoise  (fail if BRISQUE rises by this)
    gate_min_vmaf: float = 65.0             # Gate 3: upscale  (vs bicubic baseline)
    gate_max_niqe: float = 6.0             # Gate 4: final    (naturalness)

    # Source format
    skip_deinterlace: bool = False  # set True if source is already progressive (e.g. OBS MP4)

    # ── Opt-in effects ────────────────────────────────────────────────────────
    stabilize: bool = False          # video stabilization (requires libvidstab in ffmpeg)
    repair_dropouts: bool = False    # VHS dropout / line artifact repair (median filter)
    balance_brightness: bool = False # auto-normalize brightness/contrast
    sharpen: bool = False            # unsharp mask after upscaling
    cas: bool = False                # Contrast Adaptive Sharpening after upscaling
    fix_ar: bool = False             # correct aspect ratio (e.g. anamorphic → square pixels)
    audio_cleanup: bool = False      # spectral noise reduction on audio track

    # ── Opt-out effects ───────────────────────────────────────────────────────
    skip_denoise: bool = False       # disable KNLMeansCL / hqdn3d
    skip_color: bool = False         # disable VHS levels correction (16-235)

    # ── Effect tuning ─────────────────────────────────────────────────────────
    sharpen_amount: float = 1.5      # unsharp luma amount (0.5–3.0)
    saturation: float = 1.0          # color saturation multiplier (1.0 = no change)
    warmth: float = 0.0              # color temperature: -1.0=cooler  0.0=neutral  +1.0=warmer
    cas_strength: float = 0.6        # CAS strength (0.0–1.0)
    ar_target: str = "4:3"           # target display aspect ratio for --fix-ar
    audio_cleanup_db: float = 10.0   # afftdn noise reduction dB (5=gentle, 20=strong)

    # Test mode
    test_mode: bool = False
    test_start: str = "00:05:00"
    test_duration: int = 10
    test_sample: bool = False   # auto-pick start from middle of video
    compare: bool = False       # generate side-by-side comparison video


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
    """Run a long-running command. Output shown only in --verbose mode; always shown on failure."""
    LOG.debug("$ %s", " ".join(str(c) for c in cmd))
    if _VERBOSE:
        subprocess.run(cmd, check=True)
        return
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        sys.stderr.buffer.write(result.stdout)
        sys.stderr.buffer.write(result.stderr)
        sys.stderr.flush()
        raise subprocess.CalledProcessError(result.returncode, cmd)


def _run_tracking(
    cmd: list[str | Path],
    *,
    on_stderr: Callable[[str], None] | None = None,
    on_stdout: Callable[[str], None] | None = None,
) -> None:
    """Run command and stream output to optional line callbacks for progress parsing."""
    LOG.debug("$ %s", " ".join(str(c) for c in cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    captured_err: list[str] = []

    def _drain(stream, cb, store):
        for line in stream:
            line = line.rstrip()
            store.append(line)
            if cb:
                cb(line)
        stream.close()

    t_out = threading.Thread(target=_drain, args=(proc.stdout, on_stdout, []))
    t_err = threading.Thread(target=_drain, args=(proc.stderr, on_stderr, captured_err))
    t_out.start()
    t_err.start()
    proc.wait()
    t_out.join()
    t_err.join()

    if proc.returncode != 0:
        for line in captured_err:
            _CONSOLE.print(line, style="red", highlight=False)
        raise subprocess.CalledProcessError(proc.returncode, cmd)


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


def _probe_frame_count(path: Path) -> int:
    """Estimate frame count from container metadata (fast, no decoding)."""
    try:
        r = _run([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ])
        val = r.stdout.strip()
        if val and val != "N/A":
            return int(val)
        # Fallback: duration × fps
        dur = _probe_duration(path)
        fps_str = _probe_fps(path)
        num, den = (fps_str.split("/") + ["1"])[:2]
        return int(dur * int(num) / int(den))
    except Exception:
        return 0


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


def _h264_flags(crf: int) -> list[str]:
    return ["-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p"]


def _h265_flags(crf: int) -> list[str]:
    return ["-c:v", "libx265", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p"]


def _video_flags(cfg: Config) -> list[str]:
    if cfg.output_codec == "prores":
        return _prores_flags()
    if cfg.output_codec == "h264":
        return _h264_flags(cfg.output_crf)
    if cfg.output_codec == "h265":
        return _h265_flags(cfg.output_crf)
    return _ffv1_flags()


_RES_MAP = {"720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)}


def _output_res_filter(cfg: Config) -> str | None:
    """Return a scale filter string for --output-res, or None if native."""
    if cfg.output_res == "native":
        return None
    if cfg.output_res in _RES_MAP:
        w, h = _RES_MAP[cfg.output_res]
        return f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos"
    if "x" in cfg.output_res:
        w, h = cfg.output_res.split("x", 1)
        return f"scale={w}:{h}:flags=lanczos"
    return None


# ── Effect filter builders ────────────────────────────────────────────────────

def _pre_upscale_filters(cfg: Config) -> list[str]:
    """ffmpeg video filters applied before upscaling (Stage 1 output)."""
    f = []
    if cfg.repair_dropouts:
        # Temporal median blend reduces VHS horizontal-line dropout artifacts
        f.append("tblend=all_mode=median")
        f.append("hqdn3d=0:0:3:3")   # temporal-only pass to smooth blended frames
    if cfg.balance_brightness:
        # Stretch histogram to use full range; preserve relative tone
        f.append("normalize=blackpt=black:whitept=white:smoothing=5")
    if cfg.warmth != 0.0:
        w = cfg.warmth * 0.15
        rr = round(max(0.5, min(1.5, 1.0 + w)), 3)
        bb = round(max(0.5, min(1.5, 1.0 - w)), 3)
        f.append(f"colorchannelmixer=rr={rr}:bb={bb}")
    if cfg.saturation != 1.0:
        f.append(f"eq=saturation={cfg.saturation:.2f}")
    return f


def _post_upscale_filters(cfg: Config) -> list[str]:
    """ffmpeg video filters applied after upscaling, before final encode."""
    f = []
    if cfg.fix_ar:
        # Map common named ratios to SAR values; Real-ESRGAN outputs square pixels so we
        # just set the display AR via setdar without scaling (no quality loss).
        f.append(f"setdar={cfg.ar_target.replace(':', '/')}")
    if cfg.cas:
        f.append(f"cas=strength={cfg.cas_strength:.2f}")
    if cfg.sharpen:
        amt = cfg.sharpen_amount
        f.append(f"unsharp=lx=5:ly=5:la={amt:.1f}:cx=3:cy=3:ca=0.0")
    return f


def _audio_filters(cfg: Config) -> list[str]:
    """ffmpeg audio filters for Stage 3."""
    if cfg.audio_cleanup:
        return [f"afftdn=nf=-{cfg.audio_cleanup_db:.0f}"]
    return []


# ── Stabilization (pre-Stage 1, optional 2-pass) ─────────────────────────────

def stage_stabilize(input_path: Path, output_path: Path, work_dir: Path,
                    on_progress: Callable[[str, int, int], None] | None = None) -> None:
    """2-pass video stabilization via libvidstab (ffmpeg must be compiled with it)."""
    # Check availability
    probe = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True)
    if "vidstabdetect" not in probe.stdout:
        LOG.warning(
            "libvidstab not available in this ffmpeg build — skipping stabilization.\n"
            "  Install: sudo apt install ffmpeg  (system ffmpeg has libvidstab)\n"
            "  Or compile jellyfin-ffmpeg with --enable-libvidstab"
        )
        import shutil as _sh
        _sh.copy2(input_path, output_path)
        return

    trf = work_dir / "stabilize.trf"
    total = _probe_frame_count(input_path)

    LOG.info("  Stabilize pass 1/2: motion analysis...")
    _run_tracking(
        ["ffmpeg", "-y", "-i", str(input_path),
         "-vf", f"vidstabdetect=shakiness=5:accuracy=15:result={trf}",
         "-f", "null", "-"],
        on_stderr=lambda l: (
            on_progress("stabilize", int(m.group(1)), total)
            if (m := re.search(r"frame=\s*(\d+)", l)) and on_progress else None
        ),
    )

    LOG.info("  Stabilize pass 2/2: applying transform...")
    _run_tracking(
        ["ffmpeg", "-y", "-i", str(input_path),
         "-vf", f"vidstabtransform=input={trf}:zoom=5:smoothing=30,unsharp=5:5:0.8:3:3:0.4",
         "-c:a", "copy", str(output_path)],
        on_stderr=lambda l: (
            on_progress("stabilize", int(m.group(1)), total)
            if (m := re.search(r"frame=\s*(\d+)", l)) and on_progress else None
        ),
    )


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


def _stage_vs_ffmpeg(
    input_path: Path, output_path: Path, cfg: Config,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> None:
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

    filters = []

    if not cfg.skip_denoise:
        luma = max(1.0, cfg.knlm_h * 3.0)
        filters.append(f"hqdn3d={luma:.1f}:{luma*0.75:.1f}:{luma*1.5:.1f}:{luma:.1f}")

    if not cfg.skip_color:
        lo = cfg.levels_min_in / 255.0
        hi = cfg.levels_max_in / 255.0
        if lo > 0.001 or hi < 0.999:
            filters.append(
                f"colorlevels=rimin={lo:.4f}:rimax={hi:.4f}"
                f":gimin={lo:.4f}:gimax={hi:.4f}"
                f":bimin={lo:.4f}:bimax={hi:.4f}"
            )

    filters.extend(_pre_upscale_filters(cfg))

    total = _probe_frame_count(input_path)
    cmd = (
        ["ffmpeg", "-y", "-i", str(input_path), "-vf", ",".join(filters)]
        + _ffv1_flags()
        + [str(output_path)]
    )
    _run_tracking(
        cmd,
        on_stderr=lambda l: (
            on_progress("vs", int(m.group(1)), total)
            if (m := re.search(r"frame=\s*(\d+)", l)) and on_progress else None
        ),
    )
    LOG.info("  Stage 1 complete (ffmpeg fallback) → %s", output_path)


def stage_vs(
    input_path: Path, output_path: Path, cfg: Config, work_dir: Path,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> None:
    if cfg.skip_deinterlace:
        LOG.info("[Stage 1] Denoise + Color correction (deinterlace skipped — progressive source)")
        # Use ffmpeg fallback when VapourSynth plugins are missing (progressive only)
        if not _vs_plugins_available():
            _stage_vs_ffmpeg(input_path, output_path, cfg, on_progress)
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
    total = _probe_frame_count(input_path)
    vspipe = subprocess.Popen(vspipe_cmd, stdout=subprocess.PIPE)
    ffmpeg = subprocess.Popen(
        ffmpeg_cmd, stdin=vspipe.stdout,
        stderr=subprocess.PIPE, text=True, errors="replace",
    )
    vspipe.stdout.close()

    for line in ffmpeg.stderr:
        m = re.search(r"frame=\s*(\d+)", line)
        if m and on_progress:
            on_progress("vs", int(m.group(1)), total)

    ffmpeg.wait()
    vspipe.wait()

    if vspipe.returncode != 0 or ffmpeg.returncode != 0:
        raise RuntimeError("Stage 1 failed — check VapourSynth and ffmpeg logs above")

    LOG.info("  Stage 1 complete → %s", output_path)


# ── Stage 2: Real-ESRGAN upscaling ───────────────────────────────────────────

def stage_upscale(
    input_path: Path, output_path: Path, cfg: Config, work_dir: Path,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> None:
    LOG.info(
        "[Stage 2] Upscaling %dx with %s (tile=%d)",
        cfg.realesrgan_scale, cfg.realesrgan_model, cfg.realesrgan_tile,
    )

    frames_in = work_dir / "frames_in"
    frames_out = work_dir / "frames_out"
    frames_in.mkdir(exist_ok=True)
    frames_out.mkdir(exist_ok=True)

    fps = _probe_fps(input_path)
    total_frames = _probe_frame_count(input_path)
    LOG.info("  Extracting frames (source fps=%s)...", fps)

    def _ffmpeg_frame_cb(line: str, on_progress):
        m = re.search(r"frame=\s*(\d+)", line)
        if m and on_progress:
            on_progress("extract", int(m.group(1)), total_frames)

    _run_tracking(
        ["ffmpeg", "-y", "-i", str(input_path), "-vf", "format=rgb24",
         str(frames_in / "frame_%08d.png")],
        on_stderr=lambda l: _ffmpeg_frame_cb(l, on_progress),
    )

    realesrgan_script = Path(cfg.realesrgan_dir).expanduser() / "inference_realesrgan.py"
    if not realesrgan_script.exists():
        raise FileNotFoundError(
            f"Real-ESRGAN inference script not found: {realesrgan_script}\n"
            f"Clone the repo: git clone https://github.com/xinntao/Real-ESRGAN\n"
            f"Then set --realesrgan-dir to its path."
        )

    tile_args = ["--tile", str(cfg.realesrgan_tile)] if cfg.realesrgan_tile > 0 else []
    LOG.info("  Running Real-ESRGAN inference...")

    def _realesrgan_cb(line: str, on_progress):
        m = re.search(r"Testing\s+(\d+)", line)
        if m and on_progress:
            on_progress("upscale", int(m.group(1)) + 1, total_frames)

    _run_tracking(
        [sys.executable, str(realesrgan_script),
         "-n", cfg.realesrgan_model,
         "-i", str(frames_in), "-o", str(frames_out),
         "-s", str(cfg.realesrgan_scale),
         "--suffix", "", *tile_args],
        on_stdout=lambda l: _realesrgan_cb(l, on_progress),
    )

    LOG.info("  Reassembling frames → %s", output_path)
    _run_tracking(
        ["ffmpeg", "-y", "-framerate", fps,
         "-i", str(frames_out / "frame_%08d.png"),
         *_ffv1_flags(), str(output_path)],
        on_stderr=lambda l: _ffmpeg_frame_cb(l, on_progress),
    )

    LOG.info("  Stage 2 complete → %s", output_path)


# ── Stage 3: Film grain + final encode ───────────────────────────────────────

def stage_final(
    video: Path, audio_source: Path, output_path: Path, cfg: Config,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> None:
    LOG.info("[Stage 3] Film grain (strength=%d) + final encode (%s)", cfg.grain_strength, cfg.output_codec)

    vfilters: list[str] = []
    vfilters.extend(_post_upscale_filters(cfg))
    res_filter = _output_res_filter(cfg)
    if res_filter:
        vfilters.append(res_filter)
    if cfg.grain_strength > 0:
        vfilters.append(f"noise=alls={cfg.grain_strength}:allf=t+u")

    afilters = _audio_filters(cfg)
    audio_codec = "aac" if cfg.output_codec in ("h264", "h265") else "flac"
    total = _probe_frame_count(video)

    _run_tracking(
        ["ffmpeg", "-y",
         "-i", str(video), "-i", str(audio_source),
         "-map", "0:v", "-map", "1:a",
         *(["-vf", ",".join(vfilters)] if vfilters else []),
         *(["-af", ",".join(afilters)] if afilters else []),
         *_video_flags(cfg), "-c:a", audio_codec,
         str(output_path)],
        on_stderr=lambda l: (
            on_progress("final", int(m.group(1)), total)
            if (m := re.search(r"frame=\s*(\d+)", l)) and on_progress else None
        ),
    )

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
    # Enforce correct container: FFV1/ProRes → .mkv, h264/h265 → .mp4
    if cfg.output_codec in ("ffv1", "prores") and output_path.suffix.lower() != ".mkv":
        output_path = output_path.with_suffix(".mkv")
        LOG.warning("Output renamed to %s — FFV1/ProRes require MKV container", output_path)
    elif cfg.output_codec in ("h264", "h265") and output_path.suffix.lower() not in (".mp4", ".mkv"):
        output_path = output_path.with_suffix(".mp4")
        LOG.warning("Output renamed to %s", output_path)

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
                "-i", str(input_path),
                "-t", str(cfg.test_duration),  # output option — exact duration limit
                "-c", "copy",
                str(clip),
            ])
            input_path = clip
            audio_source = clip

        # Test mode: skip upscaling for speed, always output side-by-side
        if cfg.test_mode:
            cfg.realesrgan_scale = 1
            cfg.compare = True

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

        ui = PipelineUI(input_path, output_path, cfg)

        with ui:
            # ── Stabilization (optional pre-stage) ──
            if cfg.stabilize:
                stab_out = work_dir / "s0_stabilized.mkv"
                ui.start_stage(0, "Stabilization")
                stage_stabilize(input_path, stab_out, work_dir, on_progress=ui.on_progress)
                ui.finish_stage(0)
                input_path = stab_out
                audio_source = stab_out

            # ── Stage 1 ──
            vs_out = work_dir / "s1_vs.mkv"
            label1 = "Deinterlace + Denoise" if not cfg.skip_denoise else "Color correction"
            ui.start_stage(1, label1)
            stage_vs(input_path, vs_out, cfg, work_dir, on_progress=ui.on_progress)
            ui.finish_stage(1)

            if not cfg.skip_deinterlace:
                ui.start_gate("Gate 1", "Deinterlace")
                g1 = gate_deinterlace(vs_out, cfg)
                report["gates"].append(g1)
                ui.finish_gate(g1)

            ui.start_gate("Gate 2", "Denoise")
            g2 = gate_denoise(input_path, vs_out, cfg, work_dir)
            report["gates"].append(g2)
            ui.finish_gate(g2)

            # ── Stage 2 ──
            if cfg.realesrgan_scale == 1:
                upscaled_out = vs_out  # skip Real-ESRGAN in test mode / scale=1
            else:
                upscaled_out = work_dir / "s2_upscaled.mkv"
                ui.start_stage(2, "AI Upscaling (Real-ESRGAN)")
                stage_upscale(vs_out, upscaled_out, cfg, work_dir, on_progress=ui.on_progress)
                ui.finish_stage(2)

                ui.start_gate("Gate 3", "Upscale quality")
                g3 = gate_upscale(upscaled_out, vs_out, cfg, work_dir)
                report["gates"].append(g3)
                ui.finish_gate(g3)

            # ── Stage 3 ──
            # In test mode the comparison video IS the output; encode to a temp first
            final_encode_path = (
                work_dir / ("s3_final" + output_path.suffix)
                if cfg.test_mode and cfg.compare
                else output_path
            )
            ui.start_stage(3, "Film Grain + Final Encode")
            stage_final(upscaled_out, audio_source, final_encode_path, cfg, on_progress=ui.on_progress)
            ui.finish_stage(3)

            ui.start_gate("Gate 4", "Final quality")
            g4 = gate_final(final_encode_path, cfg, work_dir)
            report["gates"].append(g4)
            ui.finish_gate(g4)

        # ── Comparison video ──
        compare_path: Path | None = None
        if cfg.compare:
            if cfg.test_mode:
                # Comparison IS the output; write directly to output_path
                compare_path = output_path
            else:
                compare_path = output_path.with_name(
                    output_path.stem + "_compare.mp4"
                )
            ui.start_stage(4, "Side-by-side comparison")
            make_comparison(audio_source, final_encode_path, compare_path,
                            on_progress=ui.on_progress)
            ui.finish_stage(4)

        # ── Report ──
        passed = sum(1 for g in report["gates"] if g.get("passed") is True)
        total  = sum(1 for g in report["gates"] if g.get("passed") is not None)
        report["summary"] = {"gates_passed": passed, "gates_total": total}

        # In test mode the user-visible output is the comparison video
        visible_output = compare_path if (cfg.test_mode and compare_path) else output_path
        report_path = visible_output.with_suffix("").with_suffix(".quality_report.json")
        report_path.write_text(json.dumps(report, indent=2))

        ui.show_summary(passed, total, visible_output, report_path,
                        None if cfg.test_mode else compare_path)

        return report

    finally:
        if not cfg.keep_intermediates:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            LOG.info("Intermediates kept at: %s", work_dir)


# ── Side-by-side comparison ───────────────────────────────────────────────────

def make_comparison(
    original: Path,
    processed: Path,
    output: Path,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> None:
    """Render original (left) vs restored (right) side-by-side as H.264 MP4.

    Both sides are scaled to the processed video's height so the upscaling
    quality improvement is visible. A label is burned into each half.
    """
    w, h = _probe_dimensions(processed)
    total = _probe_frame_count(processed)

    # Scale original to match processed dimensions (bicubic, same as baseline)
    # Then hstack with the processed video; burn "ORIGINAL" / "RESTORED" labels.
    filter_complex = (
        f"[0:v]scale={w}:{h}:flags=lanczos,"
        f"drawtext=text='ORIGINAL':fontsize={max(18, h//30)}:fontcolor=white"
        f":shadowcolor=black:shadowx=2:shadowy=2:x=20:y=20[left];"
        f"[1:v]scale={w}:{h}:flags=lanczos,"
        f"drawtext=text='RESTORED':fontsize={max(18, h//30)}:fontcolor=white"
        f":shadowcolor=black:shadowx=2:shadowy=2:x=20:y=20[right];"
        "[left][right]hstack=inputs=2[out]"
    )

    _run_tracking(
        [
            "ffmpeg", "-y",
            "-i", str(original),
            "-i", str(processed),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac",
            str(output),
        ],
        on_stderr=lambda l: (
            on_progress("compare", int(m.group(1)), total)
            if (m := re.search(r"frame=\s*(\d+)", l)) and on_progress else None
        ),
    )


# ── TUI ───────────────────────────────────────────────────────────────────────

_STAGE_LABELS = {1: "Stage 1", 2: "Stage 2", 3: "Stage 3"}
_GATE_ICON = {True: "[green]✓[/]", False: "[red]✗[/]", None: "[dim]–[/]"}


class PipelineUI:
    """Rich Live dashboard for the restore pipeline."""

    def __init__(self, input_path: Path, output_path: Path, cfg: Config) -> None:
        self._input = input_path
        self._output = output_path
        self._cfg = cfg

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=32),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=False,
        )
        self._stage_tasks: dict[int, TaskID] = {}
        self._gates: list[tuple[str, str, str]] = []  # (label, name, status_markup)
        self._lock = threading.Lock()
        self._live = Live(self._render(), refresh_per_second=8, console=_CONSOLE)

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        self._live.__exit__(*args)

    # ── stage control ─────────────────────────────────────────────────────────

    def start_stage(self, n: int, name: str) -> None:
        tid = self._progress.add_task(f"[cyan]Stage {n}[/]  {name}", total=None)
        with self._lock:
            self._stage_tasks[n] = tid
        self._refresh()

    def finish_stage(self, n: int) -> None:
        with self._lock:
            tid = self._stage_tasks.get(n)
        if tid is not None:
            self._progress.update(tid, completed=True, total=1,
                                  description=f"[green]Stage {n}[/]  ✓")
        self._refresh()

    def on_progress(self, step: str, done: int, total: int) -> None:
        stage_map = {"vs": 1, "extract": 2, "upscale": 2, "final": 3}
        n = stage_map.get(step, 0)
        with self._lock:
            tid = self._stage_tasks.get(n)
        if tid is not None and total > 0:
            self._progress.update(tid, completed=done, total=total)
        self._refresh()

    # ── gate control ──────────────────────────────────────────────────────────

    def start_gate(self, label: str, name: str) -> None:
        with self._lock:
            self._gates.append((label, name, "[dim]running…[/]"))
        self._refresh()

    def finish_gate(self, gate: dict) -> None:
        passed = gate.get("passed")
        icon   = _GATE_ICON[passed]
        detail = self._gate_detail(gate)
        with self._lock:
            if self._gates:
                label, name, _ = self._gates[-1]
                self._gates[-1] = (label, name, f"{icon}  {detail}")
        self._refresh()

    @staticmethod
    def _gate_detail(gate: dict) -> str:
        if "progressive_pct" in gate:
            return f"{gate['progressive_pct']*100:.1f}% progressive"
        if "delta" in gate:
            return f"BRISQUE Δ{gate['delta']:+.1f}"
        if "ssim_vs_bicubic" in gate:
            return f"SSIM {gate['ssim_vs_bicubic']:.3f}"
        if "brisque_score" in gate:
            return f"BRISQUE {gate['brisque_score']:.1f}"
        return gate.get("note", "")

    # ── rendering ─────────────────────────────────────────────────────────────

    def _render(self):
        # Header
        codec_label = self._cfg.output_codec.upper()
        if self._cfg.output_codec in ("h264", "h265"):
            codec_label += f" CRF{self._cfg.output_crf}"
        res_label = (
            f"  ·  → {self._cfg.output_res}"
            if self._cfg.output_res != "native" else ""
        )
        header = (
            f"[bold]{self._input.name}[/]  →  [bold]{self._output.name}[/]\n"
            f"[dim]{codec_label}  ·  {self._cfg.realesrgan_scale}× upscale"
            f"  ·  grain {self._cfg.grain_strength}{res_label}[/]"
        )

        # Gates table
        gate_table = Table(box=None, padding=(0, 2, 0, 0), show_header=False)
        gate_table.add_column(style="dim", width=8)
        gate_table.add_column(width=18)
        gate_table.add_column()
        with self._lock:
            gates_snapshot = list(self._gates)
        for label, name, status in gates_snapshot:
            gate_table.add_row(label, name, status)

        return Panel(
            Columns([self._progress, gate_table], equal=False, expand=True),
            title="[bold blue]VHS Restore Pipeline[/]",
            subtitle=header,
            border_style="blue",
        )

    def _refresh(self) -> None:
        self._live.update(self._render())

    # ── summary ───────────────────────────────────────────────────────────────

    def show_summary(
        self, passed: int, total: int, output: Path, report: Path,
        compare: Path | None = None,
    ) -> None:
        color = "green" if passed == total else "yellow"
        lines = [
            f"[{color}]Gates: {passed}/{total} passed[/]",
            f"Output:   {output}",
            f"Report:   {report}",
        ]
        if compare:
            lines.append(f"Compare:  [cyan]{compare}[/]  ← play this to see before/after")
        _CONSOLE.print()
        _CONSOLE.print(Panel("\n".join(lines), title="[bold green]Done[/]", border_style=color))


# ── Profiles ──────────────────────────────────────────────────────────────────

_PROFILES: dict[str, dict] = {
    "archival": {
        "output_codec": "ffv1",
        "realesrgan_scale": 2,
        "grain_strength": 4,
        "qtgmc_preset": "Slower",
        "desc": "Lossless FFV1/MKV, 2× upscale — best quality, large files",
    },
    "streaming": {
        "output_codec": "h264",
        "output_crf": 18,
        "realesrgan_scale": 2,
        "grain_strength": 2,
        "qtgmc_preset": "Slow",
        "desc": "H.264/MP4 CRF 18, 2× upscale — good quality, shareable files",
    },
    "preview": {
        "output_codec": "h264",
        "output_crf": 28,
        "realesrgan_scale": 1,
        "grain_strength": 0,
        "qtgmc_preset": "Fast",
        "desc": "H.264/MP4 CRF 28, no upscale — fast preview, small files",
    },
}


# ── Analyze command ───────────────────────────────────────────────────────────



def cmd_analyze(args: argparse.Namespace) -> None:
    import tempfile

    input_path = args.input.resolve()
    if not input_path.exists():
        _CONSOLE.print(f"[red]error:[/] file not found: {input_path}")
        sys.exit(1)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        TimeElapsedColumn(),
        TextColumn("{task.fields[note]}"),
        console=_CONSOLE,
    )

    with Live(
        Panel(progress, title=f"[bold blue]Analyzing[/]  {input_path.name}", border_style="blue"),
        console=_CONSOLE,
        refresh_per_second=8,
    ) as live:

        def refresh(prog):
            live.update(Panel(prog, title=f"[bold blue]Analyzing[/]  {input_path.name}",
                              border_style="blue"))

        # ── Step 1: probe ─────────────────────────────────────────────────────
        t1 = progress.add_task("Probing video properties", note="")

        def probe(entries: str, stream: str = "v:0") -> str:
            r = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", stream,
                "-show_entries", entries, "-of", "csv=p=0", str(input_path),
            ], capture_output=True, text=True)
            return r.stdout.strip()

        try:
            w, h  = probe("stream=width,height").split(",")
            fps   = probe("stream=r_frame_rate")
            codec = probe("stream=codec_name")
            field = probe("stream=field_order")
            dur   = float(subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(input_path),
            ], capture_output=True, text=True).stdout.strip())
        except Exception as e:
            _CONSOLE.print(f"[red]error:[/] could not probe video: {e}")
            sys.exit(1)

        mins, secs = divmod(int(dur), 60)
        hrs, mins  = divmod(mins, 60)
        dur_str    = f"{hrs}:{mins:02d}:{secs:02d}" if hrs else f"{mins}:{secs:02d}"
        interlaced = field not in ("progressive", "unknown", "")
        progress.update(t1, completed=True, total=1,
                        note=f"[dim]{w}×{h}  {dur_str}  {'interlaced' if interlaced else 'progressive'}[/]")
        refresh(progress)

        # ── Step 2: extract frame ─────────────────────────────────────────────
        t2 = progress.add_task("Extracting sample frame", note="")
        tmp_frame: Path | None = None
        mid_frame = int(dur / 2 * 25)
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                tmp_frame = Path(f.name)
            _extract_frame(input_path, tmp_frame, frame_n=mid_frame)
            progress.update(t2, completed=True, total=1,
                            note=f"[dim]frame {mid_frame}[/]")
        except Exception as e:
            progress.update(t2, completed=True, total=1, note=f"[yellow]skipped ({e})[/]")
            tmp_frame = None
        refresh(progress)

        # ── Step 3: BRISQUE ───────────────────────────────────────────────────
        t3 = progress.add_task("Measuring noise (BRISQUE)", note="")
        noise_label = "unavailable"
        noise_val: float | None = None
        try:
            import piq
            from torchvision.io import read_image
            if tmp_frame is None:
                raise RuntimeError("no frame")
            img = read_image(str(tmp_frame)).float().div(255.0).unsqueeze(0)
            noise_val = piq.brisque(img, data_range=1.0).item()
            if noise_val < 20:
                noise_label = f"{noise_val:.1f}  (clean)"
                note_markup = f"[green]{noise_label}[/]"
            elif noise_val < 40:
                noise_label = f"{noise_val:.1f}  (moderate)"
                note_markup = f"[yellow]{noise_label}[/]"
            else:
                noise_label = f"{noise_val:.1f}  (noisy)"
                note_markup = f"[red]{noise_label}[/]"
            progress.update(t3, completed=True, total=1, note=note_markup)
        except Exception as e:
            progress.update(t3, completed=True, total=1, note=f"[yellow]skipped ({e})[/]")
        finally:
            if tmp_frame:
                tmp_frame.unlink(missing_ok=True)
        refresh(progress)

    # ── Build recommendations ─────────────────────────────────────────────────
    nv = noise_val or 0
    knlm = 0.8 if nv < 20 else (2.0 if nv >= 40 else 1.2)

    # Each effect: (flag, description, recommended_on, reason)
    effects = [
        ("--skip-deinterlace" if not interlaced else None,
         "Deinterlacing",      interlaced,        "source is interlaced" if interlaced else "source is already progressive"),
        (f"--knlm-h {knlm:.1f}",
         "Noise reduction",    True,              f"BRISQUE {nv:.1f}" if noise_val else "default on"),
        ("--repair-dropouts",
         "Dropout/line repair", nv >= 30,         "high noise suggests tape dropout"),
        (None,
         "Color correction",   True,              "VHS 16–235 levels (always on)"),
        ("--balance-brightness",
         "Brightness balance",  nv >= 40,         "noisy sources often have exposure issues"),
        ("--warmth 0.3",
         "Color warmth",       False,             "opt-in — corrects cool/bluish VHS color cast"),
        ("--cas",
         "CAS sharpening",     False,             "opt-in — gentle contrast-adaptive sharpening"),
        ("--sharpen",
         "Unsharp mask",       False,             "opt-in — stronger sharpening post-upscale"),
        ("--scale 2",
         "AI upscaling",       True,              "2× Real-ESRGAN (default)"),
        ("--fix-ar",
         "Aspect ratio fix",   False,             "opt-in — use if image looks squashed"),
        ("--audio-cleanup",
         "Audio cleanup",      False,             "opt-in — reduces background hiss/hum"),
        ("--stabilize",
         "Stabilization",      False,             "opt-in — requires libvidstab in ffmpeg"),
    ]

    effects_table = Table(box=None, show_header=False, padding=(0, 1))
    effects_table.add_column(width=3)
    effects_table.add_column(width=22)
    effects_table.add_column(width=20, style="dim cyan")
    effects_table.add_column(style="dim")
    for flag, name, on, reason in effects:
        icon  = "[green]✓[/]" if on else "[dim]○[/]"
        ftext = f"[cyan]{flag}[/]" if flag and on and flag != "--skip-deinterlace" else (
                f"[dim]{flag}[/]" if flag else "")
        effects_table.add_row(icon, name, ftext, reason)

    # Build suggested full command
    extra_flags = []
    if not interlaced:
        extra_flags.append("--skip-deinterlace")
    extra_flags.append(f"--knlm-h {knlm:.1f}")
    if nv >= 30:
        extra_flags.append("--repair-dropouts")
    if nv >= 40:
        extra_flags.append("--balance-brightness")

    base_cmd = f"python pipeline/restore.py restore {input_path.name}"
    flags_str = " ".join(extra_flags)

    fstr = flags_str.strip()
    sep  = (" " + fstr) if fstr else ""
    test_10  = f"{base_cmd} preview.mp4 --test --test-sample{sep}"
    test_30  = f"{base_cmd} preview.mp4 --test --test-sample --test-duration 30{sep}"
    full_cmd = f"{base_cmd} output.mkv{sep}"

    test_table = Table(box=None, show_header=False, padding=(0, 1))
    test_table.add_column(style="dim", width=12)
    test_table.add_column(style="cyan")
    test_table.add_row("10s preview",  test_10)
    test_table.add_row("30s preview",  test_30)
    test_table.add_row("Full restore", full_cmd)

    # Properties table
    info = Table(box=None, show_header=False, padding=(0, 2))
    info.add_column(style="dim", width=14)
    info.add_column()
    info.add_row("File",        input_path.name)
    info.add_row("Resolution",  f"{w}×{h}  ({fps} fps)")
    info.add_row("Codec",       codec)
    info.add_row("Duration",    dur_str)
    info.add_row("Interlaced",  f"{'yes' if interlaced else 'no'}  ({field or 'unknown'})")
    info.add_row("Noise",       noise_label)

    profiles_table = Table(box=None, show_header=False, padding=(0, 2))
    profiles_table.add_column(style="cyan", width=20)
    profiles_table.add_column(style="dim")
    for name, prof in _PROFILES.items():
        profiles_table.add_row(f"--profile {name}", prof["desc"])

    _CONSOLE.print()
    _CONSOLE.print(Panel(info, title="[bold]Video Properties[/]", border_style="blue"))
    _CONSOLE.print(Panel(effects_table, title="[bold]Effects[/]  [dim](✓ recommended  ○ optional)[/]",
                         border_style="blue"))
    _CONSOLE.print(Panel(profiles_table, title="[bold]Output Profiles[/]", border_style="dim"))
    _CONSOLE.print(Panel(test_table, title="[bold]Commands[/]", border_style="green"))

    if args.json:
        data = {
            "file": str(input_path),
            "width": int(w), "height": int(h),
            "fps": fps, "codec": codec,
            "duration_sec": dur,
            "interlaced": interlaced,
            "field_order": field,
            "brisque": round(noise_val, 2) if noise_val is not None else None,
            "recommended_flags": extra_flags,
            "commands": {"test_10": test_10, "test_30": test_30, "full": full_cmd},
        }
        _CONSOLE.print_json(json.dumps(data))


# ── Restore command ───────────────────────────────────────────────────────────

def _restore_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "restore",
        help="Run the full restoration pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=textwrap.dedent("""\
            profiles (--profile overrides individual flags):
              archival   FFV1/MKV lossless, 2× upscale, grain=4
              streaming  H.264/MP4 CRF 18, 2× upscale, grain=2
              preview    H.264/MP4 CRF 28, no upscale, fast preset

            examples:
              %(prog)s capture.mkv output.mkv
              %(prog)s capture.mkv output.mp4 --profile streaming
              %(prog)s capture.mkv output.mkv --codec h265 --crf 22 --output-res 1080p
              %(prog)s capture.mkv preview.mp4 --profile preview --test --test-sample
        """),
    )
    p.add_argument("input",  type=Path, help="Source video file")
    p.add_argument("output", type=Path, help="Output file (.mkv for ffv1/prores, .mp4 for h264/h265)")

    p.add_argument("--profile", choices=list(_PROFILES), metavar="PROFILE",
                   help=f"Preset: {', '.join(_PROFILES)}")
    p.add_argument("--dry-run", action="store_true",
                   help="Print resolved config without processing")

    g = p.add_argument_group("Output format")
    g.add_argument("--codec", dest="output_codec", default="ffv1",
                   choices=["ffv1", "prores", "h264", "h265"],
                   help="Video codec")
    g.add_argument("--crf", type=int, default=18, dest="output_crf",
                   metavar="0-51",
                   help="Quality for h264/h265 (lower = better; 18=high, 28=medium)")
    g.add_argument("--output-res", default="native", dest="output_res",
                   metavar="RES",
                   help="Downscale after upscaling: native | 720p | 1080p | 4k | WxH")
    g.add_argument("--grain", type=int, default=4, dest="grain_strength",
                   metavar="0-10",
                   help="Film grain synthesis strength (0 = off)")
    g.add_argument("--keep-intermediates", action="store_true")

    g = p.add_argument_group("Effects — on by default (use --skip-X to disable)")
    g.add_argument("--skip-deinterlace", action="store_true",
                   help="Skip QTGMC deinterlacing (source is already progressive)")
    g.add_argument("--skip-denoise", action="store_true",
                   help="Skip noise reduction")
    g.add_argument("--skip-color", action="store_true",
                   help="Skip VHS levels correction (16–235)")
    g.add_argument("--qtgmc-preset", default="Slower",
                   choices=["Draft", "Fast", "Medium", "Slow", "Slower", "Placebo"],
                   help="QTGMC quality preset (slower = better)")

    g = p.add_argument_group("Effects — off by default (use flag to enable)")
    g.add_argument("--stabilize", action="store_true",
                   help="Video stabilization (requires libvidstab in ffmpeg)")
    g.add_argument("--repair-dropouts", action="store_true", dest="repair_dropouts",
                   help="VHS dropout / horizontal-line artifact repair")
    g.add_argument("--balance-brightness", action="store_true", dest="balance_brightness",
                   help="Auto-normalize brightness and contrast")
    g.add_argument("--sharpen", action="store_true",
                   help="Unsharp mask after upscaling")
    g.add_argument("--cas", action="store_true",
                   help="Contrast Adaptive Sharpening after upscaling (gentler than --sharpen)")
    g.add_argument("--fix-ar", action="store_true", dest="fix_ar",
                   help="Correct display aspect ratio (anamorphic / square-pixel mismatch)")
    g.add_argument("--audio-cleanup", action="store_true", dest="audio_cleanup",
                   help="Spectral noise reduction on audio track")

    g = p.add_argument_group("Effect tuning")
    g.add_argument("--knlm-h", type=float, default=1.2, metavar="STRENGTH",
                   help="Denoise strength: 0.8=light, 1.2=balanced, 2.0=aggressive")
    g.add_argument("--knlm-d", type=int, default=1,
                   help="Temporal radius in frames")
    g.add_argument("--gpu", type=int, default=0, dest="gpu_device_id")
    g.add_argument("--levels-min-in", type=int, default=16, help="Black point (VHS default: 16)")
    g.add_argument("--levels-max-in", type=int, default=235, help="White point (VHS default: 235)")
    g.add_argument("--saturation", type=float, default=1.0,
                   help="Color saturation multiplier (1.0=unchanged, 1.3=boost)")
    g.add_argument("--sharpen-amount", type=float, default=1.5, dest="sharpen_amount",
                   help="Unsharp mask strength (0.5–3.0)")
    g.add_argument("--warmth", type=float, default=0.0,
                   help="Color temperature: -1.0=cooler, 0=neutral, +1.0=warmer")
    g.add_argument("--cas-strength", type=float, default=0.6, dest="cas_strength",
                   help="Contrast Adaptive Sharpening strength (0.0–1.0)")
    g.add_argument("--ar-target", default="4:3", dest="ar_target",
                   help="Display aspect ratio for --fix-ar (e.g. 4:3, 16:9)")
    g.add_argument("--audio-cleanup-db", type=float, default=10.0, dest="audio_cleanup_db",
                   help="Audio noise floor reduction in dB (5=gentle, 20=strong)")

    g = p.add_argument_group("Upscaling")
    g.add_argument("--scale", type=int, default=2, choices=[1, 2, 4],
                   dest="realesrgan_scale",
                   help="AI upscale factor (1 = skip upscaling)")
    g.add_argument("--model", default="realesr-general-x4v3",
                   choices=["realesr-general-x4v3", "RealESRGAN_x4plus"],
                   dest="realesrgan_model")
    g.add_argument("--tile", type=int, default=256, dest="realesrgan_tile",
                   help="Tile size for VRAM-limited GPUs (0 = disable)")
    g.add_argument("--realesrgan-dir", default="~/Real-ESRGAN",
                   help="Path to cloned Real-ESRGAN repo")

    g = p.add_argument_group("Test mode")
    g.add_argument("--test", action="store_true", dest="test_mode",
                   help="Process a short sample clip instead of the full video")
    g.add_argument("--test-sample", action="store_true",
                   help="Pick the sample clip from the middle of the video")
    g.add_argument("--test-start", default="00:05:00",
                   help="Sample start time HH:MM:SS (overridden by --test-sample)")
    g.add_argument("--test-duration", type=int, default=10,
                   help="Sample length in seconds")
    g.add_argument("--compare", action="store_true",
                   help="Generate a side-by-side before/after comparison video alongside output")

    g = p.add_argument_group("Quality gate thresholds")
    g.add_argument("--gate-progressive-pct", type=float, default=0.95,
                   dest="gate_min_progressive_pct")
    g.add_argument("--gate-brisque-delta", type=float, default=5.0,
                   dest="gate_max_brisque_delta")
    g.add_argument("--gate-ssim", type=float, default=65.0, dest="gate_min_vmaf")
    g.add_argument("--gate-niqe", type=float, default=6.0, dest="gate_max_niqe")

    return p


def cmd_restore(args: argparse.Namespace) -> None:
    input_path  = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    skip = {"input", "output", "verbose", "profile", "dry_run", "subcommand"}
    cfg_kwargs = {k: v for k, v in vars(args).items() if k not in skip}

    # Profile sets defaults; any explicitly passed flag wins over the profile
    if args.profile:
        profile_vals = {k: v for k, v in _PROFILES[args.profile].items() if k != "desc"}
        cfg_defaults = asdict(Config())
        # A key is "explicitly set" when it differs from the Config dataclass default
        overridden = {k for k, v in cfg_kwargs.items() if v != cfg_defaults.get(k)}
        cfg_kwargs = {**profile_vals, **{k: v for k, v in cfg_kwargs.items() if k in overridden}}

    cfg = Config(**cfg_kwargs)

    if args.dry_run:
        print("Resolved configuration:")
        for k, v in asdict(cfg).items():
            print(f"  {k:<30} {v}")
        print(f"\nInput:   {input_path}")
        print(f"Output:  {output_path}")
        return

    run_pipeline(input_path, output_path, cfg)


# ── Enhance command ───────────────────────────────────────────────────────────

def _build_enhance_filters(args: argparse.Namespace) -> list[str]:
    """Build ffmpeg filter chain from enhance args. Order matches the full pipeline."""
    f = []
    if args.balance_brightness:
        f.append("normalize=blackpt=black:whitept=white:smoothing=5")
    if args.warmth != 0.0:
        w = args.warmth * 0.15
        rr = round(max(0.5, min(1.5, 1.0 + w)), 3)
        bb = round(max(0.5, min(1.5, 1.0 - w)), 3)
        f.append(f"colorchannelmixer=rr={rr}:bb={bb}")
    if args.saturation != 1.0:
        f.append(f"eq=saturation={args.saturation:.2f}")
    if args.cas:
        f.append(f"cas=strength={args.cas_strength:.2f}")
    if args.sharpen:
        f.append(f"unsharp=lx=5:ly=5:la={args.sharpen_amount:.1f}:cx=3:cy=3:ca=0.0")
    return f


def cmd_enhance(args: argparse.Namespace) -> None:
    input_path  = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        _CONSOLE.print(f"[red]error:[/] file not found: {input_path}")
        sys.exit(1)

    filters = _build_enhance_filters(args)

    if not filters:
        _CONSOLE.print(
            "[yellow]No effects selected — nothing to do.[/]\n"
            "Pass at least one flag: --balance-brightness, --warmth, --saturation, --cas, --sharpen"
        )
        sys.exit(1)

    applied = ", ".join(filter(None, [
        "brightness" if args.balance_brightness else "",
        f"warmth {args.warmth:+.1f}" if args.warmth != 0.0 else "",
        f"saturation ×{args.saturation:.1f}" if args.saturation != 1.0 else "",
        f"CAS {args.cas_strength:.1f}" if args.cas else "",
        f"sharpen {args.sharpen_amount:.1f}" if args.sharpen else "",
    ]))

    if args.dry_run:
        _CONSOLE.print()
        _CONSOLE.print(Panel(
            "\n".join(f"[cyan]{f}[/]" for f in filters),
            title=f"[bold]Filter chain[/]  ({applied})",
            border_style="blue",
        ))
        return

    if args.output_codec == "h264":
        vcodec = ["-c:v", "libx264", "-crf", str(args.output_crf),
                  "-preset", "slow", "-pix_fmt", "yuv420p"]
    elif args.output_codec == "h265":
        vcodec = ["-c:v", "libx265", "-crf", str(args.output_crf),
                  "-preset", "slow", "-pix_fmt", "yuv420p"]
    elif args.output_codec == "ffv1":
        vcodec = ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv422p10le"]
    else:
        vcodec = ["-c:v", "prores_ks", "-profile:v", "hq", "-pix_fmt", "yuv422p10le"]

    total = _probe_frame_count(input_path)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=32),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=False,
        console=_CONSOLE,
    )

    with Live(
        Panel(progress,
              title=f"[bold blue]Enhance[/]  {input_path.name}",
              subtitle=f"[dim]{applied}[/]",
              border_style="blue"),
        console=_CONSOLE,
        refresh_per_second=8,
    ) as live:

        def _refresh():
            live.update(Panel(
                progress,
                title=f"[bold blue]Enhance[/]  {input_path.name}",
                subtitle=f"[dim]{applied}[/]",
                border_style="blue",
            ))

        tid = progress.add_task(f"[cyan]Applying filters[/]", total=total or None)

        def _on_progress(line: str) -> None:
            if m := re.search(r"frame=\s*(\d+)", line):
                progress.update(tid, completed=int(m.group(1)), total=total)
                _refresh()

        _run_tracking(
            ["ffmpeg", "-y", "-i", str(input_path),
             "-vf", ",".join(filters),
             *vcodec, "-c:a", "copy", str(output_path)],
            on_stderr=_on_progress,
        )
        progress.update(tid, completed=total or 1, total=total or 1,
                        description="[green]Done[/]  ✓")
        _refresh()

    _CONSOLE.print()
    _CONSOLE.print(Panel(
        f"[green]✓[/]  {output_path}\n[dim]Applied: {applied}[/]",
        border_style="green",
    ))


def _enhance_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "enhance",
        help="Apply color and sharpness improvements (no VapourSynth required)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=textwrap.dedent("""\
            Each effect is opt-in. Combine freely.

            examples:
              %(prog)s restored.mkv out.mp4 --warmth 0.3 --saturation 1.3
              %(prog)s restored.mkv out.mp4 --cas --balance-brightness
              %(prog)s restored.mkv out.mp4 --warmth 0.4 --saturation 1.25 --cas --cas-strength 0.7
              %(prog)s restored.mkv out.mp4 --sharpen --sharpen-amount 1.2
              %(prog)s restored.mkv out.mp4 --dry-run --warmth 0.3 --cas
        """),
    )
    p.add_argument("input",  type=Path, help="Source video file")
    p.add_argument("output", type=Path, help="Output file")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the filter chain without processing")

    g = p.add_argument_group("Color")
    g.add_argument("--balance-brightness", action="store_true", dest="balance_brightness",
                   help="Auto-normalize brightness and contrast")
    g.add_argument("--warmth", type=float, default=0.0,
                   help="Color temperature: -1.0=cooler, 0=neutral, +1.0=warmer")
    g.add_argument("--saturation", type=float, default=1.0,
                   help="Saturation multiplier (1.0=unchanged, 1.3=boost)")

    g = p.add_argument_group("Sharpness")
    g.add_argument("--cas", action="store_true",
                   help="Contrast Adaptive Sharpening (gentle, natural-looking)")
    g.add_argument("--cas-strength", type=float, default=0.6, dest="cas_strength",
                   help="CAS strength (0.0–1.0)")
    g.add_argument("--sharpen", action="store_true",
                   help="Unsharp mask (stronger than CAS)")
    g.add_argument("--sharpen-amount", type=float, default=1.5, dest="sharpen_amount",
                   help="Unsharp mask strength (0.5–3.0)")

    g = p.add_argument_group("Output format")
    g.add_argument("--codec", dest="output_codec", default="h264",
                   choices=["ffv1", "prores", "h264", "h265"],
                   help="Video codec")
    g.add_argument("--crf", type=int, default=18, dest="output_crf",
                   metavar="0-51",
                   help="Quality for h264/h265")

    return p


# ── Trim command ──────────────────────────────────────────────────────────────

def cmd_trim(args: argparse.Namespace) -> None:
    input_path  = args.input.resolve()
    output_path = args.output.resolve() if args.output else None

    if not input_path.exists():
        _CONSOLE.print(f"[red]error:[/] file not found: {input_path}")
        sys.exit(1)

    dur = _probe_duration(input_path)

    _CONSOLE.print()

    with _CONSOLE.status("[bold]Detecting black frames…[/]"):
        result = subprocess.run(
            [
                "ffmpeg", "-i", str(input_path),
                "-vf", f"blackdetect=d={args.min_duration}:pix_th={args.threshold}",
                "-f", "null", "-",
            ],
            capture_output=True, text=True,
        )
        raw = result.stderr

    # Parse blackdetect output
    intervals = []
    for m in re.finditer(
        r"black_start:([\d.]+)\s+black_end:([\d.]+)\s+black_duration:([\d.]+)", raw
    ):
        intervals.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))

    if not intervals:
        _CONSOLE.print(Panel(
            "[green]No black screens detected.[/]  Nothing to trim.",
            border_style="green",
        ))
        return

    # Determine content window (skip leading/trailing black)
    content_start = 0.0
    content_end   = dur

    if intervals[0][0] < 0.5:          # black at beginning
        content_start = intervals[0][1]
        intervals = intervals[1:]

    if intervals and intervals[-1][1] >= dur - 0.5:   # black at end
        content_end = intervals[-1][0]
        intervals   = intervals[:-1]

    # Build results table
    table = Table(title="Detected Black Regions", border_style="blue")
    table.add_column("Start",    style="cyan",  width=12)
    table.add_column("End",      style="cyan",  width=12)
    table.add_column("Duration", style="yellow", width=10)
    table.add_column("Position", style="dim",   width=16)

    def fmt_ts(s: float) -> str:
        m, sec = divmod(s, 60)
        return f"{int(m):02d}:{sec:05.2f}"

    def position_label(start: float, end: float) -> str:
        if start < 0.5:
            return "▶ leading"
        if end >= dur - 0.5:
            return "◀ trailing"
        return "middle"

    # Re-parse all for display
    all_intervals = []
    for m2 in re.finditer(
        r"black_start:([\d.]+)\s+black_end:([\d.]+)\s+black_duration:([\d.]+)", raw
    ):
        s, e, d = float(m2.group(1)), float(m2.group(2)), float(m2.group(3))
        all_intervals.append((s, e, d))

    for s, e, d in all_intervals:
        label = position_label(s, e)
        style = "" if "middle" in label else "bold"
        table.add_row(fmt_ts(s), fmt_ts(e), f"{d:.2f}s", label, style=style)

    trimmed_dur = content_end - content_start
    summary = Table(box=None, show_header=False, padding=(0, 2))
    summary.add_column(style="dim", width=16)
    summary.add_column()
    summary.add_row("Original",      f"{fmt_ts(dur)}  ({dur:.1f}s)")
    summary.add_row("Content start", fmt_ts(content_start))
    summary.add_row("Content end",   fmt_ts(content_end))
    summary.add_row("Trimmed",       f"[green]{fmt_ts(trimmed_dur)}  ({trimmed_dur:.1f}s)[/]")
    summary.add_row("Removed",       f"[yellow]{dur - trimmed_dur:.1f}s[/]")

    _CONSOLE.print(table)
    _CONSOLE.print()
    _CONSOLE.print(Panel(summary, title="[bold]Trim Summary[/]", border_style="blue"))

    if args.dry_run:
        _CONSOLE.print("\n[dim]--dry-run: no output written[/]")
        return

    if output_path is None:
        stem = input_path.stem + "_trimmed"
        output_path = input_path.with_name(stem + input_path.suffix)

    _CONSOLE.print(f"\nWriting [cyan]{output_path.name}[/]…")
    with _CONSOLE.status("Trimming…"):
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(content_start),
            "-to", str(content_end),
            "-i", str(input_path),
            "-c", "copy",
            str(output_path),
        ], capture_output=True, check=True)

    _CONSOLE.print(Panel(
        f"[green]✓[/]  {output_path.name}  ({trimmed_dur:.1f}s)",
        border_style="green",
    ))


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    global _VERBOSE

    p = argparse.ArgumentParser(
        prog="restore.py",
        description="VHS digitization post-processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            commands:
              analyze   Probe a video file and print recommended restore settings
              trim      Remove leading/trailing black screens from a capture
              restore   Run the full pipeline (deinterlace → denoise → upscale → grain)
              enhance   Apply color/sharpness improvements independently (no VapourSynth)

            quick start:
              python pipeline/restore.py analyze capture.mp4
              python pipeline/restore.py trim capture.mp4
              python pipeline/restore.py restore capture.mp4 output.mkv
              python pipeline/restore.py restore capture.mp4 output.mp4 --profile streaming
              python pipeline/restore.py restore capture.mp4 out.mkv --test --test-sample
              python pipeline/restore.py enhance restored.mkv out.mp4 --warmth 0.3 --saturation 1.3 --cas
        """),
    )
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Show full ffmpeg output during processing")
    sub = p.add_subparsers(dest="subcommand")

    # analyze subcommand
    ap = sub.add_parser("analyze", help="Probe video and print recommended settings",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("input", type=Path, help="Video file to analyze")
    ap.add_argument("--json", action="store_true", help="Output results as JSON")

    # trim subcommand
    tp = sub.add_parser(
        "trim",
        help="Remove leading/trailing black screens",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s capture.mp4                     # auto-name output *_trimmed.mp4
              %(prog)s capture.mp4 clean.mp4           # explicit output path
              %(prog)s capture.mp4 --dry-run           # show what would be trimmed
              %(prog)s capture.mp4 --min-duration 2.0  # only remove blacks ≥ 2s
        """),
    )
    tp.add_argument("input",  type=Path, help="Source video file")
    tp.add_argument("output", type=Path, nargs="?", default=None,
                    help="Output path (default: <input>_trimmed.<ext>)")
    tp.add_argument("--threshold",    type=float, default=0.1,  metavar="0-1",
                    help="Pixel brightness threshold for black detection")
    tp.add_argument("--min-duration", type=float, default=0.5,  dest="min_duration",
                    metavar="SEC",
                    help="Minimum black region duration to consider")
    tp.add_argument("--dry-run", action="store_true",
                    help="Show detected regions without writing output")

    # restore subcommand
    _restore_parser(sub)

    # enhance subcommand
    _enhance_parser(sub)

    args = p.parse_args()

    if args.subcommand is None:
        p.print_help()
        sys.exit(0)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    _VERBOSE = args.verbose

    if args.subcommand == "analyze":
        cmd_analyze(args)
    elif args.subcommand == "trim":
        cmd_trim(args)
    elif args.subcommand == "restore":
        _check_env()
        if getattr(args, "test_sample", False) and not getattr(args, "test_mode", False):
            args.test_mode = True
        cmd_restore(args)
    elif args.subcommand == "enhance":
        cmd_enhance(args)


if __name__ == "__main__":
    main()
