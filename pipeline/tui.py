#!/usr/bin/env python3
"""
VHS Pipeline — Terminal User Interface

k9s / lazygit-inspired TUI: keyboard-driven, panel-based.

Usage:
    python pipeline/tui.py
    bash vhs.sh              (no arguments)
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from rich.text import Text
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, ScrollableContainer, Vertical
    from textual.screen import Screen
    from textual.widgets import (
        Button,
        Footer,
        Header,
        Input,
        Label,
        ListItem,
        ListView,
        RichLog,
        Select,
        Static,
        Switch,
    )
except ImportError:
    sys.exit(
        "textual not installed.\n"
        "Run:  pip install 'textual>=0.60.0'\n"
        "Or:   bash pipeline/setup_ubuntu.sh"
    )


# ── Constants ─────────────────────────────────────────────────────────────────

VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov", ".ts", ".m2ts", ".mts", ".mpg", ".mpeg"}
SCRIPT_DIR = Path(__file__).parent
RESTORE_PY = SCRIPT_DIR / "restore.py"

ACTIONS = [
    (
        "analyze",
        "Analyze",
        "Probe the file — resolution, duration, interlacing, noise level (BRISQUE).\n\n"
        "Prints recommended restore settings.\n"
        "No video processing performed.",
    ),
    (
        "trim",
        "Trim",
        "Detect and remove blank screens from the start and end of the capture.\n\n"
        "Fast — uses ffmpeg stream copy (lossless).\n"
        "Saves a *_trimmed file alongside the original.",
    ),
    (
        "preview",
        "Preview  (quick test)",
        "Fast color / effects test on a short clip — ffmpeg only.\n\n"
        "No VapourSynth or Real-ESRGAN. Great for tuning looks\n"
        "before committing to a full restore.",
    ),
    (
        "restore",
        "Restore  (full pipeline)",
        "Full AI pipeline:\n"
        "  QTGMC deinterlace → KNLMeansCL denoise\n"
        "  → Real-ESRGAN upscale → film grain synthesis\n\n"
        "Use the 'Test clip' option to run on a short sample first.",
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _safe_float(v: object, default: float) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_int(v: object, default: int) -> int:
    try:
        return int(float(v))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


# ── Shared state ──────────────────────────────────────────────────────────────

@dataclass
class State:
    input_path:  Path | None = None
    output_path: Path | None = None
    action:      str         = "restore"

    # ── Restore settings ──────────────────────────────────────────────────────
    skip_deinterlace:   bool  = False
    qtgmc_preset:       str   = "Slower"
    skip_denoise:       bool  = False
    knlm_h:             float = 1.2
    knlm_d:             int   = 1
    repair_dropouts:    bool  = False
    skip_color:         bool  = False
    saturation:         float = 1.0
    warmth:             float = 0.0
    balance_brightness: bool  = False
    cas:                bool  = False
    cas_strength:       float = 0.6
    sharpen:            bool  = False
    sharpen_amount:     float = 1.5
    realesrgan_scale:   int   = 2
    realesrgan_model:   str   = "realesr-general-x4v3"
    realesrgan_tile:    int   = 256
    output_codec:       str   = "ffv1"
    grain_strength:     int   = 4
    keep_intermediates: bool  = False
    compare:            bool  = False
    audio_cleanup:      bool  = False
    audio_cleanup_db:   float = 10.0
    stabilize:          bool  = False
    fix_ar:             bool  = False
    ar_target:          str   = "4:3"
    run_test_clip:      bool  = False
    test_sample:        bool  = True
    test_start:         str   = "00:05:00"
    test_duration:      int   = 30

    # ── Preview (test subcommand — all opt-in) ────────────────────────────────
    prev_qtgmc:          bool  = True
    prev_qtgmc_preset:   str   = "Slow"
    prev_knlm:           bool  = True
    prev_knlm_h:         float = 1.2
    prev_deinterlace:    bool  = False
    prev_levels:         bool  = True
    prev_denoise:        bool  = False
    prev_denoise_str:    float = 1.0
    prev_repair:         bool  = False
    prev_deflicker:      bool  = False
    prev_balance:        bool  = False
    prev_warmth:         float = 0.0
    prev_saturation:     float = 1.0
    prev_cas:            bool  = False
    prev_cas_strength:   float = 0.6
    prev_sharpen:        bool  = False
    prev_sharpen_amount: float = 1.5
    prev_grain:          int   = 0
    prev_fix_ar:         bool  = False
    prev_ar_target:      str   = "4:3"
    prev_codec:          str   = "h264"
    prev_test_sample:    bool  = True
    prev_test_start:     str   = "00:05:00"
    prev_test_duration:  int   = 10

    # ── Trim settings ─────────────────────────────────────────────────────────
    trim_threshold:    float = 0.1
    trim_min_duration: float = 0.5

    def auto_output(self) -> Path:
        if self.input_path is None:
            return Path("output.mkv")
        stem = self.input_path.stem
        if self.action == "preview":
            return self.input_path.with_name(f"{stem}_preview.mp4")
        if self.action == "trim":
            return self.input_path.with_name(f"{stem}_trimmed{self.input_path.suffix}")
        ext = ".mp4" if self.output_codec in ("h264", "h265") else ".mkv"
        return self.input_path.with_name(f"{stem}_restored{ext}")

    def build_cmd(self) -> list[str]:
        py  = sys.executable
        inp = str(self.input_path)
        out = str(self.output_path or self.auto_output())

        if self.action == "analyze":
            return [py, str(RESTORE_PY), "analyze", inp]

        if self.action == "trim":
            return [py, str(RESTORE_PY), "trim", inp, out,
                    "--threshold", str(self.trim_threshold),
                    "--min-duration", str(self.trim_min_duration)]

        if self.action == "preview":
            c = [py, str(RESTORE_PY), "test", inp, out]
            if self.prev_qtgmc:
                c += ["--qtgmc", "--qtgmc-preset", self.prev_qtgmc_preset]
            if self.prev_knlm:
                c += ["--knlm", "--knlm-h", str(self.prev_knlm_h)]
            if self.prev_deinterlace:  c.append("--deinterlace")
            if self.prev_levels:       c.append("--levels")
            if self.prev_denoise:      c += ["--denoise", "--denoise-strength", str(self.prev_denoise_str)]
            if self.prev_repair:       c.append("--repair-dropouts")
            if self.prev_deflicker:    c.append("--deflicker")
            if self.prev_balance:      c.append("--balance-brightness")
            if self.prev_warmth != 0:  c += ["--warmth", str(self.prev_warmth)]
            if self.prev_saturation != 1.0: c += ["--saturation", str(self.prev_saturation)]
            if self.prev_cas:          c += ["--cas", "--cas-strength", str(self.prev_cas_strength)]
            if self.prev_sharpen:      c += ["--sharpen", "--sharpen-amount", str(self.prev_sharpen_amount)]
            if self.prev_grain > 0:    c += ["--grain", str(self.prev_grain)]
            if self.prev_fix_ar:       c += ["--fix-ar", "--ar-target", self.prev_ar_target]
            c += ["--codec", self.prev_codec]
            if self.prev_test_sample:
                c += ["--test-sample", "--test-duration", str(self.prev_test_duration)]
            else:
                c += ["--test-start", self.prev_test_start,
                      "--test-duration", str(self.prev_test_duration)]
            return c

        # restore
        c = [py, str(RESTORE_PY), "restore", inp, out]
        if self.skip_deinterlace:   c.append("--skip-deinterlace")
        else:                       c += ["--qtgmc-preset", self.qtgmc_preset]
        if self.skip_denoise:       c.append("--skip-denoise")
        else:                       c += ["--knlm-h", str(self.knlm_h), "--knlm-d", str(self.knlm_d)]
        if self.skip_color:         c.append("--skip-color")
        if self.repair_dropouts:    c.append("--repair-dropouts")
        if self.balance_brightness: c.append("--balance-brightness")
        if self.stabilize:          c.append("--stabilize")
        if self.fix_ar:             c += ["--fix-ar", "--ar-target", self.ar_target]
        if self.audio_cleanup:      c += ["--audio-cleanup", "--audio-cleanup-db", str(self.audio_cleanup_db)]
        if self.cas:                c += ["--cas", "--cas-strength", str(self.cas_strength)]
        if self.sharpen:            c += ["--sharpen", "--sharpen-amount", str(self.sharpen_amount)]
        if self.saturation != 1.0:  c += ["--saturation", str(self.saturation)]
        if self.warmth != 0.0:      c += ["--warmth", str(self.warmth)]
        c += ["--scale", str(self.realesrgan_scale),
              "--model",  self.realesrgan_model,
              "--tile",   str(self.realesrgan_tile),
              "--codec",  self.output_codec,
              "--grain",  str(self.grain_strength)]
        if self.keep_intermediates: c.append("--keep-intermediates")
        if self.compare:            c.append("--compare")
        if self.run_test_clip:
            c.append("--test")
            if self.test_sample:
                c += ["--test-sample", "--test-duration", str(self.test_duration)]
            else:
                c += ["--test-start", self.test_start, "--test-duration", str(self.test_duration)]
        return c


# ── File picker ───────────────────────────────────────────────────────────────

class FilePickerScreen(Screen):
    BINDINGS = [
        Binding("q",         "app.quit",   "Quit"),
        Binding("escape",    "app.quit",   "Quit"),
        Binding("backspace", "go_up",      "Parent dir"),
    ]

    _cwd: Path
    _entries: list[tuple[str, Path, bool]]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical():
            yield Label("", id="path-label")
            yield ListView(id="file-list")
            yield Static("No file selected", id="file-info")
        yield Footer()

    def on_mount(self) -> None:
        self._cwd     = Path.cwd()
        self._entries = []
        self._refresh()

    def _refresh(self) -> None:
        self.query_one("#path-label", Label).update(
            f" Path  [cyan]{self._cwd}[/]"
        )
        lv = self.query_one("#file-list", ListView)
        lv.clear()
        self._entries = []

        # Parent dir entry
        if self._cwd.parent != self._cwd:
            self._entries.append(("../", self._cwd.parent, True))

        try:
            children = sorted(
                self._cwd.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
            for child in children:
                if child.name.startswith("."):
                    continue
                if child.is_dir():
                    self._entries.append((f"{child.name}/", child, True))
                elif child.suffix.lower() in VIDEO_EXTS:
                    try:
                        size = _fmt_size(child.stat().st_size)
                    except OSError:
                        size = ""
                    self._entries.append((f"{child.name}  ({size})", child, False))
        except PermissionError:
            pass

        for display, path, is_dir in self._entries:
            icon = "📁" if is_dir else "📼"
            lv.append(ListItem(Label(f" {icon}  {display}")))

        self.query_one("#file-info", Static).update("No file selected")

    @on(ListView.Highlighted)
    def on_highlighted(self, event: ListView.Highlighted) -> None:
        idx = event.list_view.index
        if idx is None or idx >= len(self._entries):
            return
        display, path, is_dir = self._entries[idx]
        if not is_dir:
            self.query_one("#file-info", Static).update(
                f" [bold]{path.name}[/]"
            )

    @on(ListView.Selected)
    def on_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is None or idx >= len(self._entries):
            return
        _, path, is_dir = self._entries[idx]
        if is_dir:
            self._cwd = path
            self._refresh()
        else:
            self.app.state.input_path = path
            self.app.push_screen(ActionScreen())

    def action_go_up(self) -> None:
        if self._cwd.parent != self._cwd:
            self._cwd = self._cwd.parent
            self._refresh()


# ── Action menu ───────────────────────────────────────────────────────────────

class ActionScreen(Screen):
    BINDINGS = [
        Binding("escape", "pop_screen", "Back"),
        Binding("q",      "app.quit",   "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="action-layout"):
            with Vertical(id="action-list-pane"):
                yield Label(" Action", id="action-heading")
                items = [ListItem(Label(f"  {title}")) for _, title, _ in ACTIONS]
                yield ListView(*items, id="action-list")
            with Vertical(id="action-desc-pane"):
                yield Static("", id="action-desc")
        yield Footer()

    def on_mount(self) -> None:
        fname = self.app.state.input_path
        self.sub_title = fname.name if fname else ""
        self._update_desc(0)

    @on(ListView.Highlighted)
    def on_highlighted(self, event: ListView.Highlighted) -> None:
        self._update_desc(event.list_view.index)

    @on(ListView.Selected)
    def on_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is None or idx >= len(ACTIONS):
            return
        action, _, _ = ACTIONS[idx]
        self.app.state.action = action
        if action == "analyze":
            self.app.push_screen(RunScreen())
        else:
            self.app.push_screen(SettingsScreen())

    def _update_desc(self, idx: int | None) -> None:
        if idx is None or idx >= len(ACTIONS):
            return
        _, title, desc = ACTIONS[idx]
        self.query_one("#action-desc", Static).update(
            f"\n [bold]{title}[/]\n ─────────────────────────\n\n {desc.replace(chr(10), chr(10) + ' ')}"
        )


# ── Settings form ─────────────────────────────────────────────────────────────

class SettingsScreen(Screen):
    BINDINGS = [
        Binding("escape",  "pop_screen", "Back"),
        Binding("ctrl+r",  "run",        "Run"),
        Binding("q",       "app.quit",   "Quit"),
    ]

    def compose(self) -> ComposeResult:
        s = self.app.state
        yield Header(show_clock=False)
        with ScrollableContainer(id="settings-scroll"):
            # ── Output path ───────────────────────────────────────────────────
            yield Label(" OUTPUT", classes="sec-head")
            with Horizontal(classes="row"):
                yield Label("Output file", classes="lbl")
                yield Input(
                    value=str(s.output_path or s.auto_output()),
                    id="output-path",
                    classes="wide-input",
                )

            # ── Action-specific sections ──────────────────────────────────────
            if s.action == "trim":
                yield from self._trim_section()
            elif s.action == "preview":
                yield from self._clip_section(is_preview=True)
                yield from self._preview_section()
            elif s.action == "restore":
                yield from self._restore_section()
                yield from self._clip_section(is_preview=False)

        with Horizontal(id="btn-bar"):
            yield Button("▶  Run", variant="success", id="run-btn")
            yield Button("← Back", id="back-btn")
        yield Footer()

    def on_mount(self) -> None:
        s = self.app.state
        label = {"trim": "Trim", "preview": "Preview", "restore": "Restore"}
        self.sub_title = f"{s.input_path.name if s.input_path else ''}  ·  {label.get(s.action, s.action)}"

    # ── Section generators ────────────────────────────────────────────────────

    def _trim_section(self) -> ComposeResult:
        s = self.app.state
        yield Label(" TRIM DETECTION", classes="sec-head")
        with Horizontal(classes="row"):
            yield Label("Threshold", classes="lbl")
            yield Input(value=str(s.trim_threshold), id="trim_threshold", classes="narrow")
            yield Label("0.0–1.0  (0.1 = default)", classes="hint")
        with Horizontal(classes="row"):
            yield Label("Min blank duration", classes="lbl")
            yield Input(value=str(s.trim_min_duration), id="trim_min_duration", classes="narrow")
            yield Label("seconds", classes="hint")

    def _clip_section(self, is_preview: bool) -> ComposeResult:
        s = self.app.state
        sample = s.prev_test_sample if is_preview else s.test_sample
        dur    = s.prev_test_duration if is_preview else s.test_duration
        start  = s.prev_test_start if is_preview else s.test_start

        yield Label(" TEST CLIP", classes="sec-head")
        if not is_preview:
            with Horizontal(classes="row"):
                yield Label("Run on clip only", classes="lbl")
                yield Switch(value=s.run_test_clip, id="run_test_clip")
                yield Label("off = process full video", classes="hint")
        with Horizontal(classes="row"):
            yield Label("Auto-pick from middle", classes="lbl")
            yield Switch(value=sample, id="test_sample")
        with Horizontal(classes="row"):
            yield Label("Duration (seconds)", classes="lbl")
            yield Input(value=str(dur), id="test_duration", classes="narrow")
        with Horizontal(classes="row"):
            yield Label("Start time", classes="lbl")
            yield Input(value=start, id="test_start")
            yield Label("HH:MM:SS  (ignored if auto-pick on)", classes="hint")

    def _preview_section(self) -> ComposeResult:
        s = self.app.state
        yield Label(" VAPOURSYNTH  (high quality — requires venv)", classes="sec-head")
        with Horizontal(classes="row"):
            yield Label("QTGMC deinterlace", classes="lbl")
            yield Switch(value=s.prev_qtgmc, id="prev_qtgmc")
            yield Label("better quality than yadif", classes="hint")
        with Horizontal(classes="row"):
            yield Label("QTGMC preset", classes="lbl")
            yield Select(_qtgmc_opts(), value=s.prev_qtgmc_preset, id="prev_qtgmc_preset")
        with Horizontal(classes="row"):
            yield Label("KNLMeansCL denoise", classes="lbl")
            yield Switch(value=s.prev_knlm, id="prev_knlm")
            yield Label("GPU denoising via OpenCL", classes="hint")
        with Horizontal(classes="row"):
            yield Label("KNLMeansCL strength", classes="lbl")
            yield Input(value=str(s.prev_knlm_h), id="prev_knlm_h", classes="narrow")
            yield Label("0.8=light  1.2=balanced  2.0=heavy", classes="hint")

        yield Label(" VHS CORRECTIONS  (ffmpeg)", classes="sec-head")
        with Horizontal(classes="row"):
            yield Label("Deinterlace (yadif)", classes="lbl")
            yield Switch(value=s.prev_deinterlace, id="prev_deinterlace")
            yield Label("fast alternative to QTGMC", classes="hint")
        with Horizontal(classes="row"):
            yield Label("Color levels 16–235", classes="lbl")
            yield Switch(value=s.prev_levels, id="prev_levels")
            yield Label("biggest visual impact on VHS", classes="hint")
        with Horizontal(classes="row"):
            yield Label("Denoise (hqdn3d)", classes="lbl")
            yield Switch(value=s.prev_denoise, id="prev_denoise")
        with Horizontal(classes="row"):
            yield Label("Denoise strength", classes="lbl")
            yield Input(value=str(s.prev_denoise_str), id="prev_denoise_str", classes="narrow")
            yield Label("0.5=light  1.0=moderate  2.0=heavy", classes="hint")
        with Horizontal(classes="row"):
            yield Label("Repair dropouts", classes="lbl")
            yield Switch(value=s.prev_repair, id="prev_repair")
            yield Label("horizontal-line artifacts", classes="hint")
        with Horizontal(classes="row"):
            yield Label("Deflicker", classes="lbl")
            yield Switch(value=s.prev_deflicker, id="prev_deflicker")
            yield Label("smooth brightness fluctuations", classes="hint")

        yield Label(" COLOR", classes="sec-head")
        with Horizontal(classes="row"):
            yield Label("Balance brightness", classes="lbl")
            yield Switch(value=s.prev_balance, id="prev_balance")
        with Horizontal(classes="row"):
            yield Label("Saturation", classes="lbl")
            yield Input(value=str(s.prev_saturation), id="prev_saturation", classes="narrow")
            yield Label("1.0=unchanged  1.3=boost", classes="hint")
        with Horizontal(classes="row"):
            yield Label("Warmth", classes="lbl")
            yield Input(value=str(s.prev_warmth), id="prev_warmth", classes="narrow")
            yield Label("-1.0=cool  0=neutral  +1.0=warm", classes="hint")

        yield Label(" SHARPENING", classes="sec-head")
        with Horizontal(classes="row"):
            yield Label("CAS  (gentle)", classes="lbl")
            yield Switch(value=s.prev_cas, id="prev_cas")
            yield Label("contrast-adaptive sharpening", classes="hint")
        with Horizontal(classes="row"):
            yield Label("CAS strength", classes="lbl")
            yield Input(value=str(s.prev_cas_strength), id="prev_cas_strength", classes="narrow")
            yield Label("0.0–1.0", classes="hint")
        with Horizontal(classes="row"):
            yield Label("Unsharp mask", classes="lbl")
            yield Switch(value=s.prev_sharpen, id="prev_sharpen")
            yield Label("stronger, can look artificial", classes="hint")
        with Horizontal(classes="row"):
            yield Label("Unsharp amount", classes="lbl")
            yield Input(value=str(s.prev_sharpen_amount), id="prev_sharpen_amount", classes="narrow")
            yield Label("0.5–3.0", classes="hint")

        yield Label(" OUTPUT", classes="sec-head")
        with Horizontal(classes="row"):
            yield Label("Codec", classes="lbl")
            yield Select(
                [("H.264  (default)", "h264"), ("H.265", "h265"),
                 ("FFV1  lossless", "ffv1"), ("ProRes", "prores")],
                value=s.prev_codec,
                id="prev_codec",
            )
        with Horizontal(classes="row"):
            yield Label("Film grain", classes="lbl")
            yield Input(value=str(s.prev_grain), id="prev_grain", classes="narrow")
            yield Label("0=off  4=subtle  8=natural", classes="hint")
        with Horizontal(classes="row"):
            yield Label("Fix aspect ratio", classes="lbl")
            yield Switch(value=s.prev_fix_ar, id="prev_fix_ar")
        with Horizontal(classes="row"):
            yield Label("AR target", classes="lbl")
            yield Input(value=s.prev_ar_target, id="prev_ar_target", classes="narrow")
            yield Label("4:3 or 16:9", classes="hint")

    def _restore_section(self) -> ComposeResult:
        s = self.app.state
        yield Label(" DEINTERLACING", classes="sec-head")
        with Horizontal(classes="row"):
            yield Label("Deinterlace  (QTGMC)", classes="lbl")
            yield Switch(value=not s.skip_deinterlace, id="deinterlace")
            yield Label("always recommended for VHS", classes="hint")
        with Horizontal(classes="row"):
            yield Label("QTGMC preset", classes="lbl")
            yield Select(_qtgmc_opts(), value=s.qtgmc_preset, id="qtgmc_preset")

        yield Label(" DENOISING", classes="sec-head")
        with Horizontal(classes="row"):
            yield Label("Denoise  (KNLMeansCL GPU)", classes="lbl")
            yield Switch(value=not s.skip_denoise, id="denoise")
        with Horizontal(classes="row"):
            yield Label("Strength", classes="lbl")
            yield Input(value=str(s.knlm_h), id="knlm_h", classes="narrow")
            yield Label("0.8=light  1.2=balanced  2.0=heavy", classes="hint")
        with Horizontal(classes="row"):
            yield Label("Temporal radius  (frames)", classes="lbl")
            yield Input(value=str(s.knlm_d), id="knlm_d", classes="narrow")
        with Horizontal(classes="row"):
            yield Label("Repair dropouts", classes="lbl")
            yield Switch(value=s.repair_dropouts, id="repair_dropouts")
            yield Label("horizontal-line VHS artifacts", classes="hint")

        yield Label(" COLOR", classes="sec-head")
        with Horizontal(classes="row"):
            yield Label("Color correction  (16–235)", classes="lbl")
            yield Switch(value=not s.skip_color, id="color")
            yield Label("expand VHS levels to full range", classes="hint")
        with Horizontal(classes="row"):
            yield Label("Saturation", classes="lbl")
            yield Input(value=str(s.saturation), id="saturation", classes="narrow")
            yield Label("1.0=unchanged  1.3=boost", classes="hint")
        with Horizontal(classes="row"):
            yield Label("Warmth", classes="lbl")
            yield Input(value=str(s.warmth), id="warmth", classes="narrow")
            yield Label("-1.0=cool  0=neutral  +1.0=warm", classes="hint")
        with Horizontal(classes="row"):
            yield Label("Balance brightness", classes="lbl")
            yield Switch(value=s.balance_brightness, id="balance_brightness")

        yield Label(" SHARPENING", classes="sec-head")
        with Horizontal(classes="row"):
            yield Label("CAS  (gentle)", classes="lbl")
            yield Switch(value=s.cas, id="cas")
            yield Label("contrast-adaptive sharpening", classes="hint")
        with Horizontal(classes="row"):
            yield Label("CAS strength", classes="lbl")
            yield Input(value=str(s.cas_strength), id="cas_strength", classes="narrow")
            yield Label("0.0–1.0", classes="hint")
        with Horizontal(classes="row"):
            yield Label("Unsharp mask", classes="lbl")
            yield Switch(value=s.sharpen, id="sharpen")
            yield Label("stronger, can look artificial", classes="hint")
        with Horizontal(classes="row"):
            yield Label("Unsharp amount", classes="lbl")
            yield Input(value=str(s.sharpen_amount), id="sharpen_amount", classes="narrow")
            yield Label("0.5–3.0", classes="hint")

        yield Label(" UPSCALING  (Real-ESRGAN)", classes="sec-head")
        with Horizontal(classes="row"):
            yield Label("Scale", classes="lbl")
            yield Select(
                [("1×  skip upscaling", 1), ("2×  recommended", 2), ("4×  slower / larger", 4)],
                value=s.realesrgan_scale,
                id="realesrgan_scale",
            )
        with Horizontal(classes="row"):
            yield Label("Model", classes="lbl")
            yield Select(
                [("realesr-general-x4v3  ← VHS", "realesr-general-x4v3"),
                 ("RealESRGAN_x4plus", "RealESRGAN_x4plus")],
                value=s.realesrgan_model,
                id="realesrgan_model",
            )
        with Horizontal(classes="row"):
            yield Label("Tile size  (VRAM limit)", classes="lbl")
            yield Input(value=str(s.realesrgan_tile), id="realesrgan_tile", classes="narrow")
            yield Label("256 safe for 8 GB VRAM", classes="hint")

        yield Label(" OUTPUT FORMAT", classes="sec-head")
        with Horizontal(classes="row"):
            yield Label("Codec", classes="lbl")
            yield Select(
                [("FFV1  lossless archival", "ffv1"), ("ProRes  post-production", "prores"),
                 ("H.264  streaming", "h264"), ("H.265  streaming / smaller", "h265")],
                value=s.output_codec,
                id="output_codec",
            )
        with Horizontal(classes="row"):
            yield Label("Film grain", classes="lbl")
            yield Input(value=str(s.grain_strength), id="grain_strength", classes="narrow")
            yield Label("0=off  4=natural  8=heavy", classes="hint")
        with Horizontal(classes="row"):
            yield Label("Keep work files", classes="lbl")
            yield Switch(value=s.keep_intermediates, id="keep_intermediates")
            yield Label("saves stages in .vhs_work_*/", classes="hint")
        with Horizontal(classes="row"):
            yield Label("Side-by-side compare", classes="lbl")
            yield Switch(value=s.compare, id="compare")

        yield Label(" AUDIO", classes="sec-head")
        with Horizontal(classes="row"):
            yield Label("Audio cleanup", classes="lbl")
            yield Switch(value=s.audio_cleanup, id="audio_cleanup")
            yield Label("spectral noise reduction", classes="hint")
        with Horizontal(classes="row"):
            yield Label("Cleanup strength  (dB)", classes="lbl")
            yield Input(value=str(s.audio_cleanup_db), id="audio_cleanup_db", classes="narrow")
            yield Label("5=gentle  10=moderate  20=strong", classes="hint")

        yield Label(" OTHER", classes="sec-head")
        with Horizontal(classes="row"):
            yield Label("Stabilize", classes="lbl")
            yield Switch(value=s.stabilize, id="stabilize")
            yield Label("requires libvidstab in ffmpeg", classes="hint")
        with Horizontal(classes="row"):
            yield Label("Fix aspect ratio", classes="lbl")
            yield Switch(value=s.fix_ar, id="fix_ar")
        with Horizontal(classes="row"):
            yield Label("AR target", classes="lbl")
            yield Input(value=s.ar_target, id="ar_target", classes="narrow")
            yield Label("4:3 or 16:9", classes="hint")

    # ── Events ────────────────────────────────────────────────────────────────

    @on(Button.Pressed, "#run-btn")
    def on_run_btn(self) -> None:
        self._save_state()
        self.app.push_screen(RunScreen())

    @on(Button.Pressed, "#back-btn")
    def on_back_btn(self) -> None:
        self.app.pop_screen()

    def action_run(self) -> None:
        self._save_state()
        self.app.push_screen(RunScreen())

    def _save_state(self) -> None:
        s = self.app.state

        def sw(id: str, default: bool) -> bool:
            try:
                return self.query_one(f"#{id}", Switch).value
            except Exception:
                return default

        def inp(id: str, default: str = "") -> str:
            try:
                return self.query_one(f"#{id}", Input).value
            except Exception:
                return default

        def sel(id: str, default: object) -> object:
            try:
                v = self.query_one(f"#{id}", Select).value
                return v if v is not Select.BLANK else default
            except Exception:
                return default

        out = inp("output-path")
        if out:
            s.output_path = Path(out)

        if s.action == "trim":
            s.trim_threshold    = _safe_float(inp("trim_threshold"),    s.trim_threshold)
            s.trim_min_duration = _safe_float(inp("trim_min_duration"), s.trim_min_duration)

        elif s.action == "preview":
            s.prev_qtgmc         = sw("prev_qtgmc",        s.prev_qtgmc)
            s.prev_qtgmc_preset  = sel("prev_qtgmc_preset", s.prev_qtgmc_preset)  # type: ignore[assignment]
            s.prev_knlm          = sw("prev_knlm",          s.prev_knlm)
            s.prev_knlm_h        = _safe_float(inp("prev_knlm_h"),        s.prev_knlm_h)
            s.prev_deinterlace   = sw("prev_deinterlace",   s.prev_deinterlace)
            s.prev_levels        = sw("prev_levels",        s.prev_levels)
            s.prev_denoise       = sw("prev_denoise",       s.prev_denoise)
            s.prev_denoise_str   = _safe_float(inp("prev_denoise_str"),   s.prev_denoise_str)
            s.prev_repair        = sw("prev_repair",        s.prev_repair)
            s.prev_deflicker     = sw("prev_deflicker",     s.prev_deflicker)
            s.prev_balance       = sw("prev_balance",       s.prev_balance)
            s.prev_saturation    = _safe_float(inp("prev_saturation"),    s.prev_saturation)
            s.prev_warmth        = _safe_float(inp("prev_warmth"),        s.prev_warmth)
            s.prev_cas           = sw("prev_cas",           s.prev_cas)
            s.prev_cas_strength  = _safe_float(inp("prev_cas_strength"),  s.prev_cas_strength)
            s.prev_sharpen       = sw("prev_sharpen",       s.prev_sharpen)
            s.prev_sharpen_amount = _safe_float(inp("prev_sharpen_amount"), s.prev_sharpen_amount)
            s.prev_codec         = sel("prev_codec",        s.prev_codec)  # type: ignore[assignment]
            s.prev_grain         = _safe_int(inp("prev_grain"),           s.prev_grain)
            s.prev_fix_ar        = sw("prev_fix_ar",        s.prev_fix_ar)
            s.prev_ar_target     = inp("prev_ar_target") or s.prev_ar_target
            s.prev_test_sample   = sw("test_sample",        s.prev_test_sample)
            s.prev_test_duration = _safe_int(inp("test_duration"),        s.prev_test_duration)
            s.prev_test_start    = inp("test_start") or s.prev_test_start

        elif s.action == "restore":
            s.skip_deinterlace   = not sw("deinterlace",        not s.skip_deinterlace)
            s.qtgmc_preset       = sel("qtgmc_preset",          s.qtgmc_preset)  # type: ignore[assignment]
            s.skip_denoise       = not sw("denoise",            not s.skip_denoise)
            s.knlm_h             = _safe_float(inp("knlm_h"),   s.knlm_h)
            s.knlm_d             = _safe_int(inp("knlm_d"),     s.knlm_d)
            s.repair_dropouts    = sw("repair_dropouts",        s.repair_dropouts)
            s.skip_color         = not sw("color",              not s.skip_color)
            s.saturation         = _safe_float(inp("saturation"), s.saturation)
            s.warmth             = _safe_float(inp("warmth"),   s.warmth)
            s.balance_brightness = sw("balance_brightness",     s.balance_brightness)
            s.cas                = sw("cas",                    s.cas)
            s.cas_strength       = _safe_float(inp("cas_strength"), s.cas_strength)
            s.sharpen            = sw("sharpen",                s.sharpen)
            s.sharpen_amount     = _safe_float(inp("sharpen_amount"), s.sharpen_amount)
            s.realesrgan_scale   = _safe_int(sel("realesrgan_scale", s.realesrgan_scale), s.realesrgan_scale)
            s.realesrgan_model   = sel("realesrgan_model",      s.realesrgan_model)  # type: ignore[assignment]
            s.realesrgan_tile    = _safe_int(inp("realesrgan_tile"), s.realesrgan_tile)
            s.output_codec       = sel("output_codec",          s.output_codec)  # type: ignore[assignment]
            s.grain_strength     = _safe_int(inp("grain_strength"), s.grain_strength)
            s.keep_intermediates = sw("keep_intermediates",     s.keep_intermediates)
            s.compare            = sw("compare",                s.compare)
            s.audio_cleanup      = sw("audio_cleanup",          s.audio_cleanup)
            s.audio_cleanup_db   = _safe_float(inp("audio_cleanup_db"), s.audio_cleanup_db)
            s.stabilize          = sw("stabilize",              s.stabilize)
            s.fix_ar             = sw("fix_ar",                 s.fix_ar)
            s.ar_target          = inp("ar_target") or s.ar_target
            s.run_test_clip      = sw("run_test_clip",          s.run_test_clip)
            s.test_sample        = sw("test_sample",            s.test_sample)
            s.test_duration      = _safe_int(inp("test_duration"), s.test_duration)
            s.test_start         = inp("test_start") or s.test_start


# ── Run screen ────────────────────────────────────────────────────────────────

class RunScreen(Screen):
    BINDINGS = [
        Binding("escape", "pop_screen", "Back"),
        Binding("q",      "app.quit",   "Quit"),
    ]

    _proc: asyncio.subprocess.Process | None = None

    def compose(self) -> ComposeResult:
        s = self.app.state
        inp_name = s.input_path.name if s.input_path else "?"
        out_name = (s.output_path or s.auto_output()).name

        yield Header(show_clock=False)
        with Vertical():
            yield Static(
                f"\n  [bold]{inp_name}[/]  →  [cyan]{out_name}[/]"
                f"  ·  [dim]{s.action}[/]\n",
                id="run-header",
            )
            yield Static("  Starting…", id="run-status")
            yield RichLog(highlight=False, markup=True, id="log", wrap=True)
        with Horizontal(id="run-footer"):
            yield Button("← Back", id="back-btn")
        yield Footer()

    def on_mount(self) -> None:
        self._start()

    @work(exclusive=True)
    async def _start(self) -> None:
        log    = self.query_one("#log", RichLog)
        status = self.query_one("#run-status", Static)
        cmd    = self.app.state.build_cmd()

        status.update(f"  [dim]Running[/] {' '.join(cmd[2:4])}…")
        log.write(f"[dim]$ {' '.join(cmd)}\n[/]")

        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "FORCE_COLOR": "1"},
            )
            assert self._proc.stdout is not None
            async for raw in self._proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                log.write(Text.from_ansi(line))
            await self._proc.wait()
            rc = self._proc.returncode
            if rc == 0:
                status.update("  [green bold]✓  Done[/]")
            else:
                status.update(f"  [red bold]✗  Failed  (exit {rc})[/]")
        except Exception as exc:
            status.update(f"  [red bold]✗  Error:[/]  {exc}")
            log.write(f"[red]{exc}[/]")

    @on(Button.Pressed, "#back-btn")
    def on_back(self) -> None:
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
        self.app.pop_screen()


# ── App ───────────────────────────────────────────────────────────────────────

def _qtgmc_opts() -> list[tuple[str, str]]:
    return [
        ("Draft  fastest", "Draft"),
        ("Fast", "Fast"),
        ("Medium", "Medium"),
        ("Slow", "Slow"),
        ("Slower  recommended", "Slower"),
        ("Placebo  best quality", "Placebo"),
    ]


class VHSTui(App[None]):
    TITLE = "VHS Restore"
    DARK  = True

    CSS = """
    Screen {
        background: #0d1117;
    }
    Header {
        background: #161b22;
        color: #58a6ff;
        text-style: bold;
    }
    Footer {
        background: #161b22;
    }
    .sec-head {
        background: #21262d;
        color: #79c0ff;
        text-style: bold;
        padding: 0 1;
        margin-top: 1;
        width: 100%;
        height: 1;
    }
    .row {
        height: 3;
        align: left middle;
        padding: 0 2;
    }
    .lbl {
        width: 32;
        color: #c9d1d9;
    }
    .hint {
        color: #484f58;
        margin-left: 1;
    }
    .narrow {
        width: 10;
    }
    .wide-input {
        width: 60;
    }
    Input {
        width: 32;
        background: #161b22;
        border: tall #30363d;
        color: #c9d1d9;
    }
    Input:focus {
        border: tall #388bfd;
    }
    Select {
        width: 40;
        background: #161b22;
        border: tall #30363d;
    }
    Switch {
        background: transparent;
    }
    #action-layout {
        height: 1fr;
    }
    #action-list-pane {
        width: 36;
        border-right: solid #30363d;
    }
    #action-heading {
        background: #21262d;
        color: #79c0ff;
        text-style: bold;
        padding: 0 1;
        width: 100%;
        height: 1;
    }
    #action-list {
        height: 1fr;
        background: #0d1117;
    }
    #action-desc-pane {
        padding: 0 2;
        background: #0d1117;
    }
    #action-desc {
        color: #c9d1d9;
    }
    #btn-bar {
        height: 5;
        align: left middle;
        padding: 0 2;
        background: #161b22;
        border-top: solid #30363d;
    }
    Button {
        margin: 0 1;
    }
    Button.success {
        background: #238636;
        color: #ffffff;
        border: none;
    }
    Button.success:hover {
        background: #2ea043;
    }
    #path-label {
        background: #21262d;
        color: #79c0ff;
        text-style: bold;
        padding: 0 1;
        height: 1;
    }
    #file-list {
        height: 1fr;
        background: #0d1117;
    }
    #file-info {
        height: 2;
        background: #161b22;
        border-top: solid #30363d;
        padding: 0 1;
        color: #c9d1d9;
    }
    #settings-scroll {
        height: 1fr;
        background: #0d1117;
    }
    #run-header {
        background: #161b22;
        border-bottom: solid #30363d;
        color: #c9d1d9;
    }
    #run-status {
        height: 2;
        padding: 0 2;
        color: #c9d1d9;
    }
    #log {
        height: 1fr;
        border: solid #30363d;
        margin: 0 1 1 1;
        background: #010409;
        color: #c9d1d9;
    }
    #run-footer {
        height: 4;
        align: left middle;
        padding: 0 1;
        background: #161b22;
        border-top: solid #30363d;
    }
    """

    state: State

    def on_mount(self) -> None:
        self.state = State()
        self.push_screen(FilePickerScreen())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    VHSTui().run()
