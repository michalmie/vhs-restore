#!/usr/bin/env bash
# Verify VHS pipeline dependencies are correctly installed
# Run after setup_ubuntu.sh: bash pipeline/verify.sh

VENV="$HOME/vhs-env"
VS_PLUGIN_DIR="$HOME/.local/lib/vapoursynth"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
warn() { echo -e "  ${YELLOW}!${NC} $*"; }
fail() { echo -e "  ${RED}✗${NC} $*"; FAILURES=$((FAILURES+1)); }

FAILURES=0

echo "VHS Pipeline — Dependency Verification"
echo "════════════════════════════════════════"

# Activate venv automatically — bail out early with a clear message if it doesn't exist
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -d "$VENV" ]; then
        source "$VENV/bin/activate"
        echo "  (auto-activated venv: $VENV)"
    else
        echo -e "${RED}ERROR: venv not found at $VENV${NC}"
        echo "  Run setup first:  bash pipeline/setup_ubuntu.sh"
        echo "  Then retry:       bash pipeline/verify.sh"
        exit 1
    fi
fi


# ── GPU ───────────────────────────────────────────────────────────────────────
echo ""
echo "GPU"
if command -v nvidia-smi &>/dev/null; then
    GPU=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)
    ok "nvidia-smi: $GPU"
else
    fail "nvidia-smi not found — NVIDIA driver not installed"
fi

if command -v nvcc &>/dev/null; then
    ok "nvcc: $(nvcc --version | grep release | awk '{print $6}' | tr -d ',')"
else
    warn "nvcc not found — CUDA toolkit not installed (needed for training; inference still works)"
fi

# OpenCL (needed by KNLMeansCL)
if python3 -c "import pyopencl; print(pyopencl.get_platforms())" 2>/dev/null | grep -q NVIDIA; then
    ok "OpenCL: NVIDIA platform visible"
else
    warn "pyopencl not installed or NVIDIA OpenCL platform not found"
    warn "  Install: pip install pyopencl  (for verification only, runtime uses system OpenCL)"
fi


# ── ffmpeg ────────────────────────────────────────────────────────────────────
echo ""
echo "ffmpeg"
if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg: $(ffmpeg -version 2>&1 | head -1)"
else
    fail "ffmpeg not found"
fi

if ffmpeg -filters 2>/dev/null | grep -q vmaf; then
    ok "libvmaf: present (Gate 3 VMAF check will work)"
else
    warn "libvmaf: NOT present in this ffmpeg build"
    warn "  Gate 3 (VMAF) will be skipped — see setup_ubuntu.sh step 3"
fi

if command -v ffprobe &>/dev/null; then
    ok "ffprobe: present"
else
    fail "ffprobe not found"
fi

if command -v vspipe &>/dev/null; then
    ok "vspipe: $(vspipe --version 2>&1 | head -1)"
else
    fail "vspipe not found — VapourSynth CLI not in PATH"
fi


# ── Python ────────────────────────────────────────────────────────────────────
echo ""
echo "Python"
PY_VER=$(python3 --version)
ok "Python: $PY_VER"

if [ -n "${VIRTUAL_ENV:-}" ]; then
    ok "venv: active ($VIRTUAL_ENV)"
else
    warn "venv not active — activate with: source $VENV/bin/activate"
fi


# ── VapourSynth ───────────────────────────────────────────────────────────────
echo ""
echo "VapourSynth"
python3 - <<'EOF'
import sys
try:
    import vapoursynth as vs
    print(f"  \033[0;32m✓\033[0m vapoursynth: {vs.__version__}")
except ImportError as e:
    print(f"  \033[0;31m✗\033[0m vapoursynth import failed: {e}")
    sys.exit(1)

core = vs.core
any_missing = False

plugins = {
    "ffms2":      ("ffms2", "Video source (core.ffms2.Source)"),
    "knlmeanscl": ("knlm",  "GPU denoising (core.knlm.KNLMeansCL)"),
    "mvtools":    ("mv",    "Motion compensation (needed by QTGMC)"),
}

for plugin_id, (attr, desc) in plugins.items():
    if hasattr(core, attr):
        print(f"  \033[0;32m✓\033[0m {plugin_id}: loaded ({desc})")
    else:
        print(f"  \033[0;31m✗\033[0m {plugin_id}: NOT loaded — {attr}.so missing from plugin dir")
        any_missing = True

if any_missing:
    sys.exit(1)
EOF
[ $? -ne 0 ] && FAILURES=$((FAILURES+1))

# havsfunc / QTGMC
python3 -c "import havsfunc; print('  \033[0;32m✓\033[0m havsfunc (QTGMC): imported')" 2>/dev/null || \
    echo -e "  ${RED}✗${NC} havsfunc: import failed — check site-packages"

# List installed plugins
echo ""
echo "  Plugin directory: $VS_PLUGIN_DIR"
if [ -d "$VS_PLUGIN_DIR" ]; then
    SO_COUNT=$(find "$VS_PLUGIN_DIR" -name "*.so" | wc -l)
    echo "  Installed .so files: $SO_COUNT"
    find "$VS_PLUGIN_DIR" -name "*.so" | while read -r f; do
        echo "    $(basename "$f")"
    done
else
    warn "Plugin directory does not exist: $VS_PLUGIN_DIR"
fi


# ── PyTorch + CUDA ────────────────────────────────────────────────────────────
echo ""
echo "PyTorch"
python3 - <<'EOF'
try:
    import torch
    print(f"  \033[0;32m✓\033[0m torch: {torch.__version__}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"  \033[0;32m✓\033[0m CUDA: {name} ({mem} GB VRAM)")
    else:
        print(f"  \033[1;33m!\033[0m CUDA not available — GPU acceleration disabled (inference will use CPU)")
except ImportError:
    print("  \033[0;31m✗\033[0m torch not installed: pip install torch --index-url https://download.pytorch.org/whl/cu121")

try:
    import piq
    print(f"  \033[0;32m✓\033[0m piq (BRISQUE/NIQE): imported")
except ImportError:
    print("  \033[0;31m✗\033[0m piq not installed: pip install piq")
EOF


# ── Real-ESRGAN ───────────────────────────────────────────────────────────────
echo ""
echo "Real-ESRGAN"
REALESRGAN_DIR="${REALESRGAN_DIR:-$HOME/Real-ESRGAN}"
INFERENCE_SCRIPT="$REALESRGAN_DIR/inference_realesrgan.py"

if [ -f "$INFERENCE_SCRIPT" ]; then
    ok "inference_realesrgan.py found: $REALESRGAN_DIR"
else
    fail "Real-ESRGAN not found at $REALESRGAN_DIR"
    echo "    Clone with: git clone https://github.com/xinntao/Real-ESRGAN $REALESRGAN_DIR"
fi

for model in "realesr-general-x4v3.pth" "RealESRGAN_x4plus.pth"; do
    if [ -f "$REALESRGAN_DIR/weights/$model" ]; then
        SIZE=$(du -sh "$REALESRGAN_DIR/weights/$model" | cut -f1)
        ok "Model: $model ($SIZE)"
    else
        warn "Model not downloaded: $model"
        warn "  Download: python $REALESRGAN_DIR/scripts/download_models.py"
    fi
done


# ── Pipeline script ───────────────────────────────────────────────────────────
echo ""
echo "Pipeline"
SCRIPT="$(dirname "$0")/restore.py"
if [ -f "$SCRIPT" ]; then
    ok "restore.py found"
    python3 "$SCRIPT" --help > /dev/null 2>&1 && ok "restore.py --help: OK" || \
        warn "restore.py --help failed — check script syntax"
else
    fail "restore.py not found at $SCRIPT"
fi


# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
if [ "$FAILURES" -eq 0 ]; then
    echo -e "${GREEN}All checks passed.${NC}"
else
    echo -e "${RED}$FAILURES check(s) failed — review output above.${NC}"
fi
echo ""
echo "Quick test (30-second clip):"
echo "  python pipeline/restore.py input.mkv output_test.mkv --test"
