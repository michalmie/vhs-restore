#!/usr/bin/env bash
# VHS Pipeline Setup — Ubuntu 22.04 LTS
# Supports: NVIDIA GPU, CUDA 11.x / 12.x, Python 3.12
# Usage:    bash setup_ubuntu.sh           # install / resume
#           bash setup_ubuntu.sh --clean   # wipe all installed components and re-install

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
VENV="$HOME/vhs-env"
REALESRGAN_DIR="$HOME/Real-ESRGAN"
VS_PLUGIN_DIR="$HOME/.local/lib/vapoursynth"
VS_SRC="/tmp/vs-build"
VS_LIB="/usr/local/lib/x86_64-linux-gnu/libvapoursynth.so.4"
VS_PKG="/usr/local/lib/x86_64-linux-gnu/pkgconfig"
REQUIRED_PY="3.12"   # VapourSynth R74+ requires Python ≥ 3.12

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓ $*${NC}"; }
warn() { echo -e "${YELLOW}  ! $*${NC}"; }
fail() { echo -e "${RED}  ✗ $*${NC}"; exit 1; }
step() { echo -e "\n${GREEN}[$1/8] $2${NC}"; }

# ── --clean: wipe everything this script installed ────────────────────────────
if [ "${1:-}" = "--clean" ]; then
    echo -e "${YELLOW}Cleaning all installed components...${NC}"
    rm -rf "$VENV"
    rm -rf "$VS_PLUGIN_DIR"
    rm -rf "$REALESRGAN_DIR"
    rm -rf "$VS_SRC" /tmp/vs-plugin-* /tmp/havsfunc /tmp/knlmeanscl.7z
    sudo rm -f /usr/local/lib/x86_64-linux-gnu/libvapoursynth.so* \
               /usr/local/lib/x86_64-linux-gnu/pkgconfig/vapoursynth*.pc \
               /usr/local/bin/vspipe \
               /etc/ld.so.conf.d/vapoursynth.conf
    sudo rm -rf /usr/local/include/vapoursynth
    sudo ldconfig
    echo -e "${GREEN}Clean done. Re-run without --clean to install fresh.${NC}"
    exit 0
fi


# ── 1. System packages + Python 3.12 ─────────────────────────────────────────
step 1 "System packages + Python 3.12"

sudo apt update -qq
sudo apt install -y \
    build-essential cmake git pkg-config curl wget unzip p7zip-full \
    ninja-build autoconf automake libtool nasm \
    python3 python3-pip python3-venv python3-dev \
    ffmpeg \
    libzimg-dev libnuma-dev \
    ocl-icd-opencl-dev opencl-headers \
    libffms2-dev ffmsindex \
    libfftw3-dev \
    libboost-dev

# Python 3.12 — required by VapourSynth R74+; Ubuntu 22.04 ships 3.10
if ! command -v python3.12 &>/dev/null; then
    echo "  Installing Python 3.12 via deadsnakes PPA..."
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt install -y python3.12 python3.12-dev python3.12-venv
fi

PY_BIN=$(command -v python3.12)
ok "System packages installed — Python $("$PY_BIN" --version | awk '{print $2}')"


# ── 2. NVIDIA driver + CUDA ───────────────────────────────────────────────────
step 2 "NVIDIA driver + CUDA"

if ! command -v nvidia-smi &>/dev/null; then
    warn "nvidia-smi not found — installing driver 535..."
    sudo apt install -y nvidia-driver-535
    echo ""
    warn "Driver installed. Reboot now, then re-run this script."
    warn "  sudo reboot"
    exit 0
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
ok "GPU: $GPU_NAME"

# Detect CUDA version from driver (nvidia-smi) first, fall back to nvcc
if command -v nvidia-smi &>/dev/null; then
    CUDA_MAJOR=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' | head -1)
    ok "CUDA driver version: $(nvidia-smi | grep 'CUDA Version' | grep -oP 'CUDA Version: \K[0-9.]+')"
elif command -v nvcc &>/dev/null; then
    CUDA_MAJOR=$(nvcc --version | grep -oP 'release \K[0-9]+' | head -1)
    ok "CUDA toolkit: $(nvcc --version | grep release | awk '{print $6}' | tr -d ',')"
else
    warn "Neither nvidia-smi nor nvcc found — defaulting to cu118 PyTorch wheel."
    CUDA_MAJOR=11
fi

# NVIDIA OpenCL ICD — needed by KNLMeansCL
sudo apt install -y nvidia-opencl-dev 2>/dev/null && ok "NVIDIA OpenCL ICD" || \
    warn "nvidia-opencl-dev unavailable — KNLMeansCL may fall back to CPU"


# ── 3. ffmpeg with libvmaf ────────────────────────────────────────────────────
step 3 "ffmpeg with libvmaf"

# Ensure /usr/local/bin is first so our installed binaries take precedence
export PATH=/usr/local/bin:$PATH
grep -qF '/usr/local/bin' ~/.bashrc || echo 'export PATH=/usr/local/bin:$PATH' >> ~/.bashrc

_has_vmaf() { "${1:-ffmpeg}" -filters 2>/dev/null | grep -q vmaf; }

if _has_vmaf; then
    ok "ffmpeg has libvmaf: $(which ffmpeg)"
else
    echo "  System ffmpeg lacks libvmaf — installing jellyfin-ffmpeg7..."
    curl -fsSL https://repo.jellyfin.org/ubuntu/jellyfin_team.gpg.key \
        | sudo gpg --dearmor --yes -o /usr/share/keyrings/jellyfin.gpg
    echo "deb [signed-by=/usr/share/keyrings/jellyfin.gpg arch=amd64] \
https://repo.jellyfin.org/ubuntu jammy main" \
        | sudo tee /etc/apt/sources.list.d/jellyfin.list > /dev/null
    sudo apt update -qq
    sudo apt install -y jellyfin-ffmpeg7
    sudo ln -sf /usr/lib/jellyfin-ffmpeg/ffmpeg  /usr/local/bin/ffmpeg
    sudo ln -sf /usr/lib/jellyfin-ffmpeg/ffprobe /usr/local/bin/ffprobe

    if _has_vmaf /usr/local/bin/ffmpeg; then
        ok "jellyfin-ffmpeg7 with libvmaf installed"
    else
        warn "libvmaf unavailable — Gate 3 (VMAF upscale check) will be skipped. Pipeline still works fully."
    fi
fi


# ── 4. Python virtual environment ─────────────────────────────────────────────
step 4 "Python $REQUIRED_PY virtual environment"

# If existing venv was built with the wrong Python, recreate it
if [ -d "$VENV" ]; then
    EXISTING_PY=$("$VENV/bin/python" --version 2>/dev/null | awk '{print $2}' | cut -d. -f1,2 || echo "none")
    if [ "$EXISTING_PY" != "$REQUIRED_PY" ]; then
        warn "Existing venv uses Python $EXISTING_PY — recreating with Python $REQUIRED_PY..."
        rm -rf "$VENV"
    fi
fi

"$PY_BIN" -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel setuptools -q
ok "venv: $VENV  (Python $(python3 --version | awk '{print $2}'))"


# ── 5. VapourSynth core ───────────────────────────────────────────────────────
step 5 "VapourSynth (build from source)"

# meson via pip — apt ships 0.61 which is too old; VapourSynth needs ≥ 0.63
pip install meson cython -q
ok "meson $(meson --version) (pip)"

# Skip build if the library file already exists (ldconfig cache can be unreliable across re-runs)
if [ -f "$VS_LIB" ]; then
    ok "libvapoursynth already installed — skipping build"
else
    echo "  Cloning VapourSynth..."
    rm -rf "$VS_SRC"
    git clone -q --depth=1 https://github.com/vapoursynth/vapoursynth.git "$VS_SRC"

    echo "  Building (meson + ninja)..."
    cd "$VS_SRC"
    meson setup build --prefix=/usr/local --buildtype=release
    ninja -C build -j"$(nproc)"
    sudo ninja -C build install

    # Register the x86_64 multiarch lib dir so ldconfig finds libvapoursynth
    echo '/usr/local/lib/x86_64-linux-gnu' | sudo tee /etc/ld.so.conf.d/vapoursynth.conf > /dev/null
    sudo ldconfig
    cd - > /dev/null
    ok "VapourSynth built and installed"
fi

pip install vapoursynth vsrepo -q
mkdir -p "$VS_PLUGIN_DIR"

python3 -c "import vapoursynth as vs; print('  VapourSynth', vs.__version__)" || \
    fail "VapourSynth import failed — check build output above"
ok "VapourSynth Python bindings"


# ── 6. VapourSynth plugins ────────────────────────────────────────────────────
step 6 "VapourSynth plugins"

SITE_PKGS="$(python3 -c 'import site; print(site.getsitepackages()[0])')"

# havsfunc — pure Python, provides QTGMC
echo "  Installing havsfunc (QTGMC)..."
rm -rf /tmp/havsfunc
git clone -q --depth=1 https://github.com/HomeOfVapourSynthEvolution/havsfunc /tmp/havsfunc
if [ -f /tmp/havsfunc/havsfunc.py ]; then
    cp /tmp/havsfunc/havsfunc.py "$SITE_PKGS/"
    ok "havsfunc (QTGMC)"
else
    pip install havsfunc -q && ok "havsfunc (pip)" || \
        warn "havsfunc install failed — QTGMC unavailable"
fi

# ffms2 VapourSynth plugin — build tag 2.40 against system ffmpeg 4.x
# (latest ffms2 HEAD requires ffmpeg 7.x; v2.40 requires only 4.x which is what
# Ubuntu 22.04 ships; system libffms2-dev installs only the C library, not the VS plugin)
rm -f "$VS_PLUGIN_DIR/libffms2.so"  # remove stale broken symlinks from previous attempts
if find "$VS_PLUGIN_DIR" -name "libffms2.so" 2>/dev/null | grep -q .; then
    ok "ffms2 (already installed)"
else
    FFMS2_SRC="/tmp/vs-plugin-ffms2"
    FFMS2_LOG="/tmp/vs-build-ffms2.log"
    echo "  Building ffms2 v2.40 (system ffmpeg 4.x)..."
    rm -rf "$FFMS2_SRC"
    if ! git clone -q --depth=1 --branch 2.40 https://github.com/FFMS/ffms2 "$FFMS2_SRC" >> "$FFMS2_LOG" 2>&1; then
        warn "ffms2: git clone failed — check $FFMS2_LOG"
    else
        cd "$FFMS2_SRC"
        # v2.40 uses system ffmpeg 4.x headers; VS headers are in VS_PKG's parent include dir
        if ./autogen.sh >> "$FFMS2_LOG" 2>&1 \
           && PKG_CONFIG_PATH="$VS_PKG${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
              ./configure --prefix=/usr/local --with-vapoursynth >> "$FFMS2_LOG" 2>&1 \
           && make -j"$(nproc)" >> "$FFMS2_LOG" 2>&1; then
            # autotools puts the VS plugin .so in src/vapoursynth/.libs/
            BUILT_SO=$(find . -name "libffms2.so" ! -name "*.la" 2>/dev/null | head -1)
            if [ -n "$BUILT_SO" ]; then
                cp -L "$BUILT_SO" "$VS_PLUGIN_DIR/libffms2.so"
                ok "ffms2 v2.40 (built from source)"
            else
                warn "ffms2: build succeeded but libffms2.so not found — check $FFMS2_LOG"
            fi
        else
            warn "ffms2: build failed — last 20 lines of $FFMS2_LOG:"
            tail -20 "$FFMS2_LOG" | sed 's/^/    /'
        fi
        cd - > /dev/null
    fi
fi

# Helper: build a meson-based VapourSynth plugin from source
_build_meson_plugin() {
    local name="$1" url="$2" so_pattern="$3"
    local src="/tmp/vs-plugin-$name"
    local log="/tmp/vs-build-${name}.log"
    if find "$VS_PLUGIN_DIR" -name "$so_pattern" 2>/dev/null | grep -q .; then
        ok "$name (already installed)"; return 0
    fi
    echo "  Building $name from source..."
    rm -rf "$src"
    if ! git clone -q --depth=1 --recurse-submodules "$url" "$src" 2>/dev/null; then
        warn "$name: git clone failed"; return 1
    fi
    cd "$src"
    if PKG_CONFIG_PATH="$VS_PKG${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
       meson setup build --buildtype=release -Ddefault_library=shared > "$log" 2>&1 \
       && ninja -C build -j"$(nproc)" >> "$log" 2>&1; then
        # Use cp -L to dereference symlinks (meson sometimes produces versioned .so + symlink)
        find build -name "*.so" | while read -r f; do cp -L "$f" "$VS_PLUGIN_DIR/"; done
        if find "$VS_PLUGIN_DIR" -name "$so_pattern" 2>/dev/null | grep -q .; then
            ok "$name (built from source)"
        else
            ACTUAL=$(find build -name "*.so" 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')
            warn "$name: built as '${ACTUAL:-?}' but expected '$so_pattern' — check $log"
        fi
    else
        warn "$name: build failed — last 20 lines of $log:"
        tail -20 "$log" | sed 's/^/    /'
    fi
    cd - > /dev/null
}

# mvtools — motion compensation, required by QTGMC
# meson shared_module() produces mvtools.so (no lib prefix) on Linux
_build_meson_plugin mvtools "https://github.com/dubhater/vapoursynth-mvtools" "mvtools.so"

# KNLMeansCL — GPU denoising via OpenCL
# meson shared_library() produces libknlmeanscl.so (lowercased, with lib prefix) on Linux
_build_meson_plugin knlm   "https://github.com/Khanattila/KNLMeansCL"        "libknlmeanscl.so"


# ── 7. PyTorch + Real-ESRGAN ─────────────────────────────────────────────────
step 7 "PyTorch + Real-ESRGAN"

# Pick PyTorch wheel matching installed CUDA version
if [ "${CUDA_MAJOR:-0}" -ge 12 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    echo "  CUDA $CUDA_MAJOR detected → PyTorch cu121"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    echo "  CUDA $CUDA_MAJOR detected → PyTorch cu118 (CUDA 11.x)"
fi

# Skip only if installed build matches required CUDA wheel
REQUIRED_CU=$(basename "$TORCH_INDEX")  # e.g. cu121
if python3 -c "import torch; v=torch.__version__; exit(0 if '${REQUIRED_CU}' in v else 1)" 2>/dev/null; then
    ok "PyTorch already installed ($(python3 -c 'import torch; print(torch.__version__)'))"
else
    echo "  Downloading PyTorch (~2 GB — this will take several minutes)..."
    pip install torch torchvision --index-url "$TORCH_INDEX" --upgrade
fi

# Real-ESRGAN
if [ ! -d "$REALESRGAN_DIR" ]; then
    echo "  Cloning Real-ESRGAN..."
    git clone https://github.com/xinntao/Real-ESRGAN "$REALESRGAN_DIR"
fi
cd "$REALESRGAN_DIR"
echo "  Installing Real-ESRGAN requirements (excluding torch — managed separately)..."
pip install -r requirements.txt --constraint <(echo "torch==$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '0')" 2>/dev/null || true)
python3 setup.py develop -q 2>/dev/null
cd - > /dev/null

# Re-pin torch to the correct CUDA wheel — Real-ESRGAN deps may have pulled a newer incompatible build
CURRENT_CU=$(python3 -c "import torch; v=torch.__version__; print(v.split('+')[1] if '+' in v else 'none')" 2>/dev/null)
if [ "$CURRENT_CU" != "$REQUIRED_CU" ]; then
    warn "torch $CURRENT_CU was installed by Real-ESRGAN deps — reinstalling $REQUIRED_CU..."
    pip install torch torchvision --index-url "$TORCH_INDEX" --force-reinstall --no-deps
fi

# basicsr ships a broken import for torchvision >= 0.17 — patch it
DEGRADATIONS="$VENV/lib/python3.12/site-packages/basicsr/data/degradations.py"
if [ -f "$DEGRADATIONS" ] && grep -q "functional_tensor" "$DEGRADATIONS"; then
    sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$DEGRADATIONS"
    ok "basicsr degradations.py patched (functional_tensor → functional)"
fi

# Download model weights
mkdir -p "$REALESRGAN_DIR/weights"
declare -A MODELS=(
    ["realesr-general-x4v3.pth"]="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
    ["RealESRGAN_x4plus.pth"]="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)
for fname in "${!MODELS[@]}"; do
    target="$REALESRGAN_DIR/weights/$fname"
    if [ ! -f "$target" ]; then
        echo "  Downloading model: $fname..."
        wget --show-progress -q "${MODELS[$fname]}" -O "$target" && ok "Model: $fname" || warn "Model download failed: $fname"
    else
        ok "Model: $fname (already present)"
    fi
done

ok "Real-ESRGAN → $REALESRGAN_DIR"


# ── 8. Pipeline Python dependencies ───────────────────────────────────────────
step 8 "Pipeline Python packages"

pip install piq rich "textual>=0.60.0" -q
ok "piq (quality metrics) + rich + textual (TUI)"


# ── Final verification ────────────────────────────────────────────────────────
echo ""
python3 - <<'PYCHECK'
import sys
errors = []
checks = [
    ("vapoursynth", "VapourSynth"),
    ("havsfunc",    "havsfunc / QTGMC"),
    ("piq",         "piq (quality metrics)"),
]
for mod, label in checks:
    try:
        __import__(mod)
        print(f"  \033[0;32m✓\033[0m {label}")
    except ImportError as e:
        errors.append(label)
        print(f"  \033[0;31m✗\033[0m {label}: {e}")

try:
    import torch
    if torch.cuda.is_available():
        print(f"  \033[0;32m✓\033[0m PyTorch CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  \033[1;33m!\033[0m PyTorch: CUDA not available (CPU fallback only)")
except ImportError as e:
    errors.append("PyTorch")
    print(f"  \033[0;31m✗\033[0m PyTorch: {e}")

if errors:
    print(f"\n  Failed: {', '.join(errors)}")
    sys.exit(1)
PYCHECK


# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo ""
echo "  GPU:         $GPU_NAME"
echo "  Python:      $("$PY_BIN" --version)"
echo "  venv:        $VENV"
echo "  Plugins:     $VS_PLUGIN_DIR"
echo "  Real-ESRGAN: $REALESRGAN_DIR"
echo ""
echo "  Run the pipeline (no venv activation needed):"
echo "    bash vhs.sh test input.mkv preview.mp4 --test-sample"
echo ""
echo "  Verify installation:"
echo "    bash pipeline/verify.sh"
echo ""
echo "  Example — test with color + sharpness corrections:"
echo "    bash vhs.sh test input.mkv preview.mp4 --test-sample --saturation 1.3 --warmth 0.3 --cas"
