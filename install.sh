#!/usr/bin/env bash
# ============================================================
# MedServer — Auto-Install Script (Linux / macOS)
#
# Installs everything needed to run MedGemma locally:
#   1. Checks Python 3.10+
#   2. Creates virtualenv
#   3. Installs pip dependencies (FastAPI, vLLM, PyTorch, etc.)
#   4. Configures HuggingFace token
#   5. Optionally pre-downloads the model
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
# ============================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

VENV_DIR=".venv"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10

banner() {
    echo ""
    echo -e "${CYAN}${BOLD}"
    echo "  ╔══════════════════════════════════════════════╗"
    echo "  ║        MedServer — Auto Installer           ║"
    echo "  ║     MedGemma Clinical AI Server v1.0.0      ║"
    echo "  ╚══════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
}

log()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn()  { echo -e "  ${YELLOW}⚠${NC} $1"; }
err()   { echo -e "  ${RED}✗${NC} $1"; }
info()  { echo -e "  ${CYAN}ℹ${NC} $1"; }

# ── Step 1: Check Python ──────────────────────────────────
check_python() {
    echo -e "${BOLD}[1/5] Checking Python...${NC}"

    if command -v python3 &>/dev/null; then
        PYTHON_CMD=python3
    elif command -v python &>/dev/null; then
        PYTHON_CMD=python
    else
        err "Python not found! Please install Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+."
        err "  Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
        err "  RHEL/CentOS:   sudo yum install python3"
        err "  macOS:         brew install python@3.12"
        exit 1
    fi

    PY_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
    PY_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

    if [ "$PY_MAJOR" -lt "$MIN_PYTHON_MAJOR" ] || { [ "$PY_MAJOR" -eq "$MIN_PYTHON_MAJOR" ] && [ "$PY_MINOR" -lt "$MIN_PYTHON_MINOR" ]; }; then
        err "Python ${PY_VERSION} found, but ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ required."
        exit 1
    fi

    log "Python ${PY_VERSION} found (${PYTHON_CMD})"
}

# ── Step 2: Create virtualenv ─────────────────────────────
create_venv() {
    echo -e "${BOLD}[2/5] Setting up virtual environment...${NC}"

    if [ -d "$VENV_DIR" ]; then
        warn "Virtual environment already exists. Reusing."
    else
        $PYTHON_CMD -m venv "$VENV_DIR"
        log "Created virtual environment at ${VENV_DIR}/"
    fi

    # Activate
    source "${VENV_DIR}/bin/activate"
    log "Activated virtual environment"

    # Upgrade pip
    pip install --upgrade pip setuptools wheel -q
    log "Upgraded pip, setuptools, wheel"
}

# ── Step 3: Install dependencies ──────────────────────────
install_deps() {
    echo -e "${BOLD}[3/5] Installing dependencies...${NC}"
    info "This may take 5-15 minutes (downloading vLLM, PyTorch, CUDA libs)..."
    echo ""

    pip install -e "." 2>&1 | while IFS= read -r line; do
        # Show progress on key packages
        if echo "$line" | grep -qE "Successfully installed|Downloading|Installing"; then
            echo -e "     ${line}"
        fi
    done

    log "All dependencies installed"
}

# ── Step 4: HuggingFace token ─────────────────────────────
configure_hf_token() {
    echo -e "${BOLD}[4/5] Configuring HuggingFace access...${NC}"

    if [ -n "$HF_TOKEN" ]; then
        log "HF_TOKEN found in environment"
        return
    fi

    if command -v huggingface-cli &>/dev/null; then
        HF_WHOAMI=$(huggingface-cli whoami 2>/dev/null || echo "")
        if [ -n "$HF_WHOAMI" ] && [ "$HF_WHOAMI" != "Not logged in" ]; then
            log "Already logged in to HuggingFace as: ${HF_WHOAMI}"
            return
        fi
    fi

    echo ""
    info "MedGemma models are gated — you need a HuggingFace token."
    info "Get one at: https://huggingface.co/settings/tokens"
    info ""
    info "You also need to accept the model license at:"
    info "  https://huggingface.co/google/medgemma-1.5-4b-it"
    info ""
    read -p "  Enter HuggingFace token (or press Enter to skip): " HF_INPUT

    if [ -n "$HF_INPUT" ]; then
        export HF_TOKEN="$HF_INPUT"
        echo "export HF_TOKEN=\"${HF_INPUT}\"" >> "${VENV_DIR}/bin/activate"
        log "Token saved to virtualenv activation script"

        # Also run huggingface-cli login
        if command -v huggingface-cli &>/dev/null; then
            echo "$HF_INPUT" | huggingface-cli login --token "$HF_INPUT" 2>/dev/null || true
            log "Logged in to HuggingFace CLI"
        fi
    else
        warn "Skipped. Set HF_TOKEN before running medserver, or pass --hf-token."
    fi
}

# ── Step 5: Optional model pre-download ───────────────────
predownload_model() {
    echo -e "${BOLD}[5/5] Model pre-download (optional)...${NC}"
    echo ""
    info "Available models:"
    info "  [1] MedGemma 1.5 4B   (multimodal, ~8 GB download, ≥16 GB VRAM)"
    info "  [2] MedGemma 27B      (multimodal, ~54 GB download, ≥32 GB VRAM)"
    info "  [3] MedGemma 27B Text (text-only,  ~54 GB download, ≥32 GB VRAM)"
    info "  [s] Skip — download at first run"
    echo ""
    read -p "  Pre-download model [1/2/3/s]: " MODEL_CHOICE

    case "$MODEL_CHOICE" in
        1)
            MODEL_ID="google/medgemma-1.5-4b-it"
            MODEL_KEY="4"
            ;;
        2)
            MODEL_ID="google/medgemma-27b-it"
            MODEL_KEY="27"
            ;;
        3)
            MODEL_ID="google/medgemma-27b-text-it"
            MODEL_KEY="27t"
            ;;
        *)
            info "Skipping pre-download. Model will download on first run."
            return
            ;;
    esac

    info "Downloading ${MODEL_ID}... (this may take a while)"
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "$MODEL_ID" || {
            warn "Download failed. You can retry later or it will download when you run medserver."
        }
    else
        warn "huggingface-cli not found. Model will download on first run."
    fi
}

# ── Finish ────────────────────────────────────────────────
finish() {
    echo ""
    echo -e "${GREEN}${BOLD}  ════════════════════════════════════════════${NC}"
    echo -e "${GREEN}${BOLD}   ✅ Installation Complete!${NC}"
    echo -e "${GREEN}${BOLD}  ════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${BOLD}Quick Start:${NC}"
    echo ""
    echo -e "    ${CYAN}source ${VENV_DIR}/bin/activate${NC}"
    echo -e "    ${CYAN}medserver -m 4 -p 8000${NC}"
    echo ""
    echo -e "  ${BOLD}Available Commands:${NC}"
    echo ""
    echo -e "    ${CYAN}medserver -m 4${NC}       # MedGemma 1.5 4B (multimodal)"
    echo -e "    ${CYAN}medserver -m 27${NC}      # MedGemma 27B (multimodal)"
    echo -e "    ${CYAN}medserver -m 27t${NC}     # MedGemma 27B (text-only)"
    echo -e "    ${CYAN}medserver -m 4 -q${NC}    # 4-bit quantized (lower VRAM)"
    echo -e "    ${CYAN}medserver --help${NC}     # All options"
    echo ""
    echo -e "  ${BOLD}Access:${NC}"
    echo ""
    LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "your-ip")
    echo -e "    Local:  ${CYAN}http://localhost:8000${NC}"
    echo -e "    WiFi:   ${CYAN}http://${LOCAL_IP}:8000${NC}"
    echo ""
}

# ── Main ──────────────────────────────────────────────────
banner
check_python
create_venv
install_deps
configure_hf_token
predownload_model
finish
