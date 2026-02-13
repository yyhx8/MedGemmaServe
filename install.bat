@echo off
REM ============================================================
REM MedServer — Auto-Install Script (Windows)
REM
REM Installs everything needed to run MedGemma locally:
REM   1. Checks Python 3.10+
REM   2. Creates virtualenv
REM   3. Installs pip dependencies (FastAPI, Transformers, PyTorch, etc.)
REM   4. Configures HuggingFace token
REM   5. Optionally pre-downloads the model
REM
REM Usage:
REM   install.bat
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo   ======================================================
echo        MedServer — Auto Installer (Windows)
echo        MedGemma Clinical AI Server v1.0.0
echo   ======================================================
echo.

set VENV_DIR=.venv
set PYTHON_CMD=

REM ── Step 1: Check Python ─────────────────────────────────
echo [1/5] Checking Python...

where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set PYTHON_CMD=python
) else (
    where python3 >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set PYTHON_CMD=python3
    ) else (
        echo   [X] Python not found!
        echo       Please install Python 3.10+ from https://python.org
        echo       Make sure to check "Add Python to PATH" during installation.
        pause
        exit /b 1
    )
)

for /f "tokens=*" %%i in ('%PYTHON_CMD% -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PY_VERSION=%%i
echo   [OK] Python %PY_VERSION% found

REM ── Step 2: Create virtualenv ────────────────────────────
echo [2/5] Setting up virtual environment...

if exist "%VENV_DIR%" (
    echo   [!] Virtual environment already exists. Reusing.
) else (
    REM Ensure virtualenv is installed
    %PYTHON_CMD% -m virtualenv --version >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo   [*] Installing virtualenv...
        %PYTHON_CMD% -m pip install virtualenv -q
    )

    %PYTHON_CMD% -m virtualenv %VENV_DIR%
    echo   [OK] Created virtual environment (using virtualenv)
)

call %VENV_DIR%\Scripts\activate.bat
echo   [OK] Activated virtual environment

pip install --upgrade pip setuptools wheel -q
echo   [OK] Upgraded pip

REM ── Step 3: Install dependencies ─────────────────────────
echo [3/5] Installing dependencies...
echo       This may take 5-15 minutes...
echo.

pip install -e "."

if %ERRORLEVEL% neq 0 (
    echo   [X] Installation failed. Check the error above.
    pause
    exit /b 1
)

echo   [OK] All dependencies installed

REM ── Step 4: HuggingFace token ────────────────────────────
echo [4/5] Configuring HuggingFace access...

if defined HF_TOKEN (
    echo   [OK] HF_TOKEN found in environment
    goto :skip_hf_token
)

echo.
echo   MedGemma models are gated — you need a HuggingFace token.
echo   Get one at: https://huggingface.co/settings/tokens
echo.
echo   You also need to accept the model license at:
echo   https://huggingface.co/google/medgemma-1.5-4b-it
echo.

set /p HF_INPUT="  Enter HuggingFace token (or press Enter to skip): "

if not "!HF_INPUT!"=="" (
    setx HF_TOKEN "!HF_INPUT!" >nul 2>&1
    set HF_TOKEN=!HF_INPUT!
    echo   [OK] Token saved to user environment variables
) else (
    echo   [!] Skipped. Set HF_TOKEN before running medserver.
)

:skip_hf_token

REM ── Step 5: Optional model pre-download ──────────────────
echo [5/5] Model pre-download (optional)...
echo.
echo   Available models:
echo     [1] MedGemma 1.5 4B   (multimodal, ~8 GB, needs 16+ GB VRAM)
echo     [2] MedGemma 27B      (multimodal, ~54 GB, needs 32+ GB VRAM)
echo     [3] MedGemma 27B Text (text-only,  ~54 GB, needs 32+ GB VRAM)
echo     [s] Skip — download at first run
echo.

set /p MODEL_CHOICE="  Pre-download model [1/2/3/s]: "

if "%MODEL_CHOICE%"=="1" (
    set MODEL_ID=google/medgemma-1.5-4b-it
) else if "%MODEL_CHOICE%"=="2" (
    set MODEL_ID=google/medgemma-27b-it
) else if "%MODEL_CHOICE%"=="3" (
    set MODEL_ID=google/medgemma-27b-text-it
) else (
    echo   Skipping pre-download.
    goto :finish
)

echo   Downloading %MODEL_ID%...
huggingface-cli download %MODEL_ID%
if %ERRORLEVEL% neq 0 (
    echo   [!] Download failed. It will download on first run.
)

:finish

echo.
echo   ======================================================
echo    Installation Complete!
echo   ======================================================
echo.
echo   Quick Start:
echo.
echo     %VENV_DIR%\Scripts\activate
echo     medserver -m 4 -p 8000
echo.
echo   Commands:
echo     medserver -m 4       MedGemma 1.5 4B (multimodal)
echo     medserver -m 27      MedGemma 27B (multimodal)
echo     medserver -m 27t     MedGemma 27B (text-only)
echo     medserver -m 4 -q    4-bit quantized (lower VRAM)
echo     medserver --help     All options
echo.

pause
