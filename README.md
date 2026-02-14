# 🧬 MedServer

**One-command MedGemma clinical AI server.** Auto-installs everything, downloads the model, and serves a premium clinical web UI over your local network.

```bash
medserver -m 4 -p 7070 -ip 192.168.1.50
```

![Clinical Dark UI](medserver/static/screenshot.png)

---

## ⚡ Key Features

- **🚀 One-command Setup**: Automated installer for Linux, macOS, and Windows.
- **🧠 Clinical Reasoning**: Native support for MedGemma "Thinking Process" traces with collapsible UI.
- **🖼️ Multimodal Mastery**: Analysis of CT, MRI, and X-ray images with an integrated clinical lightbox (zoom/pan support).
- **💎 Premium Clinical UI**: Professional dark-mode interface designed for clinical environments.
- **📂 Session Persistence**: Automatic local history management with persistent chat sessions.
- **📱 Mobile Optimized**: Responsive design with touch-friendly lightbox and pinch-to-zoom support.
- **⚙️ Dual-Engine Architecture**: Automatically selects the fastest inference engine (SGLang or Transformers).
- **📊 Live System Status**: Integrated hardware widget showing GPU info and server health.

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yyhx8/MedGemmaServe.git
cd MedGemmaServe

# Linux / macOS
chmod +x install.sh
./install.sh

# Windows
install.bat
```

The installer will:
- ✅ Check Python 3.10+
- ✅ Create a virtualenv and install all dependencies (FastAPI, Transformers, SGLang, PyTorch, CUDA)
- ✅ Configure your HuggingFace token (needed for gated models)
- ✅ Optionally pre-download the model weights

> 💡 **Tip:** You can also create a `.env` file in the root directory with `HF_TOKEN=your_token_here` to avoid passing it via CLI.

### 2. Run

```bash
source .venv/bin/activate   # Linux/macOS
# or: .venv\Scripts\activate  # Windows

medserver -m 4
```

Open `http://localhost:8000` in your browser. Done. 🎉

---

## 🎛️ CLI Reference

```
medserver [-m MODEL] [-p PORT] [-ip HOST] [-q] [--workers N] [--hf-token TOKEN]
```

| Flag | Description | Default |
|------|-------------|---------|
| `-m`, `--model` | **Required.** Model to serve: `4`, `27`, or `27t` | — |
| `-p`, `--port` | Server port | `8000` |
| `-ip`, `--host` | Bind address (WiFi IP or `0.0.0.0`) | `0.0.0.0` |
| `-q`, `--quantize` | Enable 4-bit quantization (reduces VRAM ~50%) | off |
| `-v`, `--version` | Show program's version number and exit | — |
| `--workers` | Number of server workers (uvicorn) | `1` |
| `--hf-token` | HuggingFace API token | `$HF_TOKEN` env var |
| `--max-model-len` | Max context length in tokens | `8192` |
| `--gpu-memory-utilization` | GPU memory fraction to use | `0.90` |
| `--log-level` | Logging: debug/info/warning/error | `info` |

### Examples

```bash
# Serve 4B multimodal on default port
medserver -m 4

# Serve on specific WiFi address and port
medserver -m 4 -p 7070 -ip 192.168.1.50

# Serve 27B text-only with quantization
medserver -m 27t -q

# Pass HuggingFace token inline
medserver -m 4 --hf-token hf_xxxxxxxxxxxxxxxxxxxxx
```

---

## 🚀 Dual-Engine Architecture

MedServer automatically selects the most efficient inference engine based on your hardware:

1.  **SGLang Engine** (High Performance):
    - **Trigger:** Linux + NVIDIA Ampere GPU (or newer, CC >= 8.0) + `sglang` installed.
    - **Benefits:** Up to 5x faster throughput, advanced memory management (RadixAttention), and optimized streaming.
2.  **Transformers Engine** (Universal Compatibility):
    - **Trigger:** Windows, older GPUs (e.g., T4/RTX 20-series), or if `sglang` is missing.
    - **Benefits:** Runs everywhere PyTorch runs. Uses `bitsandbytes` for 4-bit quantization.

---

## 🧬 Available Models

| Flag | Model | Type | Size | Min VRAM | Best For |
|------|-------|------|------|----------|----------|
| `-m 4` | MedGemma 1.5 4B | Multimodal | ~8 GB | 16 GB | CXR, CT, MRI, general clinical Q&A |
| `-m 27` | MedGemma 27B | Multimodal | ~54 GB | 32 GB | Complex imaging + EHR analysis |
| `-m 27t` | MedGemma 27B Text | Text-only | ~54 GB | 32 GB | Clinical reasoning, summarization |

### HuggingFace Model IDs
- `google/medgemma-1.5-4b-it`
- `google/medgemma-27b-it`
- `google/medgemma-27b-text-it`

---

## 🖥️ Hardware Requirements

| GPU | VRAM | Models Supported |
|-----|------|------------------|
| RTX 4090 | 24 GB | 4B, 4B quantized |
| RTX A5000 | 24 GB | 4B, 4B quantized |
| A100 40GB | 40 GB | 4B, 27B (quantized), 27B text |
| A100 80GB | 80 GB | All models |
| H100 | 80 GB | All models |

> 💡 **Tip:** Use `-q` for 4-bit quantization to run larger models on less VRAM.

---

## 🌐 Network Access (WiFi Serving)

MedServer binds to `0.0.0.0` by default, making it accessible to any device on your local network:

1. Find your machine's IP: `hostname -I` (Linux) or `ipconfig` (Windows)
2. Run: `medserver -m 4 -p 8000`
3. On other devices, open: `http://<your-ip>:8000`

To bind to a specific interface: `medserver -m 4 -ip 192.168.1.50`

---

## 📁 Project Structure

```
MedGemmaServe/
├── setup.py                  # Package config (pip install -e .)
├── install.sh                # Linux/macOS auto-installer
├── install.bat               # Windows auto-installer
├── README.md
└── medserver/
    ├── __init__.py            # Version info
    ├── cli.py                 # CLI entrypoint (medserver command)
    ├── models.py              # Model registry & API schemas
    ├── engine.py              # Hybrid engine (SGLang + Transformers)
    ├── server.py              # FastAPI server
    └── static/
        ├── index.html         # Clinical web UI
        ├── styles.css         # Premium dark theme
        └── app.js             # Frontend application logic
```

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Clinical web UI |
| `GET` | `/api/health` | Server status, GPU info, model status |
| `GET` | `/api/models` | All available model variants |
| `GET` | `/api/model-info` | Currently loaded model details |
| `POST` | `/api/chat` | Chat completions (SSE streaming) |
| `POST` | `/api/analyze` | Multimodal image analysis (multipart form) |

### Chat API Example

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Differential diagnosis for chest pain?"}],
    "max_tokens": 1024,
    "temperature": 0.3,
    "stream": true
  }'
```

---

## ⚠️ Clinical Disclaimer

MedGemma is developed by Google for **research and clinical decision support**. It is:

- ❌ **NOT** FDA-approved or CE-marked
- ❌ **NOT** a substitute for clinical judgment
- ❌ **NOT** intended for direct patient care without validation

All AI outputs require professional clinical validation before use in patient care.

---

## 📄 License

Apache 2.0 — See [LICENSE](LICENSE) for details.

MedGemma models are subject to the [Health AI Developer Foundation License](https://huggingface.co/google/medgemma-1.5-4b-it).
