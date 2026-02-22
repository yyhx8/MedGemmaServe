# 🧬 MedServer

**An Edge-Optimized, Human-Centered Interface for MedGemma in Clinical Settings.**

MedServer is a one-command platform that auto-installs, downloads, and serves MedGemma models with a premium clinical web UI designed for high-stakes medical reasoning and multimodal analysis.

```bash
medserver -m 4 -p 7070 -ip 192.168.1.50
```

![Clinical Dark UI](medserver/static/screenshot.png)

---

## ⚡ Key Features

- **🚀 One-command Setup**: Automated installer for Linux, macOS, and Windows.
- **🧠 Clinical Reasoning**: Native support for MedGemma "Thinking Process" traces with collapsible UI, powered by a self-healing **Hybrid Engine**.
- **🖼️ Multimodal Mastery**: Analysis of medical images (single or multiple) with an integrated clinical lightbox.
- **💎 Premium Clinical UI**: Professional dark-mode interface designed for clinical environments.
- **🛠️ Message Management**: Interactive controls to Copy, Edit, or Delete messages for clinical workflow flexibility.
- **🎨 Refined Typography**: Optimized markdown rendering with clear spacing and hierarchical clarity.
- **📂 Session Persistence**: Automatic local history management with persistent chat sessions.
- **📱 Mobile Optimized**: Responsive design with touch-friendly lightbox and pinch-to-zoom support.
- **⚙️ Dual-Engine Architecture**: Automatically selects the fastest inference engine (SGLang or Transformers).
- **🛡️ Robust Security**: Built-in rate limiting and concurrency controls to prevent server overload.
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
| `--force-transformers` | Force Transformers engine (disables SGLang) | off |
| `-v`, `--version` | Show program's version number and exit | — |
| `--workers` | Number of server workers (uvicorn) | `1` |
| `--max-user-streams` | Max concurrent streams per user IP | `1` |
| `--rate-limit` | API rate limit (e.g., '10/minute') | `20/minute` |
| `--max-history-messages` | Max messages allowed in chat history | `100` |
| `--max-text-length` | Max characters allowed per individual message | `50000` |
| `--max-image-count` | Max images allowed per chat message | `10` |
| `--max-payload-mb` | Max image upload size in MB | `20` |
| `--show-hardware-stats` | Expose GPU/VRAM usage to frontend | `False` |
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

MedServer features a hybrid architecture that automatically selects the most efficient inference engine, with **automatic fallback** for maximum reliability:

1.  **SGLang Engine** (High Performance):
    - **Trigger:** Linux + NVIDIA Ampere GPU (or newer, CC >= 8.0) + `sglang` installed.
    - **Benefits:** Up to 5x faster throughput, advanced memory management (RadixAttention), and optimized streaming.
    - **Quantization Note:** SGLang currently defaults to `bfloat16` (16-bit) and ignores the `-q` flag. If you require 4-bit quantization on Linux to save VRAM, use the `--force-transformers` flag to use the Transformers backend.
2.  **Transformers Engine** (Universal Compatibility):
    - **Trigger:** Windows, older GPUs, or if SGLang fails to load.
    - **Benefits:** Runs everywhere PyTorch runs. Uses `bitsandbytes` for 4-bit quantization.

> 🛡️ **Automatic Fallback:** If the high-performance SGLang engine fails to initialize (e.g., due to specific driver incompatibilities), MedServer will automatically fall back to the universal Transformers engine to ensure the service remains available.

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

## 🛡️ Robustness & Security

MedServer includes built-in protection against overload and abuse:

- **Rate Limiting:** Prevents API spam by limiting requests per IP address (default: 20/minute). Configurable via `--rate-limit`.
- **Concurrency Control:** Limits the number of simultaneous active generation streams per user to prevent GPU OOM errors (default: 1). Configurable via `--max-user-streams`.
- **Graceful Degradation:** Returns HTTP 429 (Too Many Requests) when limits are exceeded, ensuring the server remains responsive for other users.

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
| `POST` | `/api/chat` | Chat completions (SSE streaming, supports multiple images) |
| `POST` | `/api/analyze` | Image analysis (multipart form: image + prompt) |

### Chat API Example (Multiple Images)

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user", 
        "content": "Compare these two scans for progression.",
        "image_data": ["data:image/jpeg;base64,...", "data:image/jpeg;base64,..."]
      }
    ],
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
