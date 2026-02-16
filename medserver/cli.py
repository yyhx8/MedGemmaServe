"""CLI entrypoint for MedServer.

Usage:
    medserver -p 7070 -ip 192.168.1.50 -m 4
    medserver --port 8000 --host 0.0.0.0 --model 27t --quantize
    medserver --help
"""

import argparse
import asyncio
import logging
import os
import socket
import sys
import textwrap

BANNER = r"""
  __  __          _ ____
 |  \/  | ___  __| / ___|  ___ _ ____   _____ _ __
 | |\/| |/ _ \/ _` \___ \ / _ \ '__\ \ / / _ \ '__|
 | |  | |  __/ (_| |___) |  __/ |   \ V /  __/ |
 |_|  |_|\___|\__,_|____/ \___|_|    \_/ \___|_|

 [ MedGemma Clinical AI Server — v{version} ]
"""


def get_local_ip() -> str:
    """Get the machine's local WiFi/LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="medserver",
        description="MedGemma Clinical AI Server — one-command model serving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              medserver -m 4                        # Serve MedGemma 1.5 4B on default port 8000
              medserver -p 7070 -m 27               # Serve MedGemma 27B multimodal on port 7070
              medserver -p 7070 -ip 192.168.1.50 -m 4   # Bind to specific WiFi IP
              medserver -m 4 -q                     # 4-bit quantized (lower VRAM)
              medserver -m 27t                      # Text-only 27B model

            Models:
              -m 4    MedGemma 1.5 4B  (multimodal, ≥16GB VRAM)
              -m 27   MedGemma 27B     (multimodal, ≥32GB VRAM)
              -m 27t  MedGemma 27B     (text-only,  ≥32GB VRAM)
        """),
    )

    from medserver import __version__
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"MedServer {__version__}",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "-ip", "--host",
        type=str,
        default="0.0.0.0",
        help="Bind address — use your WiFi IP or 0.0.0.0 for all interfaces (default: 0.0.0.0)",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        choices=["4", "27", "27t"],
        help="Model to serve: 4 = MedGemma 1.5 4B, 27 = 27B multimodal, 27t = 27B text-only",
    )
    parser.add_argument(
        "-q", "--quantize",
        action="store_true",
        default=False,
        help="Enable 4-bit quantization (reduces VRAM usage ~50%%)",
    )
    parser.add_argument(
        "--force-transformers",
        action="store_true",
        default=False,
        help="Force use of Transformers engine even if SGLang is available",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for downloading gated models (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum context length in tokens (default: 8192)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory to use (default: 0.90)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn workers (default: 1)",
    )
    parser.add_argument(
        "--max-user-streams",
        type=int,
        default=1,
        help="Maximum simultaneous generation streams per user IP (default: 1)",
    )
    parser.add_argument(
        "--rate-limit",
        type=str,
        default="20/minute",
        help="Rate limit per user IP (e.g., '10/minute', '100/day') (default: 20/minute)",
    )

    return parser


def check_gpu():
    """Check GPU availability and print info."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("\n⚠️  WARNING: No CUDA GPU detected!")
            print("   MedGemma requires a CUDA-capable GPU to run.")
            print("   The server will attempt to start but inference will fail.")
            print()
            return False

        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  🖥️  GPU: {gpu_name}")
        print(f"  📊 VRAM: {vram:.1f} GB")
        if gpu_count > 1:
            print(f"  🔢 GPU Count: {gpu_count}")
        return True
    except ImportError:
        print("\n⚠️  WARNING: PyTorch not installed. Run the install script first.")
        return False


def check_hf_token(token: str | None) -> str | None:
    """Resolve HuggingFace token from arg or environment."""
    if token:
        return token
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        print("  🔑 HuggingFace token: found in environment")
        return env_token
    print("  ⚠️  No HuggingFace token found. MedGemma models are gated —")
    print("     you may need to set HF_TOKEN or pass --hf-token.")
    print("     Get a token at: https://huggingface.co/settings/tokens")
    return None


def main():
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    # Import here to avoid slow imports on --help
    from medserver import __version__
    from medserver.models import get_model

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Print banner
    print(BANNER.format(version=__version__))

    # Resolve model
    try:
        model = get_model(args.model)
    except KeyError as e:
        print(f"❌ {e}")
        sys.exit(1)

    print(f"  📦 Model: {model.name}")
    print(f"  🏷️  HuggingFace: {model.model_id}")
    print(f"  🔬 Modality: {model.modality}")
    print(f"  💾 Min VRAM: {model.min_vram_gb} GB")

    # Check GPU
    has_gpu = check_gpu()

    if args.quantize and not has_gpu:
        print("❌ ERROR: 4-bit quantization requires a CUDA GPU.")
        print("   Please run without -q or install CUDA/PyTorch correctly.")
        sys.exit(1)

    # Check HF token
    hf_token = check_hf_token(args.hf_token)

    # Resolve display address
    local_ip = get_local_ip()
    display_host = local_ip if args.host == "0.0.0.0" else args.host

    print()
    print(f"  🌐 Server: http://{display_host}:{args.port}")
    if args.host == "0.0.0.0":
        print(f"  📡 LAN:    http://{local_ip}:{args.port}")
        print(f"  🏠 Local:  http://127.0.0.1:{args.port}")
    if args.quantize:
        print(f"  ⚡ Quantization: 4-bit (reduced VRAM)")
    print()
    print("  " + "─" * 50)
    print("  ⏳ Loading model... (this may take a few minutes)")
    print("  " + "─" * 50)
    print()

    # Create engine and app
    from medserver.engine import MedGemmaEngine
    from medserver.server import create_app

    engine = MedGemmaEngine(
        model_id=model.model_id,
        supports_images=model.supports_images,
        quantize=args.quantize,
        force_transformers=args.force_transformers,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        hf_token=hf_token,
    )

    app = create_app(
        engine=engine,
        host=args.host,
        port=args.port,
        model_key=args.model,
        max_user_streams=args.max_user_streams,
        rate_limit=args.rate_limit,
    )

    # Startup event: load model
    @app.on_event("startup")
    async def on_startup():
        await engine.load()
        print()
        print("  " + "─" * 50)
        print(f"  ✅ Model loaded in {engine.load_time:.1f}s")
        print(f"  🚀 Server ready at http://{display_host}:{args.port}")
        print("  " + "─" * 50)
        print()

    # Launch uvicorn
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
