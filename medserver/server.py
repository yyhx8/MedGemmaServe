"""FastAPI server for MedServer — serves both API and clinical frontend."""

import asyncio
import base64
import io
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage

from medserver import __version__, __app_name__
from medserver.engine import MedGemmaEngine, get_gpu_info
from medserver.models import (
    AnalyzeRequest,
    ChatRequest,
    HealthResponse,
    list_models,
)

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger("medserver.server")

STATIC_DIR = Path(__file__).parent / "static"

# Clinical system prompt for MedGemma
SYSTEM_PROMPT = (
    "You are MedGemma, a specialized clinical AI assistant. "
    "Your goal is to provide precise, evidence-based medical information to healthcare professionals. "
    "Structure your responses clearly using clinical terminology. "
    "Always clarify that your analysis is for decision support and requires validation by a qualified clinician."
)


def create_app(
    engine: MedGemmaEngine,
    host: str = "0.0.0.0",
    port: int = 8000,
    model_key: str = "4",
    max_user_streams: int = 1,
    rate_limit: str = "20/minute",
    max_history_messages: int = 100,
    max_text_length: int = 50000,
    max_image_count: int = 10,
    max_payload_mb: int = 20,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    from medserver.models import get_model

    model_info = get_model(model_key)
    start_time = time.time()

    # Rate Limiter setup
    limiter = Limiter(key_func=get_remote_address)
    app = FastAPI(
        title=__app_name__,
        version=__version__,
        description="Self-hosted MedGemma clinical AI server",
    )
    app.state.limiter = limiter
    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please wait a moment before trying again."},
        )

    # Per-user concurrency locks
    user_locks = {}

    def get_user_lock(ip: str) -> asyncio.Semaphore:
        if ip not in user_locks:
            user_locks[ip] = asyncio.Semaphore(max_user_streams)
        return user_locks[ip]

    # Global Error Handling
    @app.exception_handler(RuntimeError)
    async def cuda_error_handler(request: Request, exc: RuntimeError):
        if "out of memory" in str(exc).lower():
            return JSONResponse(
                status_code=507,
                content={"detail": "GPU Out of Memory. Try using quantization (-q) or a smaller model."},
            )
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    # Wide-open CORS for LAN access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def serve_frontend():
        """Serve the clinical web UI."""
        index_path = STATIC_DIR / "index.html"
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

    @app.get("/api/health")
    async def health_check():
        """Server health + GPU info + model status."""
        gpu = get_gpu_info()
        return HealthResponse(
            status="ready" if engine.is_loaded else "loading",
            model_name=model_info.name,
            model_id=model_info.model_id,
            modality=model_info.modality,
            supports_images=model_info.supports_images,
            gpu_available=gpu["gpu_available"],
            gpu_name=gpu.get("gpu_name"),
            gpu_vram_gb=gpu.get("gpu_vram_gb"),
            host=host,
            port=port,
            uptime_seconds=round(time.time() - start_time, 1),
            max_text_length=max_text_length,
        )

    @app.get("/api/models")
    async def get_models():
        """Return all available MedGemma model variants."""
        models = list_models()
        return {
            "models": models,
            "active_model": model_info.param_key,
        }

    @app.get("/api/model-info")
    async def get_model_info():
        """Return info about the currently loaded model."""
        return {
            "key": model_info.param_key,
            "model_id": model_info.model_id,
            "name": model_info.name,
            "param_billions": model_info.param_billions,
            "modality": model_info.modality,
            "min_vram_gb": model_info.min_vram_gb,
            "description": model_info.description,
            "supports_images": model_info.supports_images,
            "recommended_gpus": model_info.recommended_gpus,
            "engine_load_time_s": round(engine.load_time, 1),
        }

    @app.post("/api/chat")
    @limiter.limit(rate_limit)
    async def chat(chat_data: ChatRequest, request: Request):
        """Chat completions — streaming or non-streaming."""
        if not engine.is_loaded:
            raise HTTPException(503, "Model is still loading. Please wait.")

        # Per-user lock check
        user_ip = get_remote_address(request)
        lock = get_user_lock(user_ip)
        
        if lock.locked():
             raise HTTPException(
                 429, 
                 f"You already have {max_user_streams} active stream(s). Please wait for them to finish."
             )
             
        if len(chat_data.messages) > max_history_messages:
            raise HTTPException(400, f"Too many messages. Maximum allowed is {max_history_messages}.")
            
        if chat_data.system_prompt and len(chat_data.system_prompt) > max_text_length:
            raise HTTPException(400, f"System prompt too long. Maximum allowed is {max_text_length} characters.")

        full_messages = []
        effective_system_prompt = chat_data.system_prompt if chat_data.system_prompt is not None else SYSTEM_PROMPT
        if effective_system_prompt:
            full_messages.append({"role": "system", "content": effective_system_prompt})
        
        images = []
        
        for m in chat_data.messages:
            # Create a structured message for the engine
            msg_content = m.content
            
            # String length validation
            if isinstance(msg_content, str) and len(msg_content) > max_text_length:
                raise HTTPException(400, f"Message content too long. Maximum allowed is {max_text_length} characters.")
            
            if m.image_data and len(m.image_data) > max_image_count:
                raise HTTPException(400, f"Too many images in a single message. Maximum allowed is {max_image_count}.")
            
            if m.image_data and model_info.supports_images:
                # Ensure the content has image placeholders if image_data is present
                if isinstance(msg_content, str):
                    # Convert string content to structured list and prepend placeholders
                    msg_content = [{"type": "image"}] * len(m.image_data) + [{"type": "text", "text": msg_content}]
                elif isinstance(msg_content, list):
                    # Check if the list already has enough image placeholders
                    img_placeholder_count = sum(1 for item in msg_content if item.get("type") == "image")
                    if img_placeholder_count < len(m.image_data):
                        missing = len(m.image_data) - img_placeholder_count
                        msg_content = [{"type": "image"}] * missing + msg_content
                
                # Collect the actual image data
                for img_b64 in m.image_data:
                    try:
                        header, encoded = img_b64.split(",", 1) if "," in img_b64 else (None, img_b64)
                        
                        # Crude base64 length check before decoding (Base64 is ~33% larger than raw data)
                        max_b64_len = int(max_payload_mb * 1024 * 1024 * 1.35)
                        if len(encoded) > max_b64_len:
                             raise ValueError(f"Image payload too large before decoding. Max is {max_payload_mb}MB.")
                             
                        image_bytes = base64.b64decode(encoded)
                        
                        # Strict size check after decoding
                        if len(image_bytes) > max_payload_mb * 1024 * 1024:
                             raise ValueError(f"Decoded image exceeds {max_payload_mb}MB limit.")
                             
                        pil_image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
                        images.append(pil_image)
                    except Exception as e:
                        logger.error(f"Failed to decode image: {e}")
                        raise HTTPException(400, f"Invalid image or image too large: {e}")

            full_messages.append({"role": m.role, "content": msg_content})

        # If model doesn't support images, ensure we don't pass any
        if not model_info.supports_images and images:
            logger.warning(f"Model {model_info.name} does not support images. Ignoring attached images.")
            images = []

        if chat_data.stream:
            stop_event = threading.Event()
            return StreamingResponse(
                _stream_chat(request, full_messages, chat_data.max_tokens, chat_data.temperature, images if images else None, stop_event, lock),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            async with lock:
                result = await engine.generate(
                    prompt=full_messages,
                    max_tokens=chat_data.max_tokens,
                    temperature=chat_data.temperature,
                    images=images if images else None,
                )
            return {"response": result}

    async def _stream_chat(request: Request, messages: list, max_tokens: int, temperature: float, images: Optional[list], stop_event: threading.Event, lock: asyncio.Semaphore):
        """SSE stream generator for chat responses."""
        async with lock:
            try:
                async for token in engine.stream_generate(
                    prompt=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    images=images,
                    stop_event=stop_event
                ):
                    if await request.is_disconnected():
                        stop_event.set()
                        break
                    data = json.dumps({"token": token})
                    yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                stop_event.set()

    @app.post("/api/analyze")
    @limiter.limit(rate_limit)
    async def analyze_image(
        request: Request,
        image: UploadFile = File(...),
        prompt: str = Form("Analyze this medical image and provide clinical findings."),
        max_tokens: int = Form(2048),
        temperature: float = Form(0.3),
    ):
        """Multimodal image analysis (4B and 27B multimodal models only)."""
        if not model_info.supports_images:
            raise HTTPException(400, f"Model {model_info.name} does not support images.")
             
        if len(prompt) > max_text_length:
             raise HTTPException(400, f"Prompt too long. Maximum allowed is {max_text_length} characters.")

        # Read and validate image
        if image.size is not None and image.size > max_payload_mb * 1024 * 1024:
            raise HTTPException(400, f"Image too large. Max {max_payload_mb}MB.")
            
        contents = await image.read()
        if len(contents) == 0:
            raise HTTPException(400, "Empty image file.")
        if len(contents) > max_payload_mb * 1024 * 1024:  # fallback check
            raise HTTPException(400, f"Image too large. Max {max_payload_mb}MB.")

        try:
            pil_image = PILImage.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(400, "Invalid image format.")

        # Per-user lock check
        user_ip = get_remote_address(request)
        lock = get_user_lock(user_ip)
        
        if lock.locked():
             raise HTTPException(
                 429, 
                 f"You already have {max_user_streams} active stream(s). Please wait for them to finish."
             )

        # Use structured content to ensure the engine sees the image placeholder
        user_content = [{"type": "image"}, {"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": user_content}]
        
        if SYSTEM_PROMPT:
             messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        stop_event = threading.Event()
        return StreamingResponse(
            _stream_analyze(request, messages, [pil_image], max_tokens, temperature, stop_event, lock),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def _stream_analyze(
        request: Request,
        messages: list,
        images: list,
        max_tokens: int,
        temperature: float,
        stop_event: threading.Event,
        lock: asyncio.Semaphore,
    ):
        """SSE stream generator for image analysis."""
        async with lock:
            try:
                async for token in engine.stream_generate(
                    prompt=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    images=images,
                    stop_event=stop_event
                ):
                    if await request.is_disconnected():
                        stop_event.set()
                        break
                    data = json.dumps({"token": token})
                    yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Image analysis streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                stop_event.set()

    return app
