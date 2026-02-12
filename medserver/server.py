"""FastAPI server for MedServer — serves both API and clinical frontend."""

import asyncio
import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from medserver import __version__, __app_name__
from medserver.engine import MedGemmaEngine, get_gpu_info
from medserver.models import (
    AnalyzeRequest,
    ChatRequest,
    HealthResponse,
    list_models,
)

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
) -> FastAPI:
    """Create and configure the FastAPI application."""

    from medserver.models import get_model

    model_info = get_model(model_key)
    start_time = time.time()

    app = FastAPI(
        title=__app_name__,
        version=__version__,
        description="Self-hosted MedGemma clinical AI server",
    )

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
    async def chat(request: ChatRequest, raw_request: Request):
        """Chat completions — streaming or non-streaming."""
        if not engine.is_loaded:
            raise HTTPException(503, "Model is still loading. Please wait.")

        full_messages = []
        effective_system_prompt = request.system_prompt if request.system_prompt is not None else SYSTEM_PROMPT
        if effective_system_prompt:
            full_messages.append({"role": "system", "content": effective_system_prompt})
        
        images = []
        last_images_data = None
        
        from PIL import Image as PILImage
        import base64
        import io
        import threading

        for m in request.messages:
            full_messages.append({"role": m.role, "content": m.content})
            if m.image_data and model_info.supports_images:
                last_images_data = m.image_data

        if last_images_data:
            for img_b64 in last_images_data:
                try:
                    # Handle base64 data URL
                    header, encoded = img_b64.split(",", 1) if "," in img_b64 else (None, img_b64)
                    image_bytes = base64.b64decode(encoded)
                    pil_image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
                    images.append(pil_image)
                except Exception as e:
                    logger.error(f"Failed to decode image: {e}")

        if request.stream:
            stop_event = threading.Event()
            return StreamingResponse(
                _stream_chat(raw_request, full_messages, request.max_tokens, request.temperature, images if images else None, stop_event),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            result = await engine.generate(
                prompt=full_messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                images=images if images else None,
            )
            return {"response": result}

    async def _stream_chat(request: Request, messages: list, max_tokens: int, temperature: float, images: Optional[list], stop_event: threading.Event):
        """SSE stream generator for chat responses."""
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
    async def analyze_image(
        image: UploadFile = File(...),
        prompt: str = Form("Analyze this medical image and provide clinical findings."),
        max_tokens: int = Form(2048),
        temperature: float = Form(0.3),
    ):
        """Multimodal image analysis (4B and 27B multimodal models only)."""
        if not model_info.supports_images:
            raise HTTPException(
                400,
                f"Image analysis is not supported by {model_info.name}. "
                f"Use a multimodal model (-m 4 or -m 27).",
            )

        if not engine.is_loaded:
            raise HTTPException(503, "Model is still loading. Please wait.")

        # Read and validate image
        contents = await image.read()
        if len(contents) == 0:
            raise HTTPException(400, "Empty image file.")
        if len(contents) > 20 * 1024 * 1024:  # 20MB limit
            raise HTTPException(400, "Image too large. Max 20MB.")

        try:
            from PIL import Image as PILImage
            pil_image = PILImage.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(400, "Invalid image format.")

        # Format prompt with image placeholder for Gemma
        full_prompt = engine.format_chat_prompt(
            [{"role": "user", "content": prompt}],
            system_prompt=SYSTEM_PROMPT,
            num_images=1
        )

        return StreamingResponse(
            _stream_analyze(raw_request, full_prompt, [pil_image], max_tokens, temperature),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def _stream_analyze(
        request: Request,
        prompt: str,
        images: list,
        max_tokens: int,
        temperature: float,
    ):
        """SSE stream generator for image analysis."""
        try:
            async for token in engine.stream_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                images=images,
            ):
                if await request.is_disconnected():
                    break
                data = json.dumps({"token": token})
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Image analysis streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return app
