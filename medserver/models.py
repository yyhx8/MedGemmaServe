"""Model registry and API schemas for MedServer."""

from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

@dataclass
class MedGemmaModel:
    """Metadata for a single MedGemma model variant."""

    model_id: str  # HuggingFace model ID
    name: str  # Human-friendly name
    param_key: str  # CLI -m value (e.g. "4", "27", "27t")
    param_billions: float  # Parameter count in billions
    modality: str  # "multimodal" or "text"
    min_vram_gb: int  # Minimum VRAM in GB
    description: str
    recommended_gpus: list[str] = field(default_factory=list)
    supports_images: bool = False
    max_context_length: int = 8192


# The three confirmed models
MODEL_REGISTRY: dict[str, MedGemmaModel] = {
    "4": MedGemmaModel(
        model_id="google/medgemma-1.5-4b-it",
        name="MedGemma 1.5 4B",
        param_key="4",
        param_billions=4.0,
        modality="multimodal",
        min_vram_gb=16,
        description=(
            "Latest MedGemma 1.5 — 4B multimodal model with improved support "
            "for CT, MRI, histopathology, chest X-ray, and 2D medical images. "
            "Best balance of quality and hardware accessibility."
        ),
        recommended_gpus=["RTX 4090", "RTX A5000", "A100", "L4"],
        supports_images=True,
        max_context_length=8192,
    ),
    "27": MedGemmaModel(
        model_id="google/medgemma-27b-it",
        name="MedGemma 27B Multimodal",
        param_key="27",
        param_billions=27.0,
        modality="multimodal",
        min_vram_gb=32,
        description=(
            "Full 27B multimodal model — highest quality for complex medical "
            "imaging analysis combined with clinical text reasoning. "
            "Supports EHR interpretation and longitudinal studies."
        ),
        recommended_gpus=["A100", "A6000", "H100"],
        supports_images=True,
        max_context_length=8192,
    ),
    "27t": MedGemmaModel(
        model_id="google/medgemma-27b-text-it",
        name="MedGemma 27B Text",
        param_key="27t",
        param_billions=27.0,
        modality="text",
        min_vram_gb=32,
        description=(
            "Text-only 27B model — best pure clinical reasoning performance. "
            "Optimized for medical Q&A, differential diagnosis, clinical "
            "summarization, and triage without image processing overhead."
        ),
        recommended_gpus=["A100", "A6000", "H100"],
        supports_images=False,
        max_context_length=8192,
    ),
}


def get_model(key: str) -> MedGemmaModel:
    """Look up a model by its CLI key. Raises KeyError with helpful message."""
    if key not in MODEL_REGISTRY:
        valid = ", ".join(
            f"-m {k} ({v.name})" for k, v in MODEL_REGISTRY.items()
        )
        raise KeyError(
            f"Unknown model key '{key}'. Valid options: {valid}"
        )
    return MODEL_REGISTRY[key]


def list_models() -> list[dict]:
    """Return all models as serializable dicts for the API."""
    return [
        {
            "key": m.param_key,
            "model_id": m.model_id,
            "name": m.name,
            "param_billions": m.param_billions,
            "modality": m.modality,
            "min_vram_gb": m.min_vram_gb,
            "description": m.description,
            "supports_images": m.supports_images,
            "recommended_gpus": m.recommended_gpus,
        }
        for m in MODEL_REGISTRY.values()
    ]


# ---------------------------------------------------------------------------
# API Schemas (Pydantic)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 0.3
    stream: bool = True


class AnalyzeRequest(BaseModel):
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.3


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_id: str
    modality: str
    supports_images: bool
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    host: str
    port: int
    uptime_seconds: float
