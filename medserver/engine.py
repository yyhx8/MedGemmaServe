"""vLLM engine wrapper for MedGemma model loading and inference."""

import asyncio
import logging
import time
from typing import AsyncIterator, Optional

import torch

logger = logging.getLogger("medserver.engine")


class MedGemmaEngine:
    """Wraps vLLM's AsyncLLMEngine for MedGemma inference.

    Handles model loading, tokenizer setup, and provides both
    streaming and non-streaming generation methods.
    """

    def __init__(
        self,
        model_id: str,
        supports_images: bool = False,
        quantize: bool = False,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.90,
        hf_token: Optional[str] = None,
    ):
        self.model_id = model_id
        self.supports_images = supports_images
        self.quantize = quantize
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.hf_token = hf_token

        self._engine = None
        self._tokenizer = None
        self._loaded = False
        self._load_time: float = 0
        self._request_counter: int = 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_time(self) -> float:
        return self._load_time

    async def load(self) -> None:
        """Initialize and load the vLLM engine with the MedGemma model."""
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        logger.info(f"Loading model: {self.model_id}")
        logger.info(f"Quantization: {'4-bit' if self.quantize else 'none'}")
        logger.info(f"Max context length: {self.max_model_len}")
        logger.info(f"GPU memory utilization: {self.gpu_memory_utilization:.0%}")

        start = time.monotonic()

        engine_args_kwargs = {
            "model": self.model_id,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": True,
            "enforce_eager": False,
        }

        # Quantization support
        if self.quantize:
            engine_args_kwargs["quantization"] = "bitsandbytes"
            engine_args_kwargs["enforce_eager"] = True
            engine_args_kwargs["dtype"] = "bfloat16"  
            logger.info("Using 4-bit BitsAndBytes quantization")

        # HuggingFace token for gated models
        if self.hf_token:
            engine_args_kwargs["download_dir"] = None  # default cache
            # vLLM reads HF_TOKEN env var automatically, but we also set it
            import os
            os.environ["HF_TOKEN"] = self.hf_token

        # Multimodal configuration
        if self.supports_images:
            engine_args_kwargs["limit_mm_per_prompt"] = {"image": 5}
            logger.info("Multimodal mode enabled (image support)")

        engine_args = AsyncEngineArgs(**engine_args_kwargs)
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

        self._load_time = time.monotonic() - start
        self._loaded = True
        logger.info(
            f"Model loaded successfully in {self._load_time:.1f}s"
        )

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        images: Optional[list] = None,
    ) -> str:
        """Generate a complete (non-streaming) response."""
        chunks = []
        async for chunk in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            images=images,
        ):
            chunks.append(chunk)
        return "".join(chunks)

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        images: Optional[list] = None,
    ) -> AsyncIterator[str]:
        """Stream tokens as they are generated."""
        from vllm import SamplingParams

        if not self._loaded:
            raise RuntimeError("Engine not loaded. Call load() first.")

        self._request_counter += 1
        request_id = f"medserver-{self._request_counter}"

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.05,
            stop=["<end_of_turn>", "<eos>"],
        )

        # Build multi-modal inputs if images provided
        inputs = {"prompt": prompt}
        if images and self.supports_images:
            multi_modal_data = {"image": images}
            inputs["multi_modal_data"] = multi_modal_data

        # Stream results
        results_generator = self._engine.generate(
            inputs,
            sampling_params,
            request_id=request_id,
        )

        previous_text = ""
        async for request_output in results_generator:
            if request_output.outputs:
                current_text = request_output.outputs[0].text
                new_text = current_text[len(previous_text):]
                previous_text = current_text
                if new_text:
                    yield new_text

    def format_chat_prompt(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Format messages into Gemma chat template.

        MedGemma uses the Gemma 3 chat format:
        <start_of_turn>user
        {message}<end_of_turn>
        <start_of_turn>model
        {response}<end_of_turn>
        """
        parts = []

        # System prompt as first user context
        if system_prompt:
            parts.append(f"<start_of_turn>user\n{system_prompt}<end_of_turn>")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "assistant":
                parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")

        # Add the model turn prompt
        parts.append("<start_of_turn>model\n")

        return "\n".join(parts)


def get_gpu_info() -> dict:
    """Get GPU information for health checks."""
    info = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_vram_gb": None,
        "gpu_count": 0,
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_mem
        info["gpu_vram_gb"] = round(total_mem / (1024**3), 1)
    return info
