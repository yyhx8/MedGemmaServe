"""Hybrid engine for MedGemma inference (SGLang + Transformers)."""

import asyncio
import logging
import platform
import threading
import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, List, Dict, Any, Union

import torch

# Configure logging
logger = logging.getLogger("medserver.engine")


class BaseEngine(ABC):
    """Abstract base class for inference engines."""

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

        self._loaded = False
        self._load_time: float = 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_time(self) -> float:
        return self._load_time

    @abstractmethod
    async def load(self) -> None:
        """Initialize and load the model."""
        pass

    @abstractmethod
    async def stream_generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: int = 2048,
        temperature: float = 0.3,
        images: Optional[list] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> AsyncIterator[str]:
        """Stream tokens as they are generated."""
        pass

    async def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
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

    def format_chat_prompt(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        num_images: int = 0,
    ) -> str:
        """Format messages into Gemma chat template with vision support."""
        parts = []

        # MedGemma 1.5 (PaliGemma 2 based) requires <image> tokens inside the user turn.
        # We'll inject them into the first user message.
        image_injected = False
        image_tokens = ""
        if num_images > 0:
             # Standard PaliGemma format is <image> tokens followed by a newline
             image_tokens = "<image>" * num_images + "\n"
             logger.debug(f"Prepared {num_images} image tokens for injection")

        # Extract system prompt
        if not system_prompt:
            for msg in messages:
                if msg.get("role") == "system":
                    system_prompt = msg.get("content", "")
                    if isinstance(system_prompt, list):
                        text_parts = [item.get("text", "") for item in system_prompt if item.get("type") == "text"]
                        system_prompt = " ".join(text_parts)
                    break

        # System prompt as virtual user turn
        if system_prompt:
            parts.append(f"<start_of_turn>user\n{system_prompt}<end_of_turn>")

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if isinstance(content, list):
                text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                content = " ".join(text_parts)

            if role == "user":
                # Inject image tokens into the first user turn if they haven't been yet
                if not image_injected and image_tokens:
                    content = image_tokens + content
                    image_injected = True
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "assistant" or role == "model":
                parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")

        # Ensure tokens are added even if no user message was found (unlikely but safe)
        if not image_injected and image_tokens and parts:
             parts[0] = parts[0].replace("<start_of_turn>user\n", f"<start_of_turn>user\n{image_tokens}")

        # Final model turn trigger
        parts.append("<start_of_turn>model\n")
        
        full_prompt = "\n".join(parts)
        return full_prompt


class SGLangEngine(BaseEngine):
    """SGLang implementation (Linux + Ampere+ GPUs for high performance)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine = None

    async def load(self) -> None:
        try:
            import sglang
        except ImportError:
            raise ImportError("sglang not installed. This engine requires Linux.")

        logger.info(f"Loading model with SGLang: {self.model_id}")
        start = time.monotonic()

        # Set env var for HF token if needed
        if self.hf_token:
            import os
            os.environ["HF_TOKEN"] = self.hf_token

        # Initialize SGLang Engine
        # SGLang's Engine API mimics vLLM somewhat but is optimized for structured gen
        # We use the lower-level Engine or Runtime depending on version.
        # Assuming v0.1+ pattern:
        self._engine = sglang.Engine(
            model_path=self.model_id,
            max_model_len=self.max_model_len,
            mem_fraction_static=self.gpu_memory_utilization,
            trust_remote_code=True,
            tp_size=1,  # Tensor parallelism (default 1 for simplicity)
        )

        self._load_time = time.monotonic() - start
        self._loaded = True
        logger.info(f"SGLang model loaded in {self._load_time:.1f}s")

    async def stream_generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: int = 2048,
        temperature: float = 0.3,
        images: Optional[list] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> AsyncIterator[str]:
        if not self._loaded:
            raise RuntimeError("Engine not loaded.")

        # Always format structured prompt for consistency
        final_prompt = prompt
        if isinstance(prompt, list):
             final_prompt = self.format_chat_prompt(prompt, num_images=len(images) if images else 0)

        # Construct input
        inputs = {"text": final_prompt}
        if images and self.supports_images:
             # SGLang takes a list of PIL images or base64 strings
             inputs["image_data"] = images

        sampling_params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "stop": ["<end_of_turn>", "<eos>", "<|endoftext|>"],
        }

        # SGLang usually returns cumulative text in the 'text' field.
        # We need to yield deltas for the frontend.
        generator = self._engine.generate(
            inputs,
            sampling_params,
            stream=True
        )

        last_len = 0
        async for output in generator:
            if stop_event and stop_event.is_set():
                break
            
            new_text = output["text"]
            delta = new_text[last_len:]
            if delta:
                yield delta
            last_len = len(new_text)


class TransformersEngine(BaseEngine):
    """Transformers implementation (compatible with older GPUs/Windows)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None
        self._tokenizer = None
        self._processor = None

    async def load(self) -> None:
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
            AutoProcessor,
            BitsAndBytesConfig,
            TextIteratorStreamer
        )

        logger.info(f"Loading model with Transformers: {self.model_id}")
        start = time.monotonic()

        # Determine optimal compute dtype
        # NOTE: MedGemma (Gemma 2/3) is unstable in float16 (NaN errors).
        # We must use bfloat16 (Ampere+) or float32 (T4/Older).
        major, minor = (0, 0)
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
        
        if major >= 8:
            compute_dtype = torch.bfloat16
            logger.info(f"Using bfloat16 precision (Compute Capability {major}.{minor} detected)")
        else:
            compute_dtype = torch.float32
            logger.info(f"Falling back to float32 for stability (CC {major}.{minor} lacks stable float16 for Gemma)")

        # Quantization config
        quantization_config = None
        if self.quantize:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info(f"Enabling 4-bit quantization (compute_dtype={compute_dtype})")

        # Load Tokenizer/Processor
        try:
            if self.supports_images:
                self._processor = AutoProcessor.from_pretrained(
                    self.model_id, token=self.hf_token, trust_remote_code=True
                )
                if self._processor.tokenizer.pad_token_id is None:
                    self._processor.tokenizer.pad_token_id = self._processor.tokenizer.eos_token_id
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id, token=self.hf_token, trust_remote_code=True
                )
                if self._tokenizer.pad_token_id is None:
                    self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        except Exception as e:
            logger.error(f"Failed to load tokenizer/processor: {e}")
            raise

        # Load Model
        # Use AutoModelForImageTextToText for multimodal models (MedGemma 1.5)
        model_class = AutoModelForImageTextToText if self.supports_images else AutoModelForCausalLM
        
        self._model = model_class.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            token=self.hf_token,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True,
        )

        self._load_time = time.monotonic() - start
        self._loaded = True
        logger.info(f"Transformers model loaded in {self._load_time:.1f}s")

    async def stream_generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: int = 2048,
        temperature: float = 0.3,
        images: Optional[list] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> AsyncIterator[str]:
        from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

        if not self._loaded:
            raise RuntimeError("Engine not loaded.")

        # Prepare inputs
        inputs = None
        
        # Multimodal input handling
        if self.supports_images and self._processor and images:
            # Try to use processor's apply_chat_template (preferred for MedGemma 1.5)
            if hasattr(self._processor, "apply_chat_template") and isinstance(prompt, list):
                try:
                    formatted_messages = []
                    image_added = False
                    for msg in prompt:
                        role = msg.get("role")
                        content = msg.get("content", "")
                        
                        if role == "user" and not image_added:
                            new_content = []
                            for img in images:
                                new_content.append({"type": "image", "image": img})
                            
                            if isinstance(content, list):
                                new_content.extend([c for c in content if c.get("type") == "text"])
                            else:
                                new_content.append({"type": "text", "text": content})
                            
                            image_added = True
                            formatted_messages.append({"role": role, "content": new_content})
                        else:
                            # Standard text content
                            if isinstance(content, list):
                                text_content = " ".join([c.get("text", "") for c in content if c.get("type") == "text"])
                                formatted_messages.append({"role": role, "content": text_content})
                            else:
                                formatted_messages.append({"role": role, "content": content})
                    
                    inputs = self._processor.apply_chat_template(
                        formatted_messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt"
                    ).to(self._model.device)
                    
                    # If apply_chat_template didn't include pixel_values, generate them manually
                    if "pixel_values" not in inputs and images:
                        pixel_values = self._processor(images=images, return_tensors="pt")["pixel_values"]
                        inputs["pixel_values"] = pixel_values.to(self._model.device)

                    if "pixel_values" in inputs:
                        inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)
                except Exception as e:
                    logger.warning(f"apply_chat_template failed: {e}. Falling back to manual formatting.")
                    inputs = None

            if inputs is None:
                # Fallback to manual formatting
                final_prompt = prompt
                num_images = len(images) if images else 0
                if isinstance(prompt, list):
                    final_prompt = self.format_chat_prompt(prompt, num_images=num_images)
                
                inputs = self._processor(
                    text=[final_prompt],
                    images=images,
                    return_tensors="pt",
                    padding=True
                ).to(self._model.device)
        else:
            # Text-only input or engine without multimodal support
            final_prompt = prompt
            if isinstance(prompt, list):
                final_prompt = self.format_chat_prompt(prompt, num_images=0)
            
            if self._processor:
                 inputs = self._processor(
                    text=[final_prompt],
                    images=None,
                    return_tensors="pt"
                ).to(self._model.device)
            elif self._tokenizer:
                inputs = self._tokenizer(
                    final_prompt, 
                    return_tensors="pt"
                ).to(self._model.device)

        streamer = TextIteratorStreamer(
            self._processor.tokenizer if self._processor else self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        class StopOnEvent(StoppingCriteria):
            def __init__(self, event):
                self.event = event
            def __call__(self, input_ids, scores, **kwargs):
                return self.event.is_set() if self.event else False

        do_sample = temperature > 0
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            pad_token_id=self._processor.tokenizer.pad_token_id if self._processor else self._tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnEvent(stop_event)]) if stop_event else None
        )
        
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = 0.95
            generation_kwargs["repetition_penalty"] = 1.05
        
        # Run generation in a separate thread
        thread = threading.Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield tokens from streamer
        try:
            for text in streamer:
                yield text
                if stop_event and stop_event.is_set():
                    break
                await asyncio.sleep(0)
        finally:
            if stop_event:
                stop_event.set()
            thread.join()


def get_gpu_info() -> dict:
    """Get GPU information for health checks."""
    info = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_vram_gb": None,
        "gpu_count": 0,
        "compute_capability": None,
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory
        info["gpu_vram_gb"] = round(total_mem / (1024**3), 1)
        
        cc = torch.cuda.get_device_capability(0)
        info["compute_capability"] = f"{cc[0]}.{cc[1]}"
        
    return info


def MedGemmaEngine(
    model_id: str,
    supports_images: bool = False,
    quantize: bool = False,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.90,
    hf_token: Optional[str] = None,
) -> BaseEngine:
    """Factory: Returns SGLangEngine (Linux high-perf) or TransformersEngine (Fallback)."""
    
    use_sglang = False
    reason = "Unknown"

    # 1. Check OS
    is_linux = platform.system() == "Linux"
    if not is_linux:
        reason = "OS is not Linux (Windows detected)"
    
    # 2. Check SGLang installation & GPU Capability
    if is_linux:
        try:
            import sglang
            # Check Compute Capability >= 8.0 (Ampere+)
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                if major >= 8:
                    use_sglang = True
                else:
                    reason = f"GPU Compute Capability {major}.{minor} < 8.0 (SGLang requires Ampere+)"
            else:
                reason = "No CUDA GPU detected"
        except ImportError:
            reason = "sglang not installed"

    if use_sglang:
        logger.info(f"Selecting SGLang Engine (Linux + CC >= 8.0 detected)")
        return SGLangEngine(
            model_id=model_id,
            supports_images=supports_images,
            quantize=quantize,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            hf_token=hf_token,
        )
    else:
        logger.info(f"Selecting Transformers Engine ({reason})")
        return TransformersEngine(
            model_id=model_id,
            supports_images=supports_images,
            quantize=quantize,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            hf_token=hf_token,
        )
