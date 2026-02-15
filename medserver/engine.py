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


class SGLangEngine(BaseEngine):
    """SGLang implementation (Linux + Ampere+ GPUs for high performance)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine = None
        self._processor = None

    async def load(self) -> None:
        try:
            import sglang
            from transformers import AutoProcessor
        except ImportError:
            raise ImportError("sglang or transformers not installed. This engine requires Linux.")

        logger.info(f"Loading model with SGLang: {self.model_id}")
        start = time.monotonic()

        # Set env var for HF token if needed
        if self.hf_token:
            import os
            os.environ["HF_TOKEN"] = self.hf_token

        # Load Processor for chat template
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, token=self.hf_token, trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load processor for SGLang: {e}")
            raise

        # Initialize SGLang Engine
        # We explicitly set dtype="bfloat16" for Ampere+ compatibility and stability
        self._engine = sglang.Engine(
            model_path=self.model_id,
            max_model_len=self.max_model_len,
            mem_fraction_static=self.gpu_memory_utilization,
            trust_remote_code=True,
            tp_size=1,
            dtype="bfloat16",
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

        # SGLang formatting using official chat template
        final_prompt = prompt
        if isinstance(prompt, list):
             try:
                 final_prompt = self._processor.apply_chat_template(
                     prompt,
                     add_generation_prompt=True,
                     tokenize=False
                 )
             except Exception as e:
                 logger.error(f"SGLang apply_chat_template failed: {e}")
                 raise RuntimeError(f"Failed to format prompt with chat template: {e}")

        inputs = {"text": final_prompt}
        if images and self.supports_images:
             inputs["image_data"] = images

        sampling_params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "stop": ["<end_of_turn>", "<eos>", "<|endoftext|>"],
        }

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
        
        # Format messages for the official chat template
        if isinstance(prompt, list):
            formatted_messages = []
            image_idx = 0
            
            for msg in prompt:
                role = msg.get("role")
                content = msg.get("content", "")
                msg_content = []
                
                if isinstance(content, str):
                    msg_content.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            msg_content.append({"type": "text", "text": item.get("text", "")})
                        elif item.get("type") == "image" and images and image_idx < len(images):
                            msg_content.append({"type": "image", "image": images[image_idx]})
                            image_idx += 1
                
                formatted_messages.append({"role": role, "content": msg_content})

            try:
                # Apply official chat template
                if self.supports_images and self._processor:
                    inputs = self._processor.apply_chat_template(
                        formatted_messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt"
                    ).to(self._model.device)
                    
                    # Ensure pixel_values are correctly attached and typed
                    if "pixel_values" not in inputs and images:
                         pixel_values = self._processor(images=images, return_tensors="pt")["pixel_values"]
                         inputs["pixel_values"] = pixel_values.to(self._model.device)
                    
                    if "pixel_values" in inputs:
                         inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)
                elif self._tokenizer:
                    # Text-only path
                    final_prompt = self._tokenizer.apply_chat_template(
                        formatted_messages,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    inputs = self._tokenizer(final_prompt, return_tensors="pt").to(self._model.device)
            except Exception as e:
                logger.error(f"Transformers apply_chat_template failed: {e}")
                raise RuntimeError(f"Failed to format prompt with chat template: {e}")
        else:
            # Raw string prompt (rarely used via API but supported for internal tests)
            if self._tokenizer:
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            elif self._processor:
                inputs = self._processor(text=[prompt], return_tensors="pt").to(self._model.device)

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
            
            # Non-blocking wait for the thread to finish
            max_wait = 5.0  # seconds
            wait_start = time.monotonic()
            while thread.is_alive() and (time.monotonic() - wait_start < max_wait):
                await asyncio.sleep(0.1)
            
            if thread.is_alive():
                logger.warning(f"Generation thread did not terminate within {max_wait}s")


class HybridEngine(BaseEngine):
    """
    Engine that attempts to use SGLang for high performance, but automatically
    falls back to Transformers if SGLang fails to load.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine: Optional[BaseEngine] = None
        self._kwargs = kwargs

    async def load(self) -> None:
        use_sglang = False
        
        # Initial check for SGLang compatibility
        if platform.system() == "Linux":
            try:
                import sglang
                if torch.cuda.is_available():
                    major, _ = torch.cuda.get_device_capability()
                    if major >= 8:
                        use_sglang = True
            except ImportError:
                pass

        if use_sglang:
            try:
                logger.info("Attempting to load high-performance SGLang engine...")
                self._engine = SGLangEngine(**self._kwargs)
                await self._engine.load()
                self._loaded = True
                self._load_time = self._engine.load_time
                return
            except Exception as e:
                logger.warning(f"SGLang load failed, falling back to Transformers: {e}")
                self._engine = None

        # Fallback to Transformers
        logger.info("Loading universal Transformers engine...")
        self._engine = TransformersEngine(**self._kwargs)
        await self._engine.load()
        self._loaded = True
        self._load_time = self._engine.load_time

    async def stream_generate(self, *args, **kwargs) -> AsyncIterator[str]:
        if not self._engine:
            raise RuntimeError("Engine not loaded.")
        async for chunk in self._engine.stream_generate(*args, **kwargs):
            yield chunk


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
    """Factory: Returns a HybridEngine with automatic fallback capability."""
    return HybridEngine(
        model_id=model_id,
        supports_images=supports_images,
        quantize=quantize,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        hf_token=hf_token,
    )
