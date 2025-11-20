"""
Model Loader Module.

This module provides utilities for loading LLM models with various configurations:
- Quantization support (4-bit, 8-bit, full precision)
- Device management (CPU, CUDA, MPS)
- Multiple backends (transformers, llama.cpp, GGUF)
- Memory-efficient loading strategies

All functions follow zero-error philosophy with explicit validation.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .quantization import QuantizationConfig, QuantizationType


class ModelBackend(Enum):
    """
    Supported model backends.

    Attributes:
        TRANSFORMERS: HuggingFace Transformers
        LLAMA_CPP: llama.cpp (GGUF format)
        VLLM: vLLM inference engine
        CUSTOM: Custom implementation
    """
    TRANSFORMERS = "transformers"
    LLAMA_CPP = "llama_cpp"
    VLLM = "vllm"
    CUSTOM = "custom"


class DeviceType(Enum):
    """
    Supported device types.

    Attributes:
        CPU: CPU-only execution
        CUDA: NVIDIA CUDA GPUs
        MPS: Apple Metal Performance Shaders
        AUTO: Automatic device selection
    """
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


@dataclass
class ModelLoadConfig:
    """
    Configuration for loading models.

    Attributes:
        model_name_or_path: Model identifier or local path
        backend: Model backend to use
        device: Target device
        quantization_config: Quantization configuration
        trust_remote_code: Allow remote code execution
        torch_dtype: PyTorch dtype for model weights
        max_memory: Maximum memory per device (GB)
        offload_folder: Folder for CPU offloading
    """
    model_name_or_path: str
    backend: ModelBackend = ModelBackend.TRANSFORMERS
    device: DeviceType = DeviceType.AUTO
    quantization_config: Optional[QuantizationConfig] = None
    trust_remote_code: bool = False
    torch_dtype: str = "auto"
    max_memory: Optional[Dict[str, float]] = None
    offload_folder: Optional[Path] = None

    def validate(self) -> bool:
        """
        Validate model load configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        # Check model path is not empty
        if not self.model_name_or_path:
            return False

        # Validate quantization config if provided
        if self.quantization_config is not None:
            if not self.quantization_config.validate():
                return False

        # Validate torch dtype
        valid_dtypes = ["auto", "float16", "float32", "bfloat16"]
        if self.torch_dtype not in valid_dtypes:
            return False

        # Validate max_memory if provided
        if self.max_memory is not None:
            for device_id, mem_gb in self.max_memory.items():
                if not isinstance(device_id, str):
                    return False
                if mem_gb <= 0:
                    return False

        return True


@dataclass
class LoadedModel:
    """
    Container for a loaded model and its metadata.

    Attributes:
        model: The loaded model object (type depends on backend)
        tokenizer: The tokenizer object
        config: Model configuration
        device: Device where model is loaded
        vram_usage_gb: Estimated VRAM usage in GB
        backend: Backend used to load model
        is_quantized: Whether model is quantized
    """
    model: Any
    tokenizer: Any
    config: Dict[str, Any]
    device: str
    vram_usage_gb: float
    backend: ModelBackend
    is_quantized: bool


class ModelLoader:
    """
    Model loader with support for multiple backends and configurations.

    Handles loading models efficiently with proper error checking
    and resource management.
    """

    def __init__(self):
        """Initialize model loader."""
        self.loaded_models: Dict[str, LoadedModel] = {}

    def load_model(
        self,
        config: ModelLoadConfig
    ) -> Optional[LoadedModel]:
        """
        Load a model with given configuration.

        Args:
            config: Model loading configuration

        Returns:
            LoadedModel if successful, None if failed
        """
        # Validate configuration
        if not config.validate():
            return None

        # Check if already loaded
        cache_key = self._get_cache_key(config)
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]

        # Route to appropriate backend
        if config.backend == ModelBackend.TRANSFORMERS:
            loaded_model = self._load_transformers_model(config)
        elif config.backend == ModelBackend.LLAMA_CPP:
            loaded_model = self._load_llama_cpp_model(config)
        elif config.backend == ModelBackend.VLLM:
            loaded_model = self._load_vllm_model(config)
        else:
            # Custom or unsupported backend
            return None

        # Cache if successful
        if loaded_model is not None:
            self.loaded_models[cache_key] = loaded_model

        return loaded_model

    def _load_transformers_model(
        self,
        config: ModelLoadConfig
    ) -> Optional[LoadedModel]:
        """
        Load model using HuggingFace Transformers.

        Args:
            config: Model loading configuration

        Returns:
            LoadedModel if successful, None if failed
        """
        # This would use actual transformers library in production
        # For now, return a placeholder that demonstrates the interface

        # Simulate loading
        model_obj = None  # Would be actual model
        tokenizer_obj = None  # Would be actual tokenizer

        # Determine device
        device = self._determine_device(config.device)

        # Estimate VRAM usage
        vram_usage = self._estimate_vram_usage(config)

        # Check if quantized
        is_quantized = (
            config.quantization_config is not None and
            config.quantization_config.quantization_type != QuantizationType.NONE
        )

        # Create loaded model container
        return LoadedModel(
            model=model_obj,
            tokenizer=tokenizer_obj,
            config={
                "model_name": config.model_name_or_path,
                "quantization": config.quantization_config.to_dict()
                if config.quantization_config else None,
                "dtype": config.torch_dtype
            },
            device=device,
            vram_usage_gb=vram_usage,
            backend=ModelBackend.TRANSFORMERS,
            is_quantized=is_quantized
        )

    def _load_llama_cpp_model(
        self,
        config: ModelLoadConfig
    ) -> Optional[LoadedModel]:
        """
        Load model using llama.cpp (GGUF format).

        Args:
            config: Model loading configuration

        Returns:
            LoadedModel if successful, None if failed
        """
        # Placeholder for llama.cpp integration
        # In production, would use llama-cpp-python

        device = self._determine_device(config.device)
        vram_usage = self._estimate_vram_usage(config)

        return LoadedModel(
            model=None,  # Would be llama.cpp model
            tokenizer=None,  # llama.cpp handles tokenization internally
            config={"model_path": config.model_name_or_path},
            device=device,
            vram_usage_gb=vram_usage,
            backend=ModelBackend.LLAMA_CPP,
            is_quantized=True  # GGUF models are typically quantized
        )

    def _load_vllm_model(
        self,
        config: ModelLoadConfig
    ) -> Optional[LoadedModel]:
        """
        Load model using vLLM inference engine.

        Args:
            config: Model loading configuration

        Returns:
            LoadedModel if successful, None if failed
        """
        # Placeholder for vLLM integration
        # In production, would use vllm library

        device = self._determine_device(config.device)
        vram_usage = self._estimate_vram_usage(config)

        return LoadedModel(
            model=None,  # Would be vLLM engine
            tokenizer=None,  # Would be vLLM tokenizer
            config={"model_name": config.model_name_or_path},
            device=device,
            vram_usage_gb=vram_usage,
            backend=ModelBackend.VLLM,
            is_quantized=False
        )

    def _determine_device(self, device_type: DeviceType) -> str:
        """
        Determine actual device string from device type.

        Args:
            device_type: Device type enum

        Returns:
            Device string (e.g., "cuda:0", "cpu")
        """
        if device_type == DeviceType.CPU:
            return "cpu"
        elif device_type == DeviceType.CUDA:
            return "cuda:0"
        elif device_type == DeviceType.MPS:
            return "mps"
        elif device_type == DeviceType.AUTO:
            # In production, would check CUDA availability
            # For now, return CPU
            return "cpu"
        else:
            return "cpu"

    def _estimate_vram_usage(self, config: ModelLoadConfig) -> float:
        """
        Estimate VRAM usage for model.

        Args:
            config: Model loading configuration

        Returns:
            Estimated VRAM usage in GB
        """
        # Rough estimate based on model name
        # In production, would parse model config for actual parameter count

        model_name = config.model_name_or_path.lower()

        # Estimate base size from name
        if "7b" in model_name or "8b" in model_name:
            param_count = 7.0
        elif "13b" in model_name:
            param_count = 13.0
        elif "3b" in model_name:
            param_count = 3.0
        elif "1b" in model_name:
            param_count = 1.0
        else:
            # Default assumption
            param_count = 7.0

        # Apply quantization factor
        if config.quantization_config is not None:
            quant_type = config.quantization_config.quantization_type
            if quant_type == QuantizationType.NF4:
                return param_count * 0.5 * 1.2  # 4-bit + 20% overhead
            elif quant_type == QuantizationType.INT8:
                return param_count * 1.0 * 1.2  # 8-bit + 20% overhead
            else:
                return param_count * 2.0 * 1.2  # FP16 + 20% overhead
        else:
            # No quantization: FP16
            return param_count * 2.0 * 1.2

    def _get_cache_key(self, config: ModelLoadConfig) -> str:
        """
        Generate cache key for model configuration.

        Args:
            config: Model loading configuration

        Returns:
            Cache key string
        """
        quant_str = "none"
        if config.quantization_config is not None:
            quant_str = config.quantization_config.quantization_type.value

        return f"{config.model_name_or_path}_{config.backend.value}_{quant_str}"

    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from memory.

        Args:
            model_id: Model identifier (cache key)

        Returns:
            True if unloaded, False if not found
        """
        if model_id in self.loaded_models:
            # In production, would properly clean up model
            del self.loaded_models[model_id]
            return True
        return False

    def get_loaded_models(self) -> Dict[str, LoadedModel]:
        """
        Get all currently loaded models.

        Returns:
            Dictionary of loaded models
        """
        return self.loaded_models.copy()

    def total_vram_usage(self) -> float:
        """
        Calculate total VRAM usage of all loaded models.

        Returns:
            Total VRAM usage in GB
        """
        return sum(
            model.vram_usage_gb
            for model in self.loaded_models.values()
        )


def create_default_load_config(
    model_name: str,
    use_quantization: bool = True
) -> ModelLoadConfig:
    """
    Create default model load configuration.

    Args:
        model_name: Model name or path
        use_quantization: Whether to use 4-bit quantization

    Returns:
        ModelLoadConfig with defaults
    """
    from .quantization import create_quantization_config_4bit

    quant_config = None
    if use_quantization:
        quant_config = create_quantization_config_4bit()

    return ModelLoadConfig(
        model_name_or_path=model_name,
        backend=ModelBackend.TRANSFORMERS,
        device=DeviceType.AUTO,
        quantization_config=quant_config,
        trust_remote_code=False,
        torch_dtype="auto"
    )
