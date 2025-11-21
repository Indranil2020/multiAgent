"""
Model management module for LLM operations.

This module provides comprehensive model loading, pooling, and inference capabilities:

1. Model Loader (model_loader.py):
   - Support for multiple backends (Transformers, VLLM, TensorRT)
   - Device management (CPU, CUDA, MPS)
   - Quantization support (4-bit, 8-bit, FP16)
   - Model caching and preloading
   - Memory-efficient loading strategies

2. Model Pool (model_pool.py):
   - Shared model instances across multiple agents
   - VRAM optimization through model reuse
   - Priority-based model access
   - Automatic model eviction and loading
   - Pool statistics and monitoring
   - Concurrent access management

3. Inference Engine (inference_engine.py):
   - High-level inference API
   - Multiple sampling strategies
   - Batch generation support
   - Generation parameters and constraints
   - Streaming and non-streaming modes
   - Token-level control

4. Quantization (quantization.py):
   - Model quantization utilities
   - VRAM estimation for different quantization levels
   - Quality vs. memory trade-offs
   - Quantization configuration management

Key Concepts:
- **Model Pool Pattern**: Load ONE model instance, share across N agents
  - Without pool: N agents * model_size VRAM
  - With pool: 1 * model_size VRAM (saves (N-1) * model_size)
  - Example: 10 agents * 6GB = 60GB -> 6GB with pool (10x savings!)

- **Quantization**: Reduce precision to save VRAM
  - FP32: Full precision (baseline)
  - FP16: Half precision (~50% VRAM reduction, minimal quality loss)
  - INT8: 8-bit integers (~75% VRAM reduction, small quality loss)
  - INT4: 4-bit integers (~87.5% VRAM reduction, noticeable quality loss)

Usage:
    from llm.models import (
        ModelLoader,
        ModelPool,
        InferenceEngine,
        QuantizationType
    )

    # Load a model
    loader = ModelLoader()
    model = loader.load_model(
        "deepseek-coder-6.7b",
        quantization=QuantizationType.INT8
    )

    # Use model pool for shared access
    pool = ModelPool()
    pool.add_model("primary", model, priority=ModelPriority.HIGH)

    # Get model from pool
    with pool.acquire_model("primary") as model_instance:
        # Use model instance

    # High-level inference
    engine = InferenceEngine(model)
    result = engine.generate(
        prompt="def hello():",
        max_tokens=100,
        temperature=0.7
    )
"""

from llm.models.model_loader import (
    ModelBackend,
    DeviceType,
    ModelLoadConfig,
    LoadedModel,
    ModelLoader,
)

from llm.models.model_pool import (
    ModelPriority,
    ModelStatus,
    ModelPoolConfig,
    PoolStatistics,
    SharedModelInstance,
    ModelPool,
    PoolManager,
)

from llm.models.inference_engine import (
    SamplingStrategy,
    GenerationConfig,
    GenerationResult,
    BatchGenerationResult,
    InferenceEngine,
)

from llm.models.quantization import (
    QuantizationType,
    QuantizationConfig,
    QuantizationEstimator,
)


__all__ = [
    # Model loader
    "ModelBackend",
    "DeviceType",
    "ModelLoadConfig",
    "LoadedModel",
    "ModelLoader",

    # Model pool
    "ModelPriority",
    "ModelStatus",
    "ModelPoolConfig",
    "PoolStatistics",
    "SharedModelInstance",
    "ModelPool",
    "PoolManager",

    # Inference engine
    "SamplingStrategy",
    "GenerationConfig",
    "GenerationResult",
    "BatchGenerationResult",
    "InferenceEngine",

    # Quantization
    "QuantizationType",
    "QuantizationConfig",
    "QuantizationEstimator",
]
