"""
Model Quantization Module.

This module provides utilities for quantizing LLM models to reduce VRAM usage
while maintaining acceptable performance. Supports 4-bit and 8-bit quantization.

Quantization Benefits:
- 4-bit: ~4x VRAM reduction (e.g., 6GB model -> 1.5GB)
- 8-bit: ~2x VRAM reduction (e.g., 6GB model -> 3GB)
- Trade-off: Slight quality degradation vs massive memory savings

All utilities follow zero-error philosophy with explicit validation.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class QuantizationType(Enum):
    """
    Supported quantization types.

    Attributes:
        NONE: No quantization (full precision)
        INT8: 8-bit integer quantization
        INT4: 4-bit integer quantization
        NF4: 4-bit NormalFloat quantization (bitsandbytes)
    """
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"


@dataclass
class QuantizationConfig:
    """
    Configuration for model quantization.

    Attributes:
        quantization_type: Type of quantization to use
        compute_dtype: Computation dtype (float16, float32, bfloat16)
        use_double_quant: Use double quantization for additional compression
        bnb_4bit_quant_type: Quantization type for 4-bit (nf4 or fp4)
        load_in_8bit: Load model in 8-bit mode
        load_in_4bit: Load model in 4-bit mode
    """
    quantization_type: QuantizationType = QuantizationType.NONE
    compute_dtype: str = "float16"
    use_double_quant: bool = False
    bnb_4bit_quant_type: str = "nf4"
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    def validate(self) -> bool:
        """
        Validate quantization configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        # Check conflicting settings
        if self.load_in_8bit and self.load_in_4bit:
            return False

        # Validate compute dtype
        valid_dtypes = ["float16", "float32", "bfloat16"]
        if self.compute_dtype not in valid_dtypes:
            return False

        # Validate 4-bit quant type
        if self.load_in_4bit:
            valid_quant_types = ["nf4", "fp4"]
            if self.bnb_4bit_quant_type not in valid_quant_types:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "quantization_type": self.quantization_type.value,
            "compute_dtype": self.compute_dtype,
            "use_double_quant": self.use_double_quant,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit
        }


class QuantizationEstimator:
    """
    Estimate VRAM requirements for different quantization levels.

    Helps determine optimal quantization strategy based on available VRAM.
    """

    @staticmethod
    def estimate_vram_gb(
        param_count_billion: float,
        quantization_type: QuantizationType
    ) -> float:
        """
        Estimate VRAM usage in GB for a given model size and quantization.

        Args:
            param_count_billion: Model parameter count in billions
            quantization_type: Type of quantization

        Returns:
            Estimated VRAM usage in GB
        """
        # Base formula: params (billions) * bytes_per_param
        # Add ~20% overhead for activation, KV cache, etc.

        if quantization_type == QuantizationType.NONE:
            # FP16: 2 bytes per parameter
            base_gb = param_count_billion * 2.0
        elif quantization_type == QuantizationType.INT8:
            # INT8: 1 byte per parameter
            base_gb = param_count_billion * 1.0
        elif quantization_type in [QuantizationType.INT4, QuantizationType.NF4]:
            # 4-bit: 0.5 bytes per parameter
            base_gb = param_count_billion * 0.5
        else:
            # Unknown: assume FP16
            base_gb = param_count_billion * 2.0

        # Add 20% overhead
        return base_gb * 1.2

    @staticmethod
    def recommend_quantization(
        param_count_billion: float,
        available_vram_gb: float
    ) -> Optional[QuantizationType]:
        """
        Recommend quantization type based on model size and available VRAM.

        Args:
            param_count_billion: Model parameter count in billions
            available_vram_gb: Available VRAM in GB

        Returns:
            Recommended QuantizationType, or None if model won't fit
        """
        # Try each quantization level from least to most aggressive
        for quant_type in [
            QuantizationType.NONE,
            QuantizationType.INT8,
            QuantizationType.NF4
        ]:
            required_vram = QuantizationEstimator.estimate_vram_gb(
                param_count_billion,
                quant_type
            )

            if required_vram <= available_vram_gb:
                return quant_type

        # Model won't fit even with 4-bit quantization
        return None

    @staticmethod
    def get_memory_savings(
        param_count_billion: float,
        from_quant: QuantizationType,
        to_quant: QuantizationType
    ) -> float:
        """
        Calculate memory savings when changing quantization.

        Args:
            param_count_billion: Model parameter count
            from_quant: Original quantization
            to_quant: Target quantization

        Returns:
            Memory savings in GB (positive = savings, negative = increase)
        """
        from_vram = QuantizationEstimator.estimate_vram_gb(
            param_count_billion,
            from_quant
        )
        to_vram = QuantizationEstimator.estimate_vram_gb(
            param_count_billion,
            to_quant
        )

        return from_vram - to_vram


def create_quantization_config_4bit() -> QuantizationConfig:
    """
    Create recommended 4-bit quantization configuration.

    Returns:
        QuantizationConfig for 4-bit quantization
    """
    return QuantizationConfig(
        quantization_type=QuantizationType.NF4,
        compute_dtype="float16",
        use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        load_in_8bit=False,
        load_in_4bit=True
    )


def create_quantization_config_8bit() -> QuantizationConfig:
    """
    Create recommended 8-bit quantization configuration.

    Returns:
        QuantizationConfig for 8-bit quantization
    """
    return QuantizationConfig(
        quantization_type=QuantizationType.INT8,
        compute_dtype="float16",
        use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        load_in_8bit=True,
        load_in_4bit=False
    )


def create_quantization_config_none() -> QuantizationConfig:
    """
    Create configuration for no quantization (full precision).

    Returns:
        QuantizationConfig for full precision
    """
    return QuantizationConfig(
        quantization_type=QuantizationType.NONE,
        compute_dtype="float16",
        use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        load_in_8bit=False,
        load_in_4bit=False
    )


def get_optimal_config(
    param_count_billion: float,
    available_vram_gb: float
) -> Optional[QuantizationConfig]:
    """
    Get optimal quantization configuration for given constraints.

    Args:
        param_count_billion: Model parameter count in billions
        available_vram_gb: Available VRAM in GB

    Returns:
        Optimal QuantizationConfig, or None if model won't fit
    """
    recommended_type = QuantizationEstimator.recommend_quantization(
        param_count_billion,
        available_vram_gb
    )

    if recommended_type is None:
        return None

    if recommended_type == QuantizationType.NF4:
        return create_quantization_config_4bit()
    elif recommended_type == QuantizationType.INT8:
        return create_quantization_config_8bit()
    else:
        return create_quantization_config_none()
