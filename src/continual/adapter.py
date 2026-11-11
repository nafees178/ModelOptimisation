import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


@dataclass
class LoRAConfigWrapper:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class ContinualAdaptationManager:
    """Implements parameter-efficient continual learning as described in
    'Continual Learning and Catastrophic Forgetting' (van de Ven et al.).

    Strategy:
    1. Freeze the quantized low-bit base model (100% immune to catastrophic forgetting).
    2. Attach lightweight low-rank adapters (~10MB) for sequential task or domain customization.
    3. Support dynamic adapter switching for multi-domain on-device personalization.
    """

    def __init__(self, config: Optional[LoRAConfigWrapper] = None):
        self.config = config or LoRAConfigWrapper()

    def prepare_for_continual_training(self, model: Any) -> PeftModel:
        """Prepares a quantized model for low-rank adapter training with gradient checkpointing."""
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.bias,
            task_type=TaskType.CAUSAL_LM,
        )

        peft_model = get_peft_model(model, peft_config)
        self.print_trainable_parameters(peft_model)
        return peft_model

    @staticmethod
    def print_trainable_parameters(model: Any) -> Dict[str, Any]:
        """Calculates trainable vs frozen parameters to prove minimal footprint."""
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        percentage = 100 * trainable_params / all_param
        stats = {
            "trainable_params": trainable_params,
            "all_params": all_param,
            "trainable_percentage": percentage,
            "adapter_size_mb": (trainable_params * 4) / (1024**2),  # FP32 weights in MB
        }
        print(
            f"[ContinualLearning] Trainable: {trainable_params:,} / {all_param:,} "
            f"({percentage:.3f}%) | Adapter Size: ~{stats['adapter_size_mb']:.2f} MB"
        )
        return stats

    @staticmethod
    def save_adapter(model: PeftModel, save_dir: str, adapter_name: str = "default"):
        """Saves only the tiny adapter weights (preserving disk space on edge hardware)."""
        target_path = os.path.join(save_dir, adapter_name)
        os.makedirs(target_path, exist_ok=True)
        model.save_pretrained(target_path)
        print(f"[ContinualLearning] Adapter '{adapter_name}' saved to {target_path}")

    @staticmethod
    def load_adapter(base_model: Any, adapter_path: str, adapter_name: str = "default") -> PeftModel:
        """Loads a specialized domain adapter onto the existing frozen base model."""
        print(f"[ContinualLearning] Loading adapter from {adapter_path}...")
        return PeftModel.from_pretrained(base_model, adapter_path, adapter_name=adapter_name)
