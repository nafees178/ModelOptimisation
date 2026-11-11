import gc
import os
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class QuantizationMode(str, Enum):
    FP16 = "baseline_fp16"
    INT8 = "int8_weight_dynamic"
    INT4_NF4 = "int4_nf4_double_quant"
    INT4_FP4 = "int4_fp4"


class ModelQuantizer:
    """Orchestrates model quantization modes and enforces protection of sensitive layers
    (LM Head and Embeddings) to prevent compression artifacts as documented in Hohman et al. (Apple CHI '24).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        preserve_layers: Optional[List[str]] = None,
        device_map: str = "auto",
    ):
        self.model_name = model_name
        self.preserve_layers = preserve_layers or ["lm_head", "model.embed_tokens"]
        self.device_map = device_map

    def get_bnb_config(self, mode: QuantizationMode) -> Optional[BitsAndBytesConfig]:
        if mode == QuantizationMode.FP16:
            return None

        elif mode == QuantizationMode.INT8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=self.preserve_layers,
            )

        elif mode == QuantizationMode.INT4_NF4:
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
                llm_int8_skip_modules=self.preserve_layers,
            )

        elif mode == QuantizationMode.INT4_FP4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_use_double_quant=False,
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_skip_modules=self.preserve_layers,
            )

        else:
            raise ValueError(f"Unknown quantization mode: {mode}")

    def load_model_and_tokenizer(
        self,
        mode: QuantizationMode,
    ) -> Tuple[Any, Any, float]:
        """Loads model under the selected quantization mode and records precise load time."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n[ModelQuantizer] Loading {self.model_name} in mode: {mode.value}...")
        start_time = time.perf_counter()

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        bnb_config = self.get_bnb_config(mode)

        if mode == QuantizationMode.FP16:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map=self.device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

        load_time_sec = time.perf_counter() - start_time
        print(f"[ModelQuantizer] Successfully loaded in {load_time_sec:.2f}s.")
        return model, tokenizer, load_time_sec

    @staticmethod
    def analyze_quantized_layers(model: Any) -> Dict[str, Any]:
        """Inspects module hierarchy to verify which layers were quantized vs preserved in full precision."""
        stats = {
            "linear8bit": 0,
            "linear4bit": 0,
            "linear_fp16": 0,
            "preserved_layers": [],
        }

        for name, module in model.named_modules():
            mod_type = type(module).__name__
            if "Linear8bitLt" in mod_type:
                stats["linear8bit"] += 1
            elif "Linear4bit" in mod_type:
                stats["linear4bit"] += 1
            elif isinstance(module, torch.nn.Linear):
                stats["linear_fp16"] += 1
                stats["preserved_layers"].append(name)

        return stats
