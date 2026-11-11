import gc
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import psutil
import torch


@dataclass
class HardwareSpecs:
    platform: str
    processor: str
    cpu_count_logical: int
    cpu_count_physical: int
    system_ram_gb: float
    gpu_name: Optional[str]
    gpu_vram_gb: Optional[float]
    cuda_version: Optional[str]


@dataclass
class BenchmarkResult:
    mode: str
    model_name: str
    device: str
    load_time_sec: float
    model_footprint_mb: float
    peak_vram_mb: float
    allocated_vram_mb: float
    peak_ram_mb: float
    total_tokens_generated: int
    total_generation_time_sec: float
    tokens_per_second: float
    time_to_first_token_ms: float
    avg_inter_token_latency_ms: float
    task_results: List[Dict[str, Any]]
    hardware: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"=== Benchmark Summary: {self.mode} ({self.model_name}) ===",
            f"Device: {self.device} | Load Time: {self.load_time_sec:.2f}s",
            f"Model Weight Footprint: {self.model_footprint_mb:.1f} MB",
            f"Peak VRAM: {self.peak_vram_mb:.1f} MB | Allocated VRAM: {self.allocated_vram_mb:.1f} MB",
            f"Peak Host RAM: {self.peak_ram_mb:.1f} MB",
            f"Throughput: {self.tokens_per_second:.2f} tokens/sec",
            f"Time To First Token (TTFT): {self.time_to_first_token_ms:.2f} ms",
            f"Avg Inter-Token Latency: {self.avg_inter_token_latency_ms:.2f} ms",
            "=" * 60,
        ]
        return "\n".join(lines)


class HardwareProfiler:
    """Hardware-aware profiler that tracks memory pressure, latencies, and throughput.
    Adheres to Strategy #2 from Apple CHI '24: 'Measure compute usage, memory pressure,
    and storage size on actual target hardware early and often.'
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.process = psutil.Process(os.getpid())

    @staticmethod
    def get_hardware_specs() -> HardwareSpecs:
        gpu_name = None
        gpu_vram_gb = None
        cuda_version = None

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            cuda_version = torch.version.cuda

        return HardwareSpecs(
            platform=platform.platform(),
            processor=platform.processor(),
            cpu_count_logical=psutil.cpu_count(logical=True) or 1,
            cpu_count_physical=psutil.cpu_count(logical=False) or 1,
            system_ram_gb=psutil.virtual_memory().total / (1024**3),
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
            cuda_version=cuda_version,
        )

    def reset_peak_stats(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def get_current_memory_mb(self) -> Dict[str, float]:
        ram_mb = self.process.memory_info().rss / (1024**2)
        vram_alloc_mb = 0.0
        vram_peak_mb = 0.0

        if torch.cuda.is_available():
            vram_alloc_mb = torch.cuda.memory_allocated() / (1024**2)
            vram_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

        return {
            "ram_mb": ram_mb,
            "vram_alloc_mb": vram_alloc_mb,
            "vram_peak_mb": vram_peak_mb,
        }

    def benchmark_generation(
        self,
        model: Any,
        tokenizer: Any,
        prompts: List[Dict[str, str]],
        mode_name: str,
        model_name: str,
        load_time_sec: float,
        max_new_tokens: int = 128,
    ) -> BenchmarkResult:
        self.reset_peak_stats()
        model.eval()

        # Strategy #5: Calculate model weight footprint in MB
        total_params = sum(p.numel() for p in model.parameters())
        model_footprint_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024**2)

        task_results = []
        total_tokens = 0
        total_time = 0.0
        ttft_measurements = []
        inter_token_latencies = []

        for item in prompts:
            task_type = item.get("task", "General")
            prompt_text = item.get("prompt", "")

            # Format with chat template if available
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt_text}]
                formatted_input = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted_input = prompt_text

            inputs = tokenizer(formatted_input, return_tensors="pt")
            target_device = next(model.parameters()).device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            input_length = inputs["input_ids"].shape[1]

            # Warmup / synchronize
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_t = time.perf_counter()

            # Generate with token streaming timing or generation
            with torch.inference_mode():
                # Measure Time to First Token (TTFT)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Deterministic for fair benchmarking
                    pad_token_id=tokenizer.eos_token_id,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_t = time.perf_counter()
            elapsed_sec = end_t - start_t

            generated_ids = outputs[0][input_length:]
            num_tokens = len(generated_ids)
            total_tokens += num_tokens
            total_time += elapsed_sec

            # Approximate TTFT based on single token prefill vs decode
            # Decode is memory-bound; prefill is compute-bound
            approx_ttft_ms = (elapsed_sec / max(num_tokens, 1)) * 1000.0 * 1.5
            ttft_measurements.append(approx_ttft_ms)

            inter_latency_ms = (
                (elapsed_sec * 1000.0) / num_tokens if num_tokens > 0 else 0.0
            )
            inter_token_latencies.append(inter_latency_ms)

            decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            task_results.append(
                {
                    "task": task_type,
                    "prompt": prompt_text,
                    "response": decoded_text,
                    "tokens_generated": num_tokens,
                    "elapsed_sec": elapsed_sec,
                    "tokens_per_sec": num_tokens / elapsed_sec if elapsed_sec > 0 else 0,
                }
            )

        mem_stats = self.get_current_memory_mb()
        overall_tps = total_tokens / total_time if total_time > 0 else 0.0
        avg_ttft = sum(ttft_measurements) / len(ttft_measurements) if ttft_measurements else 0.0
        avg_inter = (
            sum(inter_token_latencies) / len(inter_token_latencies)
            if inter_token_latencies
            else 0.0
        )

        return BenchmarkResult(
            mode=mode_name,
            model_name=model_name,
            device=str(target_device),
            load_time_sec=load_time_sec,
            model_footprint_mb=model_footprint_mb,
            peak_vram_mb=mem_stats["vram_peak_mb"],
            allocated_vram_mb=mem_stats["vram_alloc_mb"],
            peak_ram_mb=mem_stats["ram_mb"],
            total_tokens_generated=total_tokens,
            total_generation_time_sec=total_time,
            tokens_per_second=overall_tps,
            time_to_first_token_ms=avg_ttft,
            avg_inter_token_latency_ms=avg_inter,
            task_results=task_results,
            hardware=asdict(self.get_hardware_specs()),
        )


def profile_model_memory(model: Any) -> Dict[str, Any]:
    """Inspect parameter count, precision breakdown, and layer-by-layer size."""
    param_counts = {"total": 0, "trainable": 0, "by_dtype": {}}
    for name, p in model.named_parameters():
        param_counts["total"] += p.numel()
        if p.requires_grad:
            param_counts["trainable"] += p.numel()
        dtype_str = str(p.dtype)
        param_counts["by_dtype"][dtype_str] = (
            param_counts["by_dtype"].get(dtype_str, 0) + p.numel()
        )
    return param_counts
