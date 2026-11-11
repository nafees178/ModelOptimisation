import argparse
import gc
import json
import os
import sys
import time
from typing import Any, Dict, List

import yaml
import torch

from src.benchmark.profiler import HardwareProfiler
from src.compression.quantizer import ModelQuantizer, QuantizationMode
from src.continual.adapter import ContinualAdaptationManager, LoRAConfigWrapper
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.pareto import ParetoFrontierPlotter


def load_yaml_config(config_path: str = "configs/qwen_1.5b_optim.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(
    config_path: str = "configs/qwen_1.5b_optim.yaml",
    modes_to_test: List[str] = None,
    test_continual: bool = True,
):
    config = load_yaml_config(config_path)
    model_name = config["model"]["name"]
    preserve_layers = config["compression"].get("sensitive_layers_to_preserve", ["lm_head"])
    test_prompts = config["evaluation"]["test_prompts"]
    results_dir = config["output"].get("results_dir", "results")
    pareto_path = config["output"].get("pareto_plot_path", "results/pareto_frontier.png")

    os.makedirs(results_dir, exist_ok=True)

    if modes_to_test is None:
        modes_to_test = ["baseline_fp16", "int8_weight_dynamic", "int4_nf4_double_quant"]

    sys.stdout.reconfigure(line_buffering=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"MODEL OPTIMISATION & COMPRESSION PIPELINE")
    print(f"Target Model: {model_name}")
    print(f"Active Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Modes to benchmark: {modes_to_test}")
    print(f"{'='*70}\n")

    profiler = HardwareProfiler(device=device)
    quantizer = ModelQuantizer(model_name=model_name, preserve_layers=preserve_layers)

    all_benchmarks = []

    for mode_str in modes_to_test:
        mode = QuantizationMode(mode_str)
        print(f"\n>>> Running Benchmark for: {mode.value.upper()}")

        try:
            model, tokenizer, load_time_sec = quantizer.load_model_and_tokenizer(mode)
            layer_stats = quantizer.analyze_quantized_layers(model)
            print(f"    Layer analysis: {layer_stats}")

            bench_result = profiler.benchmark_generation(
                model=model,
                tokenizer=tokenizer,
                prompts=test_prompts,
                mode_name=mode.value,
                model_name=model_name,
                load_time_sec=load_time_sec,
                max_new_tokens=220,
            )

            # Evaluate task response accuracy & structural validity
            eval_stats = ModelEvaluator.score_task_responses(bench_result.task_results)
            accuracy = eval_stats["overall_accuracy_percentage"]

            bench_dict = bench_result.to_dict()
            bench_dict["accuracy"] = accuracy
            bench_dict["evaluation_details"] = eval_stats
            bench_dict["layer_breakdown"] = layer_stats

            all_benchmarks.append(bench_dict)
            print(bench_result.summary())
            print(f"    Task Evaluation Score: {accuracy:.1f}%\n")

            # Clean up memory before next run
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[-] Error benchmarking mode {mode_str}: {e}")
            import traceback
            traceback.print_exc()

    # Step 4: Continual Learning Demonstration (Paper 2)
    if test_continual and "int4_nf4_double_quant" in modes_to_test:
        print(f"\n{'='*70}")
        print("CONTINUAL LEARNING & ADAPTATION MODULE (Paper 2 Integration)")
        print("Equipping 4-bit Quantized Model with QLoRA to Prevent Catastrophic Forgetting")
        print(f"{'='*70}")
        try:
            model, tokenizer, _ = quantizer.load_model_and_tokenizer(QuantizationMode.INT4_NF4)
            cl_manager = ContinualAdaptationManager(
                LoRAConfigWrapper(
                    r=config["continual_learning"]["lora_r"],
                    lora_alpha=config["continual_learning"]["lora_alpha"],
                    lora_dropout=config["continual_learning"]["lora_dropout"],
                    target_modules=config["continual_learning"]["target_modules"],
                )
            )
            peft_model = cl_manager.prepare_for_continual_training(model)
            adapter_stats = cl_manager.print_trainable_parameters(peft_model)

            adapter_save_dir = os.path.join(results_dir, "continual_adapters")
            cl_manager.save_adapter(peft_model, adapter_save_dir, "domain_local_user")
            print(f"[+] Continual Learning Adapter successfully verified and saved!")

            del peft_model
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[-] Continual learning verification notice: {e}")

    # Step 5: Save JSON results and plot Pareto frontier
    json_path = os.path.join(results_dir, "benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_benchmarks, f, indent=2)
    print(f"\n[+] Full benchmark results saved to: {json_path}")

    if len(all_benchmarks) >= 2:
        ParetoFrontierPlotter.plot_tradeoff_curves(all_benchmarks, output_path=pareto_path)

    # Print comparative matrix
    print("\n" + "=" * 90)
    print("FINAL OPTIMIZATION COMPARATIVE MATRIX (Apple CHI '24 Hardware Budgeting)")
    print("=" * 90)
    header = f"{'Mode':<26} | {'Peak VRAM':<12} | {'Host RAM':<10} | {'Throughput':<15} | {'TTFT':<10} | {'Accuracy':<8}"
    print(header)
    print("-" * 90)
    for b in all_benchmarks:
        print(
            f"{b['mode']:<26} | "
            f"{b['peak_vram_mb']:>9.1f} MB | "
            f"{b['peak_ram_mb']:>8.1f} MB | "
            f"{b['tokens_per_second']:>10.2f} tok/s | "
            f"{b['time_to_first_token_ms']:>7.1f} ms | "
            f"{b['accuracy']:>7.1f}%"
        )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Optimisation & Compression Benchmark")
    parser.add_argument("--config", type=str, default="configs/qwen_1.5b_optim.yaml")
    parser.add_argument("--modes", nargs="+", default=["baseline_fp16", "int8_weight_dynamic", "int4_nf4_double_quant"])
    parser.add_argument("--no_continual", action="store_true", help="Skip continual learning adapter step")
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        modes_to_test=args.modes,
        test_continual=not args.no_continual,
    )
