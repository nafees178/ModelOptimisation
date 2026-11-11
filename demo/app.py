import gc
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path regardless of launch working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import torch

from src.benchmark.profiler import HardwareProfiler
from src.compression.quantizer import ModelQuantizer, QuantizationMode

# Global cached model and tokenizer
CURRENT_MODEL = None
CURRENT_TOKENIZER = None
CURRENT_MODE = None
QUANTIZER = ModelQuantizer(model_name="Qwen/Qwen2.5-1.5B-Instruct")
PROFILER = HardwareProfiler()


def switch_model(mode_str: str):
    global CURRENT_MODEL, CURRENT_TOKENIZER, CURRENT_MODE

    if CURRENT_MODE == mode_str and CURRENT_MODEL is not None:
        return f"Already loaded mode: {mode_str}"

    status = f"Switching to {mode_str}..."
    del CURRENT_MODEL
    del CURRENT_TOKENIZER
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mode = QuantizationMode(mode_str)
    CURRENT_MODEL, CURRENT_TOKENIZER, load_t = QUANTIZER.load_model_and_tokenizer(mode)
    CURRENT_MODE = mode_str
    mem = PROFILER.get_current_memory_mb()
    return f"Loaded {mode_str} in {load_t:.2f}s | VRAM: {mem['vram_alloc_mb']:.1f} MB | Host RAM: {mem['ram_mb']:.1f} MB"


def generate_text(prompt: str, max_tokens: int, temperature: float, top_p: float):
    global CURRENT_MODEL, CURRENT_TOKENIZER, CURRENT_MODE

    if CURRENT_MODEL is None:
        switch_model("int4_nf4_double_quant")

    if not prompt.strip():
        return "", "Empty prompt"

    start_mem = PROFILER.get_current_memory_mb()
    messages = [{"role": "user", "content": prompt}]
    formatted = CURRENT_TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = CURRENT_TOKENIZER(formatted, return_tensors="pt")
    device = next(CURRENT_MODEL.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_t = time.perf_counter()

    with torch.inference_mode():
        outputs = CURRENT_MODEL.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-4),
            top_p=top_p,
            pad_token_id=CURRENT_TOKENIZER.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_t

    gen_ids = outputs[0][in_len:]
    num_tokens = len(gen_ids)
    response = CURRENT_TOKENIZER.decode(gen_ids, skip_special_tokens=True)
    end_mem = PROFILER.get_current_memory_mb()

    tps = num_tokens / elapsed if elapsed > 0 else 0
    stats = (
        f"**Mode:** `{CURRENT_MODE}`\n\n"
        f"• **Tokens Generated:** {num_tokens}\n"
        f"• **Time Elapsed:** {elapsed:.2f}s\n"
        f"• **Throughput:** **{tps:.1f} tokens/sec**\n"
        f"• **Peak VRAM:** {end_mem['vram_peak_mb']:.1f} MB\n"
        f"• **Active VRAM:** {end_mem['vram_alloc_mb']:.1f} MB\n"
        f"• **Host RAM:** {end_mem['ram_mb']:.1f} MB"
    )
    return response, stats


def build_app():
    specs = HardwareProfiler.get_hardware_specs()
    hw_info = (
        f"**Hardware Platform:** {specs.platform} | **CPU:** {specs.processor} ({specs.cpu_count_physical} Cores) | "
        f"**System RAM:** {specs.system_ram_gb:.1f} GB | **GPU:** {specs.gpu_name or 'None'} ({specs.gpu_vram_gb:.1f} GB VRAM)"
    )

    with gr.Blocks(title="On-Device LLM Optimization Playground") as demo:
        gr.Markdown(
            "# ⚡ On-Device Model Optimization & Compression Playground\n"
            "Testing real-time 'feel', latency, and memory footprint based on **Apple CHI '24** and **Continual Learning** frameworks."
        )
        gr.Markdown(hw_info)

        with gr.Tab("💬 Live Inference & Hardware Profiling"):
            with gr.Row():
                with gr.Column(scale=1):
                    mode_radio = gr.Radio(
                        choices=["baseline_fp16", "int8_weight_dynamic", "int4_nf4_double_quant"],
                        value="int4_nf4_double_quant",
                        label="Select Compression Mode",
                        info="Switch precision to inspect live memory and speed trade-offs.",
                    )
                    load_btn = gr.Button("Load / Switch Model", variant="secondary")
                    load_status = gr.Markdown("Status: Ready to load")

                    max_tok_slider = gr.Slider(16, 512, value=128, step=16, label="Max New Tokens")
                    temp_slider = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="Temperature (0 = Deterministic)")
                    top_p_slider = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P")

                with gr.Column(scale=2):
                    prompt_input = gr.Textbox(
                        lines=3,
                        label="User Prompt",
                        placeholder="Enter your query or choose an example below...",
                        value="Explain how 4-bit quantization reduces memory footprint without severe accuracy loss.",
                    )
                    submit_btn = gr.Button("Generate Inference", variant="primary")
                    output_text = gr.Textbox(lines=8, label="Generated Output")
                    metrics_md = gr.Markdown("Metrics will appear here after generation.")

            gr.Examples(
                examples=[
                    ["A train travels from A to B at 60 mph and returns at 40 mph. What is its average speed for the round trip?", 128, 0.0, 0.9],
                    ["Write a fast Python decorator to measure function execution time.", 150, 0.0, 0.9],
                    ["Generate a JSON summary of on-device LLM trade-offs.", 100, 0.0, 0.9],
                ],
                inputs=[prompt_input, max_tok_slider, temp_slider, top_p_slider],
            )

            load_btn.click(fn=switch_model, inputs=[mode_radio], outputs=[load_status])
            submit_btn.click(
                fn=generate_text,
                inputs=[prompt_input, max_tok_slider, temp_slider, top_p_slider],
                outputs=[output_text, metrics_md],
            )

        with gr.Tab("📊 Pareto Frontier Analysis"):
            gr.Markdown(
                "### Multi-Metric Trade-off (Accuracy vs. Latency vs. Memory)\n"
                "As demonstrated in Apple CHI '24 (Section 5.6.1), the optimal on-device deployment is found at the **'knee of the curve'**."
            )
            img_path = str(PROJECT_ROOT / "results" / "pareto_frontier.png")
            pareto_img = gr.Image(img_path if os.path.exists(img_path) else None, label="Empirical Pareto Trade-off Curve")

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False)
