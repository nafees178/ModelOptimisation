 # ⚡ On-Device LLM Optimization & Continual Adaptation Toolkit

  An end-to-end model compression and on-device adaptation framework designed to optimize Large Language Models (LLMs) for low-spec edge devices, laptops, and consumer GPUs without catastrophic accuracy degradation.

  Built upon the empirical methodologies established in:
  1. **Apple (CHI '24)**: *"Model Compression in Practice: Lessons Learned from Practitioners Creating On-device Machine Learning Experiences"* (Fred Hohman et al.)
  2. **Continual Learning & Catastrophic Forgetting** (Gido M. van de Ven, Nicholas Soures, Dhireesha Kudithipudi)

  ### Repository contents

  | Folder | What it is |
  |---|---|
  | `src/`, `run_optimization_pipeline.py`, `demo/` | Qwen2.5-1.5B optimization and continual-adaptation toolkit (RTX 4060, described below) |
  | [`llama-6gb-compression/`](llama-6gb-compression/) | **New:** Llama-3.2 1B/3B compression study on a 6 GB RTX 4050: 30 variants of static, dynamic and hybrid quantization, pruning, and "bigger-then-compress" ([report PDF](llama-6gb-compression/RESULTS.pdf)) |

  ---

  ## 🧪 Llama-3.2 Compression Study (RTX 4050, 6 GB)

  Every major compression family that runs on a 6 GB laptop GPU, measured for memory, WikiText-2 perplexity, six-task zero-shot accuracy and decode speed. Full details are in [`llama-6gb-compression/`](llama-6gb-compression/).

  | Variant | Weights | PPL ↓ | Avg acc ↑ | tok/s |
  | :--- | :---: | :---: | :---: | :---: |
  | Llama-3.2-1B original (bf16) | 2.47 GB | 13.2 | 60.8% | 69 |
  | 1B int8 weight-only + `torch.compile` | 1.50 GB | 13.2 | 60.8% | 95 |
  | 1B **hybrid** static+dynamic int8 | 1.50 GB | 13.4 | 60.2% | 8* |
  | 1B all-**static** int8 | 1.50 GB | 391.7 ❌ | 36.3% | 9* |
  | 1B hybrid h25 (sensitivity-guided int8 + int4) | 1.18 GB | 14.1 | 59.3% | 27* |
  | 1B int4-HQQ + `torch.compile` | 1.07 GB | 15.0 | 59.0% | 123 |
  | 1B drop 4 layers + LoRA heal + int4 | 0.91 GB | 40.2 | 49.8% | 146 |
  | **Llama-3.2-3B int4-HQQ** | **2.37 GB** | **11.7** | **66.7%** | 60 |

  \* eager mode

  > **Key Findings:**
  > 1. **Bigger, then compress:** Llama-3B in int4 fits the same memory as the uncompressed 1B and beats it by 6 accuracy points.
  > 2. **Static vs. dynamic per layer:** the MLP `down_proj` inputs carry activation outliers up to 3,400× RMS. All-static int8 collapses; making only those layers dynamic keeps full quality.
  > 3. **Pruning is costly on small models:** Wanda 50% costs +140% perplexity at 1B. Only gentle 30% pruning keeps accuracy.
  > 4. **Batch-1 decoding is bandwidth-bound:** only smaller weights make single-user generation faster. int8 dynamic and 2:4 sparse kernels are slower at batch 1.

  ---

  ## 🎯 Key Objectives & Hardware Budgeting

  Following **Strategy #1 & #2** from the Apple CHI '24 research paper, optimization must be rooted in physical hardware metrics (VRAM, Host RAM pressure, Time-To-First-Token, and Thermal/Power limits):

  - **Target Model:** `Qwen/Qwen2.5-1.5B-Instruct` (1.54B parameters)
  - **Deployment Target:** Low-spec laptops & edge devices (target: `< 2.0 GB` RAM/VRAM footprint)
  - **Primary KPI:** Minimal memory footprint, followed by maximum inference speed, with zero perceptual collapse.

  ---

  ## 📊 Empirical Hardware Benchmarks (NVIDIA RTX 4060 8GB GPU)

  | Optimization Mode | Weight Footprint | Peak VRAM | Host RAM | Throughput | TTFT | Task Accuracy |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | **Baseline (FP16)** | 2,944.4 MB | 2,968.9 MB | 1,368.4 MB | 7.39 tok/s | 202.6 ms | **100.0%** |
  | **INT8 Dynamic (8-bit)** | 1,694.9 MB | 1,725.9 MB | 1,437.8 MB | 2.02 tok/s | 748.2 ms | **100.0%** |
  | **INT4 NF4 Double Quant** | **1,070.2 MB** | **1,125.3 MB** | **899.1 MB** | **4.46 tok/s** | **338.7 ms** | **75.0%** |

  > **Key Findings:**
  > 1. **63.6% Memory Reduction:** The 4-bit NormalFloat (NF4) mode shrinks the runtime footprint from ~3.0 GB down to **1,070 MB**, comfortably running on machines with as little as 2 GB of available RAM.
  > 2. **Protection Against Compression Artifacts (Strategy #6):** As revealed by Apple engineers, quantizing the final prediction projection causes output collapse and bounding jitter. In our architecture, the **`lm_head`** is explicitly excluded from quantization and preserved in FP16, guaranteeing stable token distributions.
  > 3. **The INT8 Outlier Penalty:** Dynamic INT8 with outlier thresholding incurs significant runtime dequantization overhead on consumer GPUs, making 4-bit NF4 double-quantization both smaller *and* faster in throughput.

  ---

  ## 🧠 Continual Learning & Catastrophic Forgetting Mitigation

  Deploying a model locally often requires on-device personalization to private user data. However, fine-tuning a small base model directly causes **catastrophic interference**, wiping out general knowledge.
Deploying a model locally often requires on-device personalization to private user data. However, fine-tuning a small base model directly causes **catastrophic interference**, wiping out general knowledge.

To solve this, we implemented **Context-Dependent Parameter Isolation (QLoRA)** as detailed in Section 3.5 of *Continual Learning and Catastrophic Forgetting*:
- **Base Model (INT4):** 100% frozen in memory (~1.0 GB). Completely immune to catastrophic forgetting.
- **Trainable LoRA Adapters:** Attached to all attention (`q, k, v, o`) and MLP (`gate, up, down`) projection layers.
- **Trainable Parameters:** Only **18,464,768 / 907,081,216 (2.036%)** parameters.
- **Adapter Weight Footprint:** **~70.4 MB**, allowing thousands of user-specific tasks or domain adapters to be swapped dynamically on disk without duplicating the base model.

---

## 🚀 Quick Start

### 1. Installation
```powershell
pip install -r requirements.txt
```

### 2. Run the Optimization & Benchmarking Pipeline
Runs automated profiling across all precision modes, tests task accuracy, exports continual learning adapters, and generates the Pareto frontier plot:
```powershell
python run_optimization_pipeline.py --modes baseline_fp16 int8_weight_dynamic int4_nf4_double_quant
```

### 3. Launch the Interactive "Feel" Demo (Apple Strategy #7)
Launch the Gradio playground to chat with different precision models side-by-side and inspect live hardware consumption:
```powershell
python demo/app.py
```

---

## 📈 Pareto Frontier Analysis
