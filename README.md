# Nemotron-3-Nano Medical LoRA Fine-tuning

Fine-tuning NVIDIA's Nemotron-3-Nano-30B model on medical Q&A data using LoRA (Low-Rank Adaptation).

## 🎯 Project Objective

Train a domain-specific medical assistant by fine-tuning the Nemotron-3-Nano base model on the MedMCQA dataset. This approach enables:
- **Efficient training**: LoRA trains only ~0.1% of parameters (adapter layers) instead of full 30B weights
- **Lower hardware requirements**: Fits on a single A100 80GB GPU
- **Fast iteration**: Training completes in hours, not days

## 📖 Learning Approach

This project takes a **progressive approach** to help you understand each layer of the stack:

1. **Start simple**: Train with standard HuggingFace (Transformers + PEFT + TRL)
2. **Then optimize**: Add Unsloth to see the performance gains
3. **Compare**: Measure the actual speedup on your hardware

This makes it easier to debug issues and understand what's "standard" vs "optimized."

## 🧠 Model Overview

| Property | Value |
|----------|-------|
| **Base Model** | [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16) |
| **Architecture** | Mamba2-Transformer Hybrid MoE (Mixture of Experts) |
| **Parameters** | 30B total, ~3B active per token |
| **Context Length** | Up to 1M tokens |
| **Precision** | BF16 |
| **License** | NVIDIA Open Model License |

### Why Nemotron-3-Nano?

Unlike traditional dense transformers, this model uses a **Mixture of Experts (MoE)** architecture where only a subset of parameters (3B out of 30B) activates per forward pass. This means:
- Faster inference than a dense 30B model
- Better performance than a dense 3B model
- Excellent for specialized domain adaptation

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Dataset** | [openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa) |
| **Domain** | Medical entrance exams (AIIMS, NEET PG) |
| **Task** | Multiple-choice question answering |
| **Size** | ~194k questions (train), ~6k (validation), ~4k (test) |
| **Subjects** | Anatomy, Biochemistry, Pharmacology, Medicine, Surgery, etc. |

### Data Format

Each example contains:
```json
{
  "question": "Which vitamin is involved in...",
  "opa": "Option A text",
  "opb": "Option B text", 
  "opc": "Option C text",
  "opd": "Option D text",
  "cop": 0,  // Correct option (0=A, 1=B, 2=C, 3=D)
  "exp": "Explanation text...",
  "subject_name": "Pharmacology"
}
```

## 💻 Hardware Requirements

| Configuration | GPU | VRAM | Training Time (est.) |
|--------------|-----|------|---------------------|
| **Multi-GPU (Best)** | 8× NVIDIA A100 | 80GB each | ~20-30 minutes |
| **Recommended** | NVIDIA A100 | 80GB | ~4-6 hours (standard) / ~2-3 hours (Unsloth) |
| **Minimum** | NVIDIA A100 | 40GB | ~6-8 hours (smaller batch) |
| **Alternative** | NVIDIA H100 | 80GB | ~2-3 hours (standard) / ~1-2 hours (Unsloth) |

> ⚠️ **Note**: Consumer GPUs (RTX 4090, etc.) may not have sufficient VRAM for this 30B model even with LoRA.

## 🛠️ Tech Stack

### Core Stack (Standard HuggingFace)

| Component | Purpose |
|-----------|---------|
| **[Transformers](https://github.com/huggingface/transformers)** | Model loading and tokenization |
| **[PEFT](https://github.com/huggingface/peft)** | Parameter-Efficient Fine-Tuning (LoRA) |
| **[TRL](https://github.com/huggingface/trl)** | `SFTTrainer` for LLM fine-tuning |
| **[Datasets](https://github.com/huggingface/datasets)** | Dataset loading and preprocessing |

### Optional Optimization

| Component | Purpose |
|-----------|---------|
| **[Unsloth](https://github.com/unslothai/unsloth)** | Drop-in optimization layer (2-3x faster, 50% less memory) |

Unsloth is added later to demonstrate performance gains while keeping the core training logic unchanged.

## 📁 Project Structure

```
nemotron-nano-v3-lora/
├── README.md                 # This file
├── AGENTS.md                 # Architecture & methodology docs
├── pyproject.toml            # Project config & dependencies (uv)
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Load & format MedMCQA dataset
│   ├── 02_model_loading.ipynb      # Load model with standard HuggingFace
│   ├── 03_training.ipynb           # Train with TRL (baseline, single GPU)
│   └── 04_unsloth_training.ipynb   # Train with Unsloth (compare performance)
├── scripts/
│   ├── train_multigpu.py     # Multi-GPU training script (DDP, standard HuggingFace)
│   └── train_unsloth.py      # Multi-GPU training with Unsloth optimization (2-3x faster)
├── logs/                     # Training logs
├── data/
│   └── medmcqa_formatted/    # Preprocessed dataset (Arrow format)
└── outputs/
    └── lora_adapter*/        # Saved LoRA adapters
```

## 🚀 Quick Start

We use [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management.

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Enter project directory
cd nemotron-nano-v3-lora

# 3. Initialize the project (creates venv)
uv sync

# 4. Add dependencies as needed (we'll do this step-by-step)
uv add datasets  # example: adds to pyproject.toml + installs

# 5. Run notebooks or scripts
uv run jupyter lab
```

## 🚀 Multi-GPU Training

For maximum speed, use Distributed Data Parallel (DDP) across all GPUs.

### Option 1: Unsloth-Optimized Training (Recommended)

2-3x faster training with optimized CUDA kernels:

```bash
# Create logs directory
mkdir -p logs

# Run training with Unsloth (4 GPUs example)
nohup uv run torchrun --nproc_per_node=4 scripts/train_unsloth.py > logs/training_unsloth_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Option 2: Standard HuggingFace Training

Baseline training without Unsloth (useful for debugging):

```bash
# Run training on 8 GPUs (background with logging)
nohup uv run torchrun --nproc_per_node=8 scripts/train_multigpu.py > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Monitor Training

```bash
# Watch the log in real-time
tail -f logs/training*.log

# Check GPU utilization
watch -n 1 nvidia-smi

# Check if training is running
ps aux | grep torchrun
```

### Training Configuration

| Parameter | Standard (`train_multigpu.py`) | Unsloth (`train_unsloth.py`) |
|-----------|-------------------------------|------------------------------|
| Per-device batch size | 4 | 32 |
| Gradient accumulation | 4 | 2 |
| Effective batch size (4 GPUs) | 4 × 4 × 4 = **64** | 32 × 2 × 4 = **256** |
| Optimizer | AdamW 8-bit | AdamW 8-bit |
| Learning rate | 2e-4 | 2e-4 |
| Epochs | 1 | 1 |

### DDP vs Model Parallelism

| Mode | How it works | Speedup |
|------|--------------|---------|
| **Model Parallelism** (notebooks) | Model sharded across GPUs | Good for large models |
| **Data Parallelism** (DDP script) | Each GPU processes different batches | **True 8x throughput** |

The multi-GPU script uses DDP for true parallel training where each GPU processes different data batches simultaneously.

> **Note**: When using quantized models (4-bit/8-bit) with DDP, each rank must load the model on its own GPU using `device_map={"": f"cuda:{local_rank}"}`. The `train_unsloth.py` script handles this automatically.

### Why uv?

| Feature | pip | uv |
|---------|-----|-----|
| Install speed | ~60s | ~2s |
| Lock file | ❌ | ✅ `uv.lock` |
| Reproducibility | Fragile | Deterministic |
| Resolution | Slow | 10-100x faster |

## 📈 Expected Outputs

After training, you'll have:
1. **LoRA adapter weights** (~100-500MB) in `outputs/checkpoints/`
2. **Training logs** with loss curves
3. **Merged model** (optional) for standalone deployment

The LoRA adapters can be:
- Loaded on top of the base model for inference
- Merged into the base model for simplified deployment
- Pushed to Hugging Face Hub for sharing

## 📚 Learning Resources

### Core Concepts
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Understanding Low-Rank Adaptation
- [NVIDIA Nemotron Paper](https://arxiv.org/abs/2512.20848) - Model architecture
- [TRL Documentation](https://huggingface.co/docs/trl) - SFTTrainer usage
- [PEFT Documentation](https://huggingface.co/docs/peft) - LoRA configuration

### Unsloth Optimization (Notebook 04)
- [Unsloth Nemotron Guide](https://docs.unsloth.ai/models/nemotron-3)
- [Reference Colab Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Nemotron-3-Nano-30B-A3B_A100.ipynb)

## 📄 License

This project uses:
- **Model**: NVIDIA Open Model License
- **Dataset**: Check MedMCQA license on Hugging Face
- **Code**: MIT License
