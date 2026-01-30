# Nemotron-3-Nano Medical LoRA Fine-tuning

Fine-tuning NVIDIA's Nemotron-3-Nano-30B model on medical Q&A data using LoRA.

## 📊 Results

| Metric | Base Model | Fine-tuned LoRA | Improvement |
|--------|-----------|-----------------|-------------|
| **Accuracy** | 34.64% | 61.73% | **+27.09 pp** |

Evaluated on MedMCQA validation set (4,183 samples). See [full results](#evaluation-by-subject) below.

## 🧠 Model & Dataset

| | |
|---|---|
| **Base Model** | [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16) |
| **Architecture** | Mamba2-Transformer Hybrid MoE (30B total, ~3B active) |
| **Dataset** | [openlifescienceai/medmcqa_formatted](https://huggingface.co/datasets/openlifescienceai/medmcqa_formatted) |
| **Task** | Medical multiple-choice QA |

## 💻 Hardware & Training Time

| GPU | Training Time |
|-----|---------------|
| 4× NVIDIA B300 (tested) | ~4 hours with Unsloth |

## 🚀 Quick Start

```bash
# Install uv and sync dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Run multi-GPU training with Unsloth
uv run torchrun --nproc_per_node=4 scripts/train_unsloth.py

# Or standard HuggingFace training
uv run torchrun --nproc_per_node=4 scripts/train_multigpu.py
```

## 📁 Project Structure

```
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Dataset exploration
│   ├── 02_model_loading.ipynb      # Model loading basics
│   └── 03_training.ipynb           # Single-GPU training
├── scripts/
│   ├── train_unsloth.py            # Multi-GPU with Unsloth (recommended)
│   ├── train_multigpu.py           # Multi-GPU standard HuggingFace
│   └── evaluate.py                 # Model evaluation
├── AGENTS.md                       # Instructions for AI agents
└── pyproject.toml                  # Dependencies (uv)
```

## 🛠️ Tech Stack

- **[Transformers](https://github.com/huggingface/transformers)** + **[PEFT](https://github.com/huggingface/peft)** + **[TRL](https://github.com/huggingface/trl)** - Core training stack
- **[Unsloth](https://github.com/unslothai/unsloth)** - 2-3x faster training, 50% less memory

## Evaluation by Subject

| Subject | Base | Fine-tuned | Change |
|---------|------|------------|--------|
| Psychiatry | 31.25% | 87.50% | +56.25 pp |
| Biochemistry | 36.26% | 77.19% | +40.93 pp |
| Physiology | 33.33% | 73.68% | +40.35 pp |
| Pharmacology | 35.80% | 72.43% | +36.63 pp |
| Pathology | 30.56% | 72.11% | +41.55 pp |
| Medicine | 35.59% | 65.42% | +29.83 pp |
| Surgery | 33.33% | 65.04% | +31.71 pp |
| Anaesthesia | 44.12% | 64.71% | +20.59 pp |
| Anatomy | 33.76% | 64.10% | +30.34 pp |
| Ophthalmology | 29.31% | 63.79% | +34.48 pp |
| Social & Preventive Medicine | 34.88% | 63.57% | +28.69 pp |
| Gynaecology & Obstetrics | 33.93% | 62.95% | +29.02 pp |
| Microbiology | 28.69% | 62.30% | +33.61 pp |
| ENT | 37.74% | 62.26% | +24.52 pp |
| Radiology | 36.23% | 60.87% | +24.64 pp |
| Skin | 29.41% | 58.82% | +29.41 pp |
| Pediatrics | 32.91% | 54.27% | +21.36 pp |
| Forensic Medicine | 41.79% | 53.73% | +11.94 pp |
| Dental | 36.12% | 52.66% | +16.54 pp |
| Orthopaedics | 35.00% | 35.00% | +0.00 pp |

## 📚 Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685) | [NVIDIA Nemotron Paper](https://arxiv.org/abs/2512.20848)
- [TRL Docs](https://huggingface.co/docs/trl) | [PEFT Docs](https://huggingface.co/docs/peft) | [Unsloth Docs](https://docs.unsloth.ai)

## 📄 License

Model: NVIDIA Open Model License | Dataset: MedMCQA License | Code: MIT
