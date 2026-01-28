# AGENTS.md - Architecture & Methodology

This document explains the technical architecture of LoRA fine-tuning for systems engineers transitioning to ML.

## 🔧 What Problem Does LoRA Solve?

### The Full Fine-tuning Problem

Traditional fine-tuning updates **all** model parameters:

```
Full Fine-tuning Memory = Model Weights + Gradients + Optimizer States
                        = 30B × 2 bytes + 30B × 2 bytes + 30B × 8 bytes
                        = ~360 GB just for the model!
```

This is impractical for most setups.

### The LoRA Solution

**LoRA (Low-Rank Adaptation)** freezes the base model and injects small trainable matrices:

```
Original: Y = W × X           (W is 4096 × 4096 = 16M params, frozen)
LoRA:     Y = W × X + B × A × X   (A is 4096 × 16, B is 16 × 4096 = 131K params, trainable)
```

| Approach | Trainable Params | Memory Required |
|----------|-----------------|-----------------|
| Full Fine-tune | 30B | ~360 GB |
| LoRA (rank=16) | ~30M | ~60 GB |
| **Reduction** | **1000x fewer** | **6x less** |

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   MedMCQA    │───▶│  Tokenizer   │───▶│  Training Batch  │   │
│  │   Dataset    │    │  (Format)    │    │  (Token IDs)     │   │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘   │
│                                                    │             │
│                                                    ▼             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Nemotron-3-Nano Base                     │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │            Frozen Transformer Layers                │  │   │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐            │  │   │
│  │  │  │  Attn   │  │   MoE   │  │  Mamba  │   × N      │  │   │
│  │  │  │ (Q,K,V) │  │ Experts │  │  Block  │            │  │   │
│  │  │  └────┬────┘  └────┬────┘  └────┬────┘            │  │   │
│  │  │       │            │            │                  │  │   │
│  │  │  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐            │  │   │
│  │  │  │  LoRA   │  │  LoRA   │  │  LoRA   │ ◀─ Trained │  │   │
│  │  │  │ Adapter │  │ Adapter │  │ Adapter │            │  │   │
│  │  │  └─────────┘  └─────────┘  └─────────┘            │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│                     ┌──────────────────┐                        │
│                     │   Loss Function  │                        │
│                     │  (Cross Entropy) │                        │
│                     └────────┬─────────┘                        │
│                              │                                   │
│                              ▼                                   │
│                     ┌──────────────────┐                        │
│                     │   Backprop only  │                        │
│                     │   through LoRA   │                        │
│                     └──────────────────┘                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 LoRA Hyperparameters Explained

For systems engineers, think of these as **tuning knobs**:

### `rank` (r) - Compression Factor
```python
lora_rank = 16  # Typical range: 8-64
```
- **What it does**: Controls the "width" of LoRA matrices (A and B)
- **Higher rank** = More capacity, more memory, better fit
- **Lower rank** = Less memory, faster training, potential underfitting
- **Analogy**: Like choosing buffer size in I/O operations

### `lora_alpha` - Learning Rate Scaling
```python
lora_alpha = 32  # Typically 2x rank
```
- **What it does**: Scales the LoRA update by `alpha/rank`
- **Analogy**: Like a PID gain parameter
- **Rule of thumb**: Start with `alpha = 2 × rank`

### `target_modules` - Where to Inject LoRA
```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```
- **What it does**: Specifies which weight matrices get LoRA adapters
- **Common targets**: Attention projections (Q, K, V, O) and MLP layers
- **Trade-off**: More modules = better adaptation, more memory

### `lora_dropout` - Regularization
```python
lora_dropout = 0.05  # 0-0.1 typical
```
- **What it does**: Randomly zeros LoRA outputs during training
- **Purpose**: Prevents overfitting to training data

## 🔄 Training Workflow

### Step 1: Data Preprocessing

Convert MedMCQA format to instruction format:

```python
# Raw MedMCQA sample
{
    "question": "Which drug causes cinchonism?",
    "opa": "Aspirin",
    "opb": "Quinine", 
    "opc": "Paracetamol",
    "opd": "Ibuprofen",
    "cop": 1,  # Correct answer is B (0-indexed)
    "exp": "Cinchonism is caused by quinine..."
}

# Transformed to instruction format
"""
<|im_start|>system
You are a medical expert. Answer the multiple choice question.<|im_end|>
<|im_start|>user
Question: Which drug causes cinchonism?

A) Aspirin
B) Quinine
C) Paracetamol
D) Ibuprofen<|im_end|>
<|im_start|>assistant
The correct answer is B) Quinine.

Explanation: Cinchonism is caused by quinine...<|im_end|>
"""
```

### Step 2: Tokenization

Convert text to token IDs that the model understands:

```python
# Text → Tokens → IDs
"Quinine" → ["Qu", "in", "ine"] → [15234, 287, 1843]
```

### Step 3: Forward Pass

```
Input IDs → Embedding → Transformer Layers (+ LoRA) → Logits → Loss
```

### Step 4: Backward Pass (LoRA Only)

```
Loss → Gradients → Update LoRA weights only (base model frozen)
```

### Step 5: Repeat

Iterate over dataset for N epochs until loss converges.

## 📈 Key Training Parameters

```python
training_args = {
    # Learning Rate
    "learning_rate": 2e-4,      # Higher than full fine-tune (since fewer params)
    "warmup_steps": 100,        # Gradually increase LR at start
    
    # Batch Size (trade-off: larger = stable gradients, more memory)
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,  # Effective batch = 4 × 4 = 16
    
    # Training Duration
    "num_train_epochs": 3,      # Full passes through dataset
    "max_steps": -1,            # Or specify exact steps
    
    # Optimization
    "optim": "adamw_8bit",      # Memory-efficient optimizer
    "weight_decay": 0.01,       # L2 regularization
    
    # Checkpointing
    "save_strategy": "steps",
    "save_steps": 500,
    
    # Precision
    "bf16": True,               # Use BF16 for A100/H100
    "fp16": False,              # Use FP16 for older GPUs
}
```

## 🎯 Medical QA Agent Use Case

After training, the model becomes a **Medical QA Agent**:

```
┌─────────────────────────────────────────────────────────┐
│                   Inference Pipeline                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  User Query ──▶ Prompt Template ──▶ Model ──▶ Response  │
│                                                          │
│  "What causes                                            │
│   diabetes?"   ──▶  System: Medical expert              │
│                     User: {query}                        │
│                     ──▶  Nemotron + LoRA                │
│                          ──▶  "Diabetes is caused by..."│
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Inference Options

| Method | Pros | Cons |
|--------|------|------|
| **Base + Adapter** | Swap adapters easily, smaller storage | Slightly slower load |
| **Merged Model** | Single model, fastest inference | Larger storage, can't swap |

## 🔍 Monitoring & Debugging

### Key Metrics to Watch

```python
# During training, monitor:
1. train_loss        # Should decrease steadily
2. eval_loss         # Should decrease, not diverge from train_loss
3. learning_rate     # Should follow warmup schedule
4. grad_norm         # Should be stable (not exploding)
```

### Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Loss explodes | LR too high | Reduce `learning_rate` by 10x |
| Loss plateaus early | LR too low | Increase `learning_rate` |
| Eval loss diverges | Overfitting | Increase `lora_dropout`, reduce epochs |
| OOM errors | Batch too large | Reduce `batch_size`, increase `gradient_accumulation` |

## 🚀 Deployment Considerations

### Option 1: HuggingFace Inference
```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "path/to/lora_adapter")
```

### Option 2: vLLM (High Throughput)
```bash
vllm serve nvidia/Nemotron-3-Nano --lora-modules medical=./lora_adapter
```

### Option 3: TensorRT-LLM (Maximum Performance)
Requires merging and conversion to TensorRT engine format.

## 📚 Glossary for Systems Engineers

| ML Term | Systems Analogy |
|---------|-----------------|
| **Epoch** | Full scan of dataset (like iterating a file once) |
| **Batch** | Processing chunk (like buffer size) |
| **Gradient** | Error signal for weight update (like feedback in control loop) |
| **Loss** | Error metric (like latency or error rate) |
| **Checkpoint** | Saved model state (like snapshot/backup) |
| **Inference** | Production serving (like handling requests) |
| **LoRA rank** | Compression level (like compression ratio) |
| **Tokenizer** | Text encoder/decoder (like serialization format) |
