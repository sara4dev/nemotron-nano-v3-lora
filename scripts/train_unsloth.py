#!/usr/bin/env python3
"""
Multi-GPU Training Script with Unsloth Optimization

Launch with:
    torchrun --nproc_per_node=4 scripts/train_unsloth.py

Or with accelerate:
    accelerate launch --num_processes=4 scripts/train_unsloth.py

This uses Unsloth's optimized kernels for 2-3x faster training.
Compatible with DDP across multiple GPUs.

Requirements:
    pip install unsloth
    # Or: pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
"""

import os
import json
import time
from pathlib import Path

import torch
# Import Unsloth - this applies kernel optimizations automatically
from unsloth import FastLanguageModel
from datasets import load_from_disk
from huggingface_hub import login
from trl import SFTTrainer, SFTConfig

# =============================================================================
# Configuration - Adjust these for your setup
# =============================================================================

# Model - use local cache path if available, otherwise HuggingFace ID
_HF_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
_LOCAL_CACHE_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/snapshots/f73a11c1f0964a5851f984b70cd31dda9a44f01c"
)
MODEL_NAME = _LOCAL_CACHE_PATH if os.path.exists(_LOCAL_CACHE_PATH) else _HF_MODEL_ID
MAX_SEQ_LENGTH = 2048

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]

# Training Configuration for B300 288GB (4 GPUs)
# Effective batch size = per_device * gradient_accumulation * num_gpus
# Example: 32 * 2 * 4 = 256
PER_DEVICE_BATCH_SIZE = 32  # B300 288GB can handle large batches
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
WARMUP_STEPS = 100

# Unsloth-specific options
# B300 has enough VRAM for BF16, keeping 4-bit disabled for better quality
LOAD_IN_4BIT = False


def main():
    # =========================================================================
    # 1. Setup
    # =========================================================================
    
    # Authenticate with HuggingFace (skip if offline mode or rate-limited)
    if os.environ.get("HF_TOKEN") and not os.environ.get("HF_HUB_OFFLINE"):
        try:
            login(token=os.environ["HF_TOKEN"])
            print("✅ Logged in to HuggingFace Hub")
        except Exception as e:
            print(f"⚠️  HuggingFace login failed (will use cached model): {e}")
    
    # Get distributed training info
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if local_rank == 0:
        print(f"🚀 Unsloth Multi-GPU Training")
        print(f"   Model: {MODEL_NAME}")
        print(f"   GPUs: {world_size}")
        print(f"   4-bit quantization: {LOAD_IN_4BIT}")
    
    # =========================================================================
    # 2. Load Dataset
    # =========================================================================
    
    data_path = Path(__file__).parent.parent / "data" / "medmcqa_formatted"
    
    # Check if data exists, if not provide instructions
    if not data_path.exists():
        if local_rank == 0:
            print(f"❌ Dataset not found at: {data_path}")
            print("   Run notebook 01_data_exploration.ipynb first to prepare the data.")
        return
    
    formatted_dataset = load_from_disk(str(data_path))
    
    if local_rank == 0:
        print(f"📊 Dataset: {len(formatted_dataset['train']):,} train, {len(formatted_dataset['validation']):,} val")
    
    # =========================================================================
    # 3. Load Model with Unsloth (Optimized Loading)
    # =========================================================================
    
    if local_rank == 0:
        print(f"⏳ Loading model with Unsloth optimizations...")
    
    # Unsloth's optimized model loading
    # - Applies fused attention kernels
    # - Enables efficient gradient checkpointing
    # - Optional 4-bit quantization
    # Use local_files_only if HF_HUB_OFFLINE is set to avoid network calls
    local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    
    # For quantized models with DDP, we must specify device_map to load on correct GPU
    # Each rank loads the model on its own GPU
    device_map = {"": f"cuda:{local_rank}"}
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=LOAD_IN_4BIT,
        trust_remote_code=True,
        local_files_only=local_only,
        device_map=device_map,  # Required for quantized models with DDP
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
    if local_rank == 0:
        print("✅ Model loaded with Unsloth optimizations")
    
    # =========================================================================
    # 4. Apply LoRA with Unsloth
    # =========================================================================
    
    # Unsloth's optimized LoRA application
    # - Uses optimized LoRA math
    # - Applies gradient checkpointing automatically
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=42,
    )
    
    if local_rank == 0:
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🔧 LoRA applied:")
        print(f"   Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"   Total params: {total_params:,}")
    
    # =========================================================================
    # 5. Configure Training
    # =========================================================================
    
    OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "lora_adapter_unsloth"
    
    # Calculate effective batch size
    effective_batch_size = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * world_size
    
    # Training arguments optimized for Unsloth + DDP
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        
        # Training duration
        num_train_epochs=NUM_EPOCHS,
        
        # Batch size - this is PER GPU
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        
        # Learning rate
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        
        # Optimization - use adamw_8bit for memory efficiency with Unsloth
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Precision
        bf16=True,
        
        # Logging
        logging_steps=10,
        logging_first_step=True,
        report_to=["tensorboard"],
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=500,
        
        # Checkpointing
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # SFT-specific
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,  # Can enable for additional speedup if sequences are short
        
        # Multi-GPU settings
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        
        seed=42,
    )
    
    if local_rank == 0:
        print(f"\n⚙️  Training configuration:")
        print(f"   Per-device batch size: {PER_DEVICE_BATCH_SIZE}")
        print(f"   Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
        print(f"   Number of GPUs: {world_size}")
        print(f"   Effective batch size: {effective_batch_size}")
        print(f"   Learning rate: {LEARNING_RATE}")
        print(f"   4-bit quantization: {LOAD_IN_4BIT}")
        print(f"   Optimizer: adamw_8bit")
    
    # =========================================================================
    # 6. Create Trainer and Train
    # =========================================================================
    
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["validation"],
        processing_class=tokenizer,
    )
    
    if local_rank == 0:
        num_training_examples = len(trainer.train_dataset)
        steps_per_epoch = num_training_examples // effective_batch_size
        estimated_time_per_step = 20  # Unsloth is ~2x faster, estimate 20s/step
        estimated_total_hours = (steps_per_epoch * estimated_time_per_step) / 3600
        
        print(f"\n📋 Training plan:")
        print(f"   Training examples: {num_training_examples:,}")
        print(f"   Steps per epoch: {steps_per_epoch:,}")
        print(f"   Total epochs: {NUM_EPOCHS}")
        print(f"   Estimated time: ~{estimated_total_hours:.1f} hours (with Unsloth speedup)")
    
    # Train
    if local_rank == 0:
        print("\n🚀 Starting Unsloth-accelerated training...")
        print("=" * 60)
    
    train_start = time.time()
    train_result = trainer.train()
    train_time = time.time() - train_start
    
    if local_rank == 0:
        print("=" * 60)
        print(f"✅ Training complete in {train_time/60:.1f} minutes!")
        print(f"\n📈 Training Summary:")
        print(f"   Total steps: {train_result.global_step}")
        print(f"   Training loss: {train_result.training_loss:.4f}")
        print(f"   Time per step: {train_time / train_result.global_step:.2f}s")
    
    # =========================================================================
    # 7. Save Model (only main process)
    # =========================================================================
    
    if local_rank == 0:
        final_adapter_path = OUTPUT_DIR / "final_adapter"
        print(f"\n💾 Saving adapter to: {final_adapter_path}")
        
        # Save LoRA adapter
        trainer.save_model(str(final_adapter_path))
        tokenizer.save_pretrained(str(final_adapter_path))
        
        # Optionally save merged model (uncomment if needed)
        # merged_path = OUTPUT_DIR / "merged_model"
        # print(f"   Saving merged model to: {merged_path}")
        # model.save_pretrained_merged(str(merged_path), tokenizer, save_method="merged_16bit")
        
        print("✅ Model saved!")
        
        # Save training info
        info_path = OUTPUT_DIR / "training_info.json"
        training_info = {
            "model_name": MODEL_NAME,
            "lora_config": {
                "r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "target_modules": TARGET_MODULES,
            },
            "training_config": {
                "effective_batch_size": effective_batch_size,
                "learning_rate": LEARNING_RATE,
                "num_epochs": NUM_EPOCHS,
                "max_seq_length": MAX_SEQ_LENGTH,
                "load_in_4bit": LOAD_IN_4BIT,
            },
            "results": {
                "total_steps": train_result.global_step,
                "training_loss": train_result.training_loss,
                "training_time_minutes": train_time / 60,
                "time_per_step_seconds": train_time / train_result.global_step,
            },
            "hardware": {
                "num_gpus": world_size,
                "gpu_type": "B300-288GB",
            },
        }
        with open(info_path, "w") as f:
            json.dump(training_info, f, indent=2)
        print(f"📄 Training info saved to: {info_path}")
    
    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    
    if local_rank == 0:
        print("\n🎉 All done!")


if __name__ == "__main__":
    main()
