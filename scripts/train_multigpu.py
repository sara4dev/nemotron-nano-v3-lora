#!/usr/bin/env python3
"""
Multi-GPU Training Script for Nemotron-3-Nano LoRA Fine-tuning

Launch with:
    torchrun --nproc_per_node=8 scripts/train_multigpu.py

Or with accelerate:
    accelerate launch --num_processes=8 scripts/train_multigpu.py

This uses Distributed Data Parallel (DDP) to run training across all GPUs.
Each GPU processes a different batch, giving true 8x parallelism.
"""

import os
import json
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


def main():
    # =========================================================================
    # 1. Setup
    # =========================================================================
    
    # Authenticate with HuggingFace
    if os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])
        print("✅ Logged in to HuggingFace Hub")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "outputs" / "training_config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    MODEL_NAME = config["model_name"]
    MAX_SEQ_LENGTH = config["training_config"]["max_seq_length"]
    
    # Get distributed training info
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if local_rank == 0:
        print(f"Configuration loaded: {MODEL_NAME}")
        print(f"Training with {world_size} GPUs")
    
    # =========================================================================
    # 2. Load Dataset
    # =========================================================================
    
    data_path = Path(__file__).parent.parent / "data" / "medmcqa_formatted"
    formatted_dataset = load_from_disk(str(data_path))
    
    if local_rank == 0:
        print(f"Dataset: {len(formatted_dataset['train']):,} train, {len(formatted_dataset['validation']):,} val")
    
    # =========================================================================
    # 3. Load Model and Tokenizer (NO device_map for DDP)
    # =========================================================================
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
    if local_rank == 0:
        print(f"Loading model: {MODEL_NAME}")
    
    # IMPORTANT: Don't use device_map="auto" for DDP!
    # Each process will load the full model and move it to its assigned GPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # No device_map - DDP handles device placement
    )
    
    if local_rank == 0:
        print("✅ Model loaded")
    
    # =========================================================================
    # 4. Apply LoRA
    # =========================================================================
    
    lora_config = LoraConfig(
        r=config["lora_config"]["r"],
        lora_alpha=config["lora_config"]["lora_alpha"],
        lora_dropout=config["lora_config"]["lora_dropout"],
        target_modules=config["lora_config"]["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias=config["lora_config"]["bias"],
    )
    
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    
    if local_rank == 0:
        model.print_trainable_parameters()
    
    # =========================================================================
    # 5. Configure Training
    # =========================================================================
    
    OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "lora_adapter_multigpu"
    
    # Training arguments for multi-GPU DDP
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        
        # Training duration
        num_train_epochs=1,
        
        # Batch size - this is PER GPU
        # Effective batch size = per_device_batch_size * gradient_accumulation * num_gpus
        # Example: 8 * 4 * 8 = 256 (H200 140GB)
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        
        # Learning rate
        learning_rate=2e-4,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        
        # Optimization
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Precision
        bf16=True,
        
        # Logging (only main process logs)
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
        packing=False,
        
        # Multi-GPU settings
        ddp_find_unused_parameters=False,  # More efficient
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        
        seed=42,
    )
    
    # Calculate effective batch size
    effective_batch_size = (
        sft_config.per_device_train_batch_size
        * sft_config.gradient_accumulation_steps
        * world_size
    )
    
    if local_rank == 0:
        print(f"\nTraining configuration:")
        print(f"  Per-device batch size: {sft_config.per_device_train_batch_size}")
        print(f"  Gradient accumulation: {sft_config.gradient_accumulation_steps}")
        print(f"  Number of GPUs: {world_size}")
        print(f"  Effective batch size: {effective_batch_size}")
    
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
        print(f"\nTraining plan:")
        print(f"  Training examples: {num_training_examples:,}")
        print(f"  Steps per epoch: {steps_per_epoch:,}")
        print(f"  Total epochs: {sft_config.num_train_epochs}")
    
    # Train
    if local_rank == 0:
        print("\n🚀 Starting training...")
        print("=" * 60)
    
    train_start = time.time()
    train_result = trainer.train()
    train_time = time.time() - train_start
    
    if local_rank == 0:
        print("=" * 60)
        print(f"✅ Training complete in {train_time/60:.1f} minutes!")
        print(f"\nTraining Summary:")
        print(f"  Total steps: {train_result.global_step}")
        print(f"  Training loss: {train_result.training_loss:.4f}")
    
    # =========================================================================
    # 7. Save Model (only main process)
    # =========================================================================
    
    if local_rank == 0:
        final_adapter_path = OUTPUT_DIR / "final_adapter"
        print(f"\nSaving adapter to: {final_adapter_path}")
        trainer.save_model(str(final_adapter_path))
        tokenizer.save_pretrained(str(final_adapter_path))
        print("✅ Model saved!")
    
    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
