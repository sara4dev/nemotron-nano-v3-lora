#!/usr/bin/env python3
"""
Merge LoRA adapter weights into the base model.

This creates a standalone model that can be served directly without LoRA loading,
which avoids SGLang's LoRA dimension issues with MoE architectures.

Usage:
    python scripts/merge_lora.py

Output:
    outputs/merged_model/ - Full merged model in BF16
"""

import os
import torch
from pathlib import Path

# Unsloth for loading the adapter
from unsloth import FastLanguageModel

def main():
    # Paths
    adapter_path = Path(__file__).parent.parent / "outputs" / "lora_adapter_unsloth" / "final_adapter"
    output_path = Path(__file__).parent.parent / "outputs" / "merged_model"
    
    print(f"📂 Adapter path: {adapter_path}")
    print(f"📂 Output path: {output_path}")
    
    if not adapter_path.exists():
        print(f"❌ Adapter not found at: {adapter_path}")
        return
    
    # Load model with adapter
    print("\n⏳ Loading model with LoRA adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        str(adapter_path),
        dtype=torch.bfloat16,
        load_in_4bit=False,
        trust_remote_code=True,
    )
    print("✅ Model loaded")
    
    # Merge and save
    print(f"\n⏳ Merging LoRA weights and saving to: {output_path}")
    print("   This may take a few minutes...")
    
    model.save_pretrained_merged(
        str(output_path),
        tokenizer,
        save_method="merged_16bit",
    )
    
    print(f"\n✅ Merged model saved to: {output_path}")
    print("\n🚀 You can now serve the model with SGLang:")
    print(f"   python -m sglang.launch_server --model-path {output_path} --trust-remote-code")

if __name__ == "__main__":
    main()
