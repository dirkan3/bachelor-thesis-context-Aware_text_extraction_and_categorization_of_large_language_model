# ========================================
# Merge LoRA Adapter with Base Model on Local Server
# ========================================

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --------- Define paths ---------
MODEL_NAME = "gemma-2-9b-it"
BASE_MODEL = f"/home/joto21mu/{MODEL_NAME}"
LORA_DIR = "/home/joto21mu/gemma-2-9b-finetuned"
OUTPUT_DIR = f"/home/joto21mu/{MODEL_NAME}-merged"

# --------- Load base model ---------
print("Loading base model from local path...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager"
)

# --------- Load and merge LoRA adapter ---------
print("Merging LoRA adapter with base model...")
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model = model.merge_and_unload()

# --------- Save merged model ---------
print(f"Saving merged model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Merged model is saved at:", OUTPUT_DIR)
