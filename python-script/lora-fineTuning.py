# ========================================
# HuggingFace Fine-Tuning in Google Colab (Auto-zip + download checkpoints after every 20 steps)
# ========================================

# ------------------- Install necessary libraries -------------------
!pip install -q transformers datasets accelerate bitsandbytes peft huggingface_hub

# ------------------- HuggingFace Login -------------------
from huggingface_hub import login

HUGGINGFACE_TOKEN = ""
login(HUGGINGFACE_TOKEN)

# ------------------- Import libraries -------------------
import gc
import os
import shutil
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm.auto import tqdm
from google.colab import files

# ------------------- Clear cache -------------------
gc.collect()
torch.cuda.empty_cache()

# ------------------- Upload training files -------------------
uploaded = files.upload()
TRAIN_FILES = list(uploaded.keys())
print(f"Uploaded training files: {TRAIN_FILES}")

# ------------------- Config -------------------
MODEL_NAME = "google/gemma-2-9b-it"
OUTPUT_DIR = "gemma-2-9b-finetuned"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "latest-checkpoint")
CONTEXT_LENGTH = 1024
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
EPOCHS = 5
LEARNING_RATE = 2e-5
RESUME_FROM_CHECKPOINT = os.path.exists(CHECKPOINT_DIR)

# ------------------- GPU Check -------------------
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Enable GPU in Runtime > Change runtime type.")

# ------------------- Load Dataset -------------------
dataset = load_dataset("json", data_files=TRAIN_FILES, split="train")

# ------------------- Model & Tokenizer -------------------
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager",
    device_map="auto",
    trust_remote_code=True,
    quantization_config=quantization_config
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ------------------- Preprocessing -------------------
def tokenize_function(example):
    full_prompt = (
        "### Instruction:\nExtract sender and recipient:\n"
        + example["input"]
        + "\n\n### Response:\n"
        + example["output"]
    )
    return tokenizer(full_prompt, truncation=True, max_length=CONTEXT_LENGTH)

tokenized_dataset = dataset.map(tokenize_function, batched=False, remove_columns=dataset.column_names)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ------------------- Custom Callbacks -------------------
class TQDMProgressBarCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.pbar = tqdm(total=state.max_steps, desc="Training Progress", position=0, leave=True)
    def on_step_end(self, args, state, control, **kwargs):
        self.pbar.update(1)
    def on_train_end(self, args, state, control, **kwargs):
        self.pbar.close()

class AutoZipCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        latest_ckpt = max([
            os.path.join(args.output_dir, d)
            for d in os.listdir(args.output_dir)
            if d.startswith("checkpoint-")
        ], key=os.path.getmtime)
        shutil.make_archive("latest_checkpoint", 'zip', latest_ckpt)
        files.download("latest_checkpoint.zip")

# ------------------- Training Arguments -------------------
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    save_steps=20,
    save_total_limit=3,
    bf16=True,
    report_to="none",
    push_to_hub=False,
    save_strategy="steps",
    fp16=False,
    resume_from_checkpoint=RESUME_FROM_CHECKPOINT,
)

# ------------------- Trainer Setup -------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    callbacks=[TQDMProgressBarCallback(), AutoZipCheckpointCallback()],
)

# ------------------- Start Training -------------------
trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)

# ------------------- Save Final Model -------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model and tokenizer saved to {OUTPUT_DIR}")

# ------------------- Zip and Download Final Output -------------------
shutil.make_archive(OUTPUT_DIR, 'zip', OUTPUT_DIR)
files.download(f"{OUTPUT_DIR}.zip")
