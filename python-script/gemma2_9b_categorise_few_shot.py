import os
import time
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------- Config -------------------
MODEL_NAME = "gemma-2-9b-it"
INPUT_FILE = "atc_comms166.txt"
CHECKPOINT_FILE = "checkpoint_state.pt"
MAX_NEW_TOKENS = 256
CONTEXT_LIMIT = 5
# ----------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
safe_model_name = MODEL_NAME.replace("/", "_")
output_file = f"{safe_model_name}_context_few_output_{timestamp}.txt"
model_path = f"/home/joto21mu/{MODEL_NAME}"

print(f"Using model: {MODEL_NAME}")
print(f"Output file: {output_file}")

# -------- Load Tokenizer and Model --------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Tokenizer loaded.")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)
model.eval()
print("Model loaded.")

# -------- Prompt Components --------
system_prompt = """[System]
You are an advanced ATC communication analyst with expertise in categorizing and interpreting radiocommunications based on established formats and rules. Your specialty lies in accurately processing and understanding context from individual lines of communication extracted from a .txt file without making assumptions or alterations.

Your task is to categorize and analyze ATC radiocommunications based on the guidelines provided. Please ensure to follow these formats and rules closely:

GENERAL MESSAGE FORMATS:

- If a message contains two callsigns at the beginning, it usually follows:
	- Format: [Recipient], [Sender], [Message]

- If a single callsign is at the beginning:
	-Format: [Recipient], [Message]

- If a callsign appears at the end:
	- Format: [Message], [Sender]

- Short acknowledgments (e.g. "roger", "thanks", "copy", etc.) are typically replies and reverse the sender/receiver from the last message.

RULES:

1. DO NOT invent or assume any new callsigns or names.
2. If the message does not clearly indicate who is sending or receiving the message, use "unknown".
3. Each block must follow:
Message:
From:
To:
4. Message must match the input.
5. Do NOT correct or edit the message.
6. If a short message (e.g., "Roger", "Thanks", "Good night") follows a clear prior instruction:
	- Swap From and To from the previous message.
"""

user_examples = """[Examples - Correct Parsing]
Input: "Tower, Delta 680 Heavy at South to the visual runway."
Message: Tower, Delta 680 Heavy at South to the visual runway.
From: Delta 680 Heavy
To: Tower

Input: "Delta 680 Heavy, LA Tower, 1220 at 8, runway 24R clear to land."
Message: Delta 680 Heavy, LA Tower, 1220 at 8, runway 24R clear to land.
From: Tower
To: Delta 680 Heavy

---

Input: "Delta 623, RNAV Delray, wind 100 at 3, runway 24L, clear for takeoff."
Message: Delta 623, RNAV Delray, wind 100 at 3, runway 24L, clear for takeoff.
From: Tower
To: Delta 623

Input: "RNAV Delray, clear for takeoff, 24L, Delta 623."
Message: RNAV Delray, clear for takeoff, 24L, Delta 623.
From: Tower
To: Delta 623

---

Input: "Avianca 083 Heavy, contact SoCal departure, good night."
Message: Avianca 083 Heavy, contact SoCal departure, good night.
From: Tower
To: Avianca 083 Heavy

Input: "Local Apollo, this is Avianca 083, can you confirm?"
Message: Local Apollo, this is Avianca 083, can you confirm?
From: Avianca 083 Heavy
To: Tower

Input: "124.3."
Message: 124.3.
From: Tower
To: Avianca 083 Heavy

Input: "124.3."
Message: 124.3.
From: Avianca 083 Heavy
To: Tower

---

Input: "Delta 355 Heavy, on the visual 24R."
Message: Delta 355 Heavy, on the visual 24R.
From: Delta 355 Heavy
To: Tower

Input: "Delta 355 Heavy, LA Tower, wind 230 at 3, runway 24R, cleared to land."
Message: Delta 355 Heavy, LA Tower, wind 230 at 3, runway 24R, cleared to land.
From: LA Tower
To: Delta 355

Input: "Cleared to land, 24R, Delta 355."
Message: Cleared to land, 24R, Delta 355.
From: Delta 355
To: Tower"""

# -------- Load Input --------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    atc_lines = [line.strip() for line in f if line.strip()]

# -------- Resume Checkpoint --------
results = []
recent_blocks = []
start_time = time.time()
start_index = 0

if os.path.exists(CHECKPOINT_FILE):
    print(f"Resuming from checkpoint: {CHECKPOINT_FILE}")
    checkpoint = torch.load(CHECKPOINT_FILE)
    results = checkpoint["results"]
    recent_blocks = checkpoint["recent_blocks"]
    start_index = checkpoint["line_index"] + 1
else:
    print("No checkpoint found. Starting fresh.")

# -------- Inference Loop --------
for idx in range(start_index, len(atc_lines)):
    line = atc_lines[idx]
    print("\n-----------------------")
    print(f"[{idx + 1}/{len(atc_lines)}] Processing: {line}")

    context_window = "[Context - Most recent ATC messages first]\n" + "\n".join(reversed(recent_blocks)) if recent_blocks else ""
    user_input_line = f'Input: "{line}"\n'
    full_prompt = f"{system_prompt}\n{context_window}\n{user_examples}\n\n{user_input_line}"

    print("\n----- FULL PROMPT -----")
    print(full_prompt)
    print("----- END PROMPT -----\n")

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    parsed_lines = decoded[len(full_prompt):].strip().splitlines()

    msg_line = parsed_lines[0] if len(parsed_lines) > 0 else ""
    from_line = parsed_lines[1] if len(parsed_lines) > 1 else ""
    to_line = parsed_lines[2] if len(parsed_lines) > 2 else ""

    msg_text = msg_line.replace("Message:", "").strip()
    sender = from_line.replace("From:", "").strip()
    receiver = to_line.replace("To:", "").strip()

    # -------- Write Output --------
    block = f"Message: {msg_text}\nFrom: {sender}\nTo: {receiver}\n"
    results.append(block + "\n")

    recent_blocks.append(block)
    if len(recent_blocks) > CONTEXT_LIMIT:
        recent_blocks.pop(0)

    # -------- Save Checkpoint --------
    torch.save({
        "results": results,
        "recent_blocks": recent_blocks,
        "line_index": idx
    }, CHECKPOINT_FILE)

    torch.cuda.empty_cache()

# -------- Final Save --------
end_time = time.time()
print(f"Completed in {end_time - start_time:.2f} seconds.")

with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(results)

# Clean up
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

print(f"Results written to {output_file}")