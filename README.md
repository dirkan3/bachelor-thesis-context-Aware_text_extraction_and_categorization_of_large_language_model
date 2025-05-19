# bachelor-thesis-context-Aware_text_extraction_and_categorization_of_large_language_model
Code for the bachelor thesis conducted in collaborating with Jönköping University (2025)

Bachelor thesis available on Diva (Link).

ATC Communication Categorization with Gemma-2-9B

This project categorizes Air Traffic Control (ATC) communication messages using the `gemma-2-9b-it` model in both zero-shot and few-shot settings. It also explores performance improvements through LoRA-based fine-tuning with cross-validation.

Repository Structure
--------------------
.
├── gemma2_9b_categorise_few_shot.py
├── gemma2_9b_categorise_zero_shot.py
├── lora-fineTuning.py
├── merge.py
├── *.jsonl                    # Training data files (cross-validation)
├── *.txt                      # Inference data files (held-out evaluation)

Process Overview
----------------

1. Data Preparation
- `train_data_97.jsonl`, `train_data_155.jsonl`, `train_data_166.jsonl`, `train_data_230.jsonl`
  → Used for 4-fold cross-validation: three files used for training, one held out for evaluation.
- `97 - 1585 Token.txt`, `155 - 2680 Token.txt`, etc.
  → These `.txt` files are evaluation inputs, each corresponding to the .jsonl version but never included in the training.

2. Fine-Tuning (LoRA)
- Fine-tuning is performed using `lora-fineTuning.py` on Google Colab with PEFT's LoRA approach.
- Tokenizer and model are quantized to 4-bit precision with BitsAndBytesConfig.
- After every 20 steps, checkpoints are zipped and downloaded.
- Four separate fine-tuned models are created, each using a different held-out `.jsonl` file for cross-validation.

3. Merging LoRA Adapters
- Each fine-tuned model is merged with the base model using `merge.py`:
  peft_model.merge_and_unload()

4. Inference: Categorization of ATC Communications
- For each `.txt` file (e.g. `97 - 1585 Token.txt`):

  It is processed four times:

  | Model              | Few-shot        | Zero-shot        |
  |--------------------|-----------------|------------------|
  | Base (`gemma-2-9b-it`) | ✅ few_shot.py | ✅ zero_shot.py |
  | Fine-tuned          | ✅ few_shot.py | ✅ zero_shot.py |

- The categorization output follows the format:
  Message: ...
  From: ...
  To: ...

- Few-shot (`gemma2_9b_categorise_few_shot.py`) uses an in-context example block.
- Zero-shot (`gemma2_9b_categorise_zero_shot.py`) omits examples but keeps task instructions.

Configuration
-------------
- Modify these configs in the scripts as needed:
  MODEL_NAME = "gemma-2-9b-it"
  INPUT_FILE = "xxx.txt"
  CONTEXT_LIMIT = 5
  MAX_NEW_TOKENS = 256

- Update model_path to point to the fine-tuned directory when needed:
  model_path = "/home/yourname/gemma-2-9b-it-finetuned"

Evaluation Strategy
-------------------
- Each model is tested on the `.txt` file that corresponds to the `.jsonl` held out from training.
- Outputs can be analyzed for:
  - Accuracy in identifying `From:` and `To:`
  - Differences between base and fine-tuned model outputs
  - Impact of context (few-shot vs. zero-shot)

Dependencies
------------
Ensure the following are installed in Colab or your environment:

pip install transformers datasets accelerate bitsandbytes peft huggingface_hub

Notes
-----
- All `.txt` inputs are real ATC messages extracted from transcripts.
- The system prompt emphasizes no assumptions, verbatim message extraction, and rules for call sign positions.
- Fine-tuning was based on realistic categorization labels provided manually in `.jsonl` files.
