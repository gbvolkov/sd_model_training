from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
OUTPUT_DIR = "smollm2_sdesk_lora"  # Directory where your adapter is saved
MERGED_DIR = "merged_model_dir"    # Directory to save the merged model

# Load the base model as usual
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16  # or torch.float16 as needed
)

# Load the adapter (LoRA weights)
model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

# Merge the LoRA adapter weights into the base model and unload the adapter
model = model.merge_and_unload()

# Save the merged model and tokenizer
model.save_pretrained(MERGED_DIR)
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(MERGED_DIR)

print(f"Merged model saved to {MERGED_DIR}")
