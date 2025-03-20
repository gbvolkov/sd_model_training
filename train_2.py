#!pip install -q transformers==4.40.0 peft==0.10.0 accelerate==0.29.0 trl==0.8.0 datasets bitsandbytes

import torch
from datasets import load_from_disk, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

import logging


DATASET_PATH = "data/datasets/sd_dataset"   # Path where the prepared dataset is saved
MODEL_NAME = "google/gemma-3-4b-it"
# Directory to save the fine-tuned model
OUTPUT_DIR = "gemma-3-4b_sdesk_lora"           

def preprocess(sample):
    """
    Preprocess a dataset sample:
    - If a system message is present, merge its content with the first user message.
    - Use the model's chat template to convert the list of messages into a formatted text string.
    """
    messages = sample["messages"]
    question = messages[2]["content"]
    context = messages[1]["content"]
    answer = messages[3]["content"]
    formatted = f"<start_of_turn>user\nAnswer this question using the context below:\nQuestion: {question}\nContext: {context}<end_of_turn>\n<start_of_turn>model\n{answer}<end_of_turn>"

    return {"text": formatted}


def load_and_split_dataset():
    """
    Load the dataset from disk and split it into training (80%) and validation (20%) sets.
    Applies preprocessing to convert the conversation messages into a single text prompt.
    """
    logging.info("Loading dataset from disk...")
    ds = load_from_disk(DATASET_PATH)
    logging.info(f"Loaded {len(ds)} examples.")
    ds = ds.train_test_split(test_size=0.2, seed=42)
    train_ds = ds["train"]
    eval_ds = ds["test"]
    logging.info("Preprocessing training dataset...")
    train_ds = train_ds.map(preprocess, remove_columns=["messages"])
    logging.info("Preprocessing validation dataset...")
    eval_ds = eval_ds.map(preprocess, remove_columns=["messages"])

    return train_ds, eval_ds

train_ds, eval_ds = load_and_split_dataset()

# 2. Model & Tokenizer Setup

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_special_tokens({"additional_special_tokens": ["<start_of_turn>", "<end_of_turn>"]})
tokenizer.pad_token = tokenizer.eos_token

from transformers import Gemma3ForCausalLM
from peft import LoraConfig, get_peft_model

model = Gemma3ForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
)

# 3. LoRA Configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["embed_tokens", "lm_head"]  # For better output quality
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Should show ~1-2% trainable params

# 4. Training Arguments
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Reduced due to no quantization
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    lr_scheduler_type="linear",
    warmup_steps=100,
    weight_decay=0.01,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    evaluation_strategy="no",
    report_to="none"
)


per_device_train_batch_size = 4
per_device_eval_batch_size = 1
gradient_accumulation_steps = 4
logging_steps = 5
learning_rate = 1e-4 # The initial learning rate for the optimizer.

max_grad_norm = 1.0
num_train_epochs=3
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
max_seq_length = 2048

training_arguments = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_strategy="no",
    eval_strategy="epoch",
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    weight_decay=0.1,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    bf16=True,
    hub_private_repo=False,
    push_to_hub=False,
    num_train_epochs=num_train_epochs,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    packing=True,
    max_seq_length=max_seq_length,
)


trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    peft_config=peft_config,
    formatting_func=lambda x: [x["text"]],  # Use our formatted text
)

# 6. Train & Save
trainer.train()

# Save adapter
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 7. Inference Example
from peft import PeftModel

merged_model = PeftModel.from_pretrained(
    model,
    "gemma-qa-lora-adapter",
    torch_dtype=torch.bfloat16
).merge_and_unload()

def generate_answer(question, context):
    prompt = f"""<start_of_turn>user
Answer this question using the context below:
Question: {question}
Context: {context}<end_of_turn>
<start_of_turn>model"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = merged_model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)


def get_context(question):
    with open('prompts/system_prompt_short.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    from AIAssistantsLib.assistants import RAGAssistantLocal, RAGAssistantMistralAI, SimpleAssistantMistralAI, SimpleAssistantYA
    from AIAssistantsLib.assistants.rag_assistants import get_retriever, KBDocumentPromptTemplate

    from langchain_core.output_parsers import StrOutputParser
    vectorestore_path = 'data/vectorstore_e5'
    user_query = "Кто такие key users?"

    assistant = RAGAssistantMistralAI(system_prompt, 
                                    vectorestore_path, 
                                    output_parser=StrOutputParser)
    reply = assistant.ask_question(user_query)
    prompt = KBDocumentPromptTemplate(-1, input_variables=["page_content", "problem_number", "actual_chunk_size"])

    context = reply['context']
    formatted_docs = [
                        prompt.format(
                            page_content=doc.page_content,
                            problem_number=doc.metadata["problem_number"],
                            actual_chunk_size=len(doc.page_content)  # Or your logic here
                        )
                        for doc in context
                    ]
    knowledge_context = "\n#EOD\n\n".join(formatted_docs[:1])
    return knowledge_context

# Test
question = "Кто такие key users?"
context = get_context(question)
answer = generate_answer(question,context)

print(answer)