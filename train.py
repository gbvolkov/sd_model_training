#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fine_tune_service_desk.py

This module fine-tunes the HuggingFaceTB/SmolLM2-1.7B-Instruct model on a custom Service Desk Q&A dataset using LoRA.
It performs the following steps:

1. Loads the dataset from disk (assumed to be saved under "data/datasets/sd_dataset")
2. Splits the dataset into training (80%) and validation (20%) sets.
3. Preprocesses each example by merging a system message (if present) into the first user message and formatting the conversation using the model's chat template.
4. Loads the tokenizer and adds any necessary special tokens (e.g., <pad>, <eos>).
5. Configures the model with LoRA adapters.
   - Option A: High-end hardware (full precision/bf16)
   - Option B: Limited GPU (4-bit quantization via BitsAndBytes, commented out – enable if needed)
6. Sets up training hyperparameters and initializes the Trainer.
7. Fine-tunes the model and evaluates it.
8. Saves the fine-tuned model and tokenizer.
9. Performs an inference test on a sample service desk query.

Make sure all required libraries are installed:
    pip install transformers datasets peft torch bitsandbytes

Adjust the flags and hyperparameters as necessary for your hardware and dataset.
"""

import os
import random
import time
import math
import logging

import torch
from datasets import load_from_disk, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

# Set up logging
logging.basicConfig(level=logging.INFO)

with open('hf_token.txt', 'r') as f:
    os.environ['HF_TOKEN'] = f.read()    


# Global constants
DATASET_PATH = "data/datasets/sd_dataset"   # Path where the prepared dataset is saved

#MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
#OUTPUT_DIR = "smollm2_sdesk_lora"           # Directory to save the fine-tuned model
#MODEL_NAME = "google/gemma-2-2b-it"
#OUTPUT_DIR = "gemma-2_sdesk_lora"           # Directory to save the fine-tuned model
#MODEL_NAME = "ministral/Ministral-3b-instruct"
#OUTPUT_DIR = "Ministral-3b_sdesk_lora"           # Directory to save the fine-tuned model
#MODEL_NAME = "mistralai/Ministral-8B-Instruct-2410"
#OUTPUT_DIR = "Ministral-8B_sdesk_lora"           # Directory to save the fine-tuned model
MODEL_NAME = "google/gemma-3-4b-it"
OUTPUT_DIR = "gemma-3-4b_sdesk_lora"           # Directory to save the fine-tuned model

def preprocess(sample):
    """
    Preprocess a dataset sample:
    - If a system message is present, merge its content with the first user message.
    - Use the model's chat template to convert the list of messages into a formatted text string.
    """
    messages = sample["messages"]
    if messages and messages[0]["role"] == "system":
        system_content = messages[0]["content"]
        if len(messages) > 2 and messages[1]["role"] == "user" and messages[2]["role"] == "user":
            #messages[2]["content"] = system_content + "\n\nQuestion:\n" + messages[2]["content"] + "\n\n" + messages[1]["content"]
            messages[2]["content"] = "\n\nQuestion:\n" + messages[2]["content"] + "\n\n" + messages[1]["content"] + "#EOC\n\n"
            messages.pop(0)  # Remove the system message after merging
        messages.pop(0)  # Remove the system message after merging
    # Format the conversation using the tokenizer's chat template (non-tokenized text)
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
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
    val_ds = ds["test"]
    logging.info("Preprocessing training dataset...")
    train_ds = train_ds.map(preprocess, remove_columns=["messages"])
    logging.info("Preprocessing validation dataset...")
    val_ds = val_ds.map(preprocess, remove_columns=["messages"])

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)

    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)

    #train_ds = train_ds.select(range(5))
    #val_ds = val_ds.select(range(2))

    return train_ds, val_ds

def configure_tokenizer(model_name):
    """
    Load the tokenizer for the model and add any necessary special tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if "eos_token" not in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({"eos_token": "<eos>"})
    return tokenizer

def configure_model(model_name, tokenizer, use_quantization=False):
    """
    Load the base model, resize token embeddings, and then attach LoRA adapters.
    
    Parameters:
        use_quantization (bool): Set True for limited GPU environments (4-bit quantization).
                                 Otherwise, use full precision (bf16/float16).
    """
    if not use_quantization:
        from transformers import AutoTokenizer, Gemma3ForCausalLM

        model = Gemma3ForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    else:
        # Option B: Limited GPU setup with 4-bit quantization (requires bitsandbytes)
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config
        )
    
    # Resize embeddings to account for any new tokens added to the tokenizer.
    # Do this BEFORE wrapping the model with LoRA.
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA parameters (common for both setups)
    #lora_rank = 16
    lora_rank = 32
    lora_alpha = 64
    lora_dropout = 0.05
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "lm_head", "embed_tokens"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
        #task_type=TaskType.QUESTION_ANS
    )
    # Wrap the model with LoRA adapters
    model = get_peft_model(model, peft_config)
    logging.info("LoRA adapters added to the model.")
    
    return model

def train_model(train_ds, val_ds, model, tokenizer):
    """
    Fine-tunes the model using the Hugging Face Trainer API.
    Configures hyperparameters, sets up a data collator, trains the model,
    evaluates on the validation set, and saves the model and tokenizer.
    """
    epochs = 2
    train_batch_size = 1
    eval_batch_size = 1
    grad_accum_steps = 4
    learning_rate = 1e-4
    logging_steps = 5
    warmup_ratio = 0.1

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        evaluation_strategy="epoch",
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        fp16=not torch.cuda.is_bf16_supported(),  # use FP16 if BF16 is not available
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        save_strategy="no",
        dataloader_drop_last=True,
        #remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    logging.info("Starting training...")
    trainer.train()
    logging.info("Training complete.")

    # Evaluate the model on the validation set
    eval_results = trainer.evaluate()
    val_loss = eval_results.get("eval_loss", None)
    if val_loss is not None:
        logging.info(f"Validation loss: {val_loss:.3f}")
        try:
            perplexity = math.exp(val_loss)
            logging.info(f"Validation perplexity: {perplexity:.2f}")
        except OverflowError:
            logging.info("Perplexity calculation overflowed.")

    # Save the fine-tuned model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    logging.info(f"Model and tokenizer saved to {OUTPUT_DIR}")
    return trainer

def inference_test(model, tokenizer):
    """
    Performs an inference test on the fine-tuned model using a sample service desk query.
    Formats a conversation with system instructions, context, and a query,
    then generates and prints the assistant's response.
    """
    model.eval()
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

    #system_prompt = "You are an AI assistant for a service desk, tasked with helping service desk specialists find solutions for problems."
    
    #knowledge_context = (
    #    "Context: ### Problem\nUser cannot connect to VPN.\n"
    #    "### Solution\nRestart the VPN client and check credentials.\n#EOD\n\n"
    #)
    #user_query = "The user is unable to connect to the corporate VPN. How should I troubleshoot this issue?"

    messages = [
        #{"role": "system", "content": prompt},
        #{"role": "user", "content": f"{prompt}\n\nQuestion:\n{query}\n\n{context}"},
        {"role": "user", "content": f"\n\nQuestion:\n{user_query}\n\nContext: {knowledge_context}#EOC\n\n"},
        #{"role": "user", "content": context},
        #{"role": "user", "content": query}
        #{"role": "assistant", "content": ""},
    ]


    # Format the messages using the model's chat template and add the generation prompt indicator
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.4,
        top_p=0.9
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response_parts = response.split("<end_of_turn>")
    assistant_response = response_parts[-1].strip() if response_parts else response.strip()

    logging.info("Inference Test Response:")
    print(assistant_response)

def main():
    global tokenizer  # make tokenizer available to the preprocess function
    tokenizer = configure_tokenizer(MODEL_NAME)
    train_ds, val_ds = load_and_split_dataset()

    # Set use_quantization to True if you need to run on a limited GPU
    use_quantization = False  # Change to True for limited GPU setups
    model = configure_model(MODEL_NAME, tokenizer, use_quantization=use_quantization)

    trainer = train_model(train_ds, val_ds, model, tokenizer)
    inference_test(model, tokenizer)

if __name__ == "__main__":
    main()
