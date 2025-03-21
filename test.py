import pandas as pd
import asyncio

import logging
import re
from nltk.tokenize import sent_tokenize
import nltk
from AIAssistantsLib.assistants import RAGAssistantLocal, RAGAssistantMistralAI, SimpleAssistantMistralAI, SimpleAssistantYA
from AIAssistantsLib.assistants.rag_assistants import get_retriever, KBDocumentPromptTemplate

from langchain_core.output_parsers import StrOutputParser

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma3ForCausalLM

import torch
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True"

MODEL_PATH = "./merged_model_dir"

#MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
#OUTPUT_DIR = "smollm2_sdesk_lora"  # Directory where your adapter is saved
#MODEL_NAME = "google/gemma-2-2b-it"
#OUTPUT_DIR = "gemma-2_sdesk_lora"           # Directory to save the fine-tuned model
#MODEL_NAME = "ministral/Ministral-3b-instruct"
#OUTPUT_DIR = "Ministral-3b_sdesk_lora"           # Directory to save the fine-tuned model
#MODEL_NAME = "mistralai/Ministral-8B-Instruct-2410"
#OUTPUT_DIR = "Ministral-8B_sdesk_lora"           # Directory to save the fine-tuned model
MODEL_NAME = "google/gemma-3-4b-it"
OUTPUT_DIR = "gemma-3-4b_sdesk_lora"           # Directory to save the fine-tuned model


#MERGED_DIR = "merged_model_dir"    # Directory to save the merged model

# Load the base model as usual


if MODEL_NAME.startswith("google/gemma"):
    base_model = Gemma3ForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        attn_implementation='eager',
        #quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        #quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )


# Load the adapter (LoRA weights)
model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)


#model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
#tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def inference_test(model, tokenizer, prompt, context, query):
    """
    Performs an inference test on the fine-tuned model using a sample service desk query.
    Formats a conversation with system instructions, context, and a query,
    then generates and prints the assistant's response.
    """
    system_prompt = "You are an AI assistant for a service desk, tasked with helping service desk specialists find solutions for problems."
    prompt = "Analyze Context and answer usr's question based on context."
    knowledge_context = (
        "Context: ### Problem\nUser cannot connect to VPN.\n"
        "### Solution\nRestart the VPN client and check credentials.\n#EOD\n\n"
    )
    user_query = "The user is unable to connect to the corporate VPN. How should I troubleshoot this issue?"

    #(prompt, context, query) = (system_prompt, knowledge_context, user_query)
    model.eval()

    messages = [
        {"role": "system", "content": prompt},
        #{"role": "user", "content": f"{prompt}\n\nQuestion:\n{query}\n\n{context}"},
        {"role": "user", "content": f"\n\nQuestion:\n{query}\n\n{context}#EOC\n\n"},
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
    response_parts = response.split("#EOC\n\n", 1) 
    #qstr: str = ""
    #qstr.split

    assistant_response = response_parts[-1].strip() if response_parts else response.strip()

    return assistant_response


vectorestore_path = 'data/vectorstore_e5'
with open('prompts/system_prompt_short.txt', 'r', encoding='utf-8') as f:
    system_prompt = f.read()

assistants = []
assistants.append(RAGAssistantMistralAI(system_prompt, 
                                    vectorestore_path, 
                                    output_parser=StrOutputParser)) 
                                    #,model_name=MODEL_PATH))

query = "Кто такие key_users?"

#query = "МКУ просит удалить планирование"
for assistant in assistants:
    reply = assistant.ask_question(query)
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
    #knowledge_context = "Context: " + "\n#EOD\n\n".join(formatted_docs)
    knowledge_context = "Context: " + "\n#EOD\n\n".join(formatted_docs[:1])
    user_query = query
    #system_prompt = "Plese use Context to answer user's question."
    system_prompt = """You are an AI assistant for a service desk, tasked with helping service desk specialists find solution for problems. Use the following process to assist users:

A user will present their problem. 
You will be provided with a context (Contex:), containing information about various IT problems and their solutions.
Analyze the user's problem and search the context for relevant entries. Consider affected systems, problem descriptions, and solution steps.

Remember to be professional, clear, and thorough in your responses. If multiple solutions are possible, present the most appropriate one first, followed by alternatives if necessary.

"""

    """
    system_prompt = "You are an AI assistant for a service desk, tasked with helping service desk specialists find solutions for problems."
    knowledge_context = (
        "Context: ### Problem\nUser cannot connect to VPN.\n"
        "### Solution\nRestart the VPN client and check credentials.\n#EOD\n\n"
    )
    user_query = "The user is unable to connect to the corporate VPN. How should I troubleshoot this issue?"
    """
    model_response = inference_test(model, tokenizer, system_prompt, knowledge_context, user_query)
    print(model_response)