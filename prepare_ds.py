from AIAssistantsLib.assistants.rag_utils.rag_utils import load_vectorstore
from AIAssistantsLib.assistants.rag_assistants import get_retriever, KBDocumentPromptTemplate
from AIAssistantsLib.assistants import RAGAssistantLocal, RAGAssistantMistralAI, SimpleAssistantMistralAI, SimpleAssistantYA
import AIAssistantsLib.config as config

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer

from langchain_core.output_parsers import StrOutputParser

import os
import time

import logging
import re

import pandas as pd



if __name__ == '__main__':

    with open("context.txt", "r", encoding="utf-8") as f:
        context = f.read()

    with open("prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    assistant_prompt = """You are an AI assistant for a service desk, tasked with helping service desk specialists find solution for problems. Use the following process to assist users:

A user will present their problem. 
You will be provided with a context (Contex:), containing information about various IT problems and their solutions.
Analyze the user's problem and search the context for relevant entries. Consider affected systems, problem descriptions, and solution steps.

Remember to be professional, clear, and thorough in your responses. If multiple solutions are possible, present the most appropriate one first, followed by alternatives if necessary.

"""

    def cleanup_text(text):
        return re.sub(r'##IMAGE##\s+\S+\.(png|jpg|jpeg|gif)', '', text)

    from langchain_core.prompts import ChatPromptTemplate, StringPromptTemplate

    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    question_gen = SimpleAssistantMistralAI(system_prompt)

    def summarise_text(text):
        brun = True
        text = cleanup_text(text)
        input_text = f"Сгенерируй вопрос на основе следующего контекста: {text}"
        while brun:
            try:
                question = question_gen.ask_question(input_text)
                brun = False
            except:
                time.sleep(2)
                brun = True

        return question

    input_text = f"Сгенерируй вопрос на основе следующего контекста: {context}"

    vectorestore_path = 'data/vectorstore_e5'

    with open('prompts/system_prompt_markdown_3.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    assistants = []
    vectorstore = load_vectorstore(vectorestore_path, config.EMBEDDING_MODEL)
    retriever = get_retriever(vectorestore_path)
    assistants.append(RAGAssistantMistralAI(system_prompt, vectorestore_path, output_parser=StrOutputParser))

    query = ''
    kb_df = pd.read_csv('kb.csv')
    kb_df = kb_df.sample(2)

    training_set = []

    prompt = KBDocumentPromptTemplate(-1, input_variables=["page_content", "problem_number", "actual_chunk_size"])

    for index, row in kb_df.iterrows():
        query = summarise_text(row['refs'])

        for assistant in assistants:
            brun = True
            while brun:
                try:
                    reply = assistant.ask_question(query)
                    context = reply['context']
                    answer = reply['answer']
                    question = reply['input']
                    formatted_docs = [
                        prompt.format(
                            page_content=doc.page_content,
                            problem_number=doc.metadata["problem_number"],
                            actual_chunk_size=len(doc.page_content)  # Or your logic here
                        )
                        for doc in context
                    ]
                    joined_docs = "\n#EOD\n\n".join(formatted_docs)
                    train_entry = {
                        "messages": [
                            {"role": "system", "content": assistant_prompt},
                            {"role": "user", "content": f"Context: {joined_docs}"},
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                        ]
                    }
                    training_set.append(train_entry)
                    brun = False
                except Exception as e:
                    logging.error(f'Error: {str(e)}')
                    time.sleep(2)
        time.sleep(0.1)
    from datasets import Dataset

    training_ds = Dataset.from_list(training_set)
    save_path = "data/datasets"
    os.makedirs(save_path, exist_ok=True)
    training_ds.save_to_disk(os.path.join(save_path, "sd_dataset"))

    #training started
    #model_name = "mistralai/Ministral-8B-Instruct-2410"
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    device_map="auto",
    # Uncomment below for 4-bit quantization if needed
    # load_in_4bit=True,
    # quantization_config=BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_quant_type="nf4"
    # )
)
