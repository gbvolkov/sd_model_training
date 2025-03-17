import pandas as pd
import asyncio

import logging
import re
from nltk.tokenize import sent_tokenize
import nltk
from AIAssistantsLib.assistants import RAGAssistantLocal, RAGAssistantMistralAI, SimpleAssistantMistralAI, SimpleAssistantYA
from langchain_core.output_parsers import StrOutputParser

vectorestore_path = 'data/vectorstore_e5'
with open('prompts/system_prompt_short.txt', 'r', encoding='utf-8') as f:
    system_prompt = f.read()

assistants = []
assistants.append(RAGAssistantLocal(system_prompt, 
                                    vectorestore_path, 
                                    output_parser=StrOutputParser, 
                                    model_name="./smollm2_sdesk_lora"))

query = "Кто такие key_users?"
for assistant in assistants:
    reply = assistant.ask_question(query)
    print(reply)
