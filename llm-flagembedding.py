## Setup LLM
import torch
from transformers import BitsAndBytesConfig
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from llama_index.readers import BeautifulSoupWebReader

## Read Document (url)
url = "https://www.theverge.com/2023/9/29/23895675/ai-bot-social-network-openai-meta-chatbots"
documents = BeautifulSoupWebReader().load_data([url])

## Set quantization 4-bit
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

## Select LLM model
llm = HuggingFaceLLM(
#    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    model_file="./LLM/zephyr-7b-alpha.Q4_K_M.gguf",
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
    query_wrapper_prompt=PromptTemplate("<s>[INST] {query_str} [/INST]</s>\n"),
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"quantization_config": quantization_config},
    # tokenizer_kwargs={},
    generate_kwargs={"temperature": 0.7, "top_k": 5, "top_p": 0.95, "do_sample": True},
    device_map="auto",
)

## choose Embedding model
from llama_index import ServiceContext
service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")

## Build a local VectorIndex
from llama_index import VectorStoreIndex
vector_index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = vector_index.as_query_engine(response_mode="compact")

from llama_index.response.notebook_utils import display_response

prompt = " "
while prompt!="quit" :
    prompt = input("?")
    response = query_engine.query("<s>[INST] {"+prompt+"} [/INST]</s>\n")
    display_response(response)

