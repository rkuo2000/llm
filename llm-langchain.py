import os
import chainlit as cl
from langchain.llms import CTransformers
from langchain import PromptTemplate, LLMChain

#model_name = "TheBloke/zephyr-7b-alpha"
#model_path = "./LLM/zephyr-7b-alpha.Q4_K_M.gguf"
model_path = "./LLM/llama-2-7b-chat.Q4_K_M.gguf"

config = {
    "max_new_tokens": 1024,
    "repetition_penalty": 1.1,
    "temperature": 0.2,
    "top_k": 50,
    "top_p": 0.9,
    "stream": True,
    "threads": int(os.cpu_count()/2)
}

llm_init = CTransformers(
    model = model_path,
#    model_file = model_file,
    model_type = "mistral",
    lib = "avx2",
    **config
)

print(llm_init)

template = """Question: {question}

Answer: Please refer to factual information and don't make up fictional data/information.
"""

prompt = PromptTemplate(template=template, input_variables=['question'])
llm_chain = LLMChain(prompt=prompt, llm=llm_init, verbose=True)
##cl.user_session.set("llm_chain", llm_chain) 

query = "What is the meaning of Life?"
result = llm_init(query)
print(result)

