import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

query = "what is the meaning of Life?"

inputs = tokenizer(query)
results = model.generate(inputs, max_length=200)

output = tokenizer.batch_decode(results)
print(output[0])
