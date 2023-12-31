import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

query = "what is the meaning of Life?"
inputs = tokenizer(query, return_tensors="pt", return_attention_mask=False)

results = model.generate(**inputs, max_length=200)
output = tokenizer.batch_decode(results)
print(output[0])
