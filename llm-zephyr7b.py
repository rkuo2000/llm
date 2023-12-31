import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

torch.set_default_device("cuda")

model_name = "HuggingFaceH4/zephyr-7b-beta"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

results = model.generate(**inputs, max_length=200)
output = tokenizer.batch_decode(results)
print(output[0])
