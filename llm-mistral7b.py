import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

config = {
    "pad_token_id" : 2,
}

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", **config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_attention_mask=False)

results = model.generate(inputs, max_length=512)
outputs = tokenizer.batch_decode(results)
print(outputs[0])

