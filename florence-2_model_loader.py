import torch
from transformers import AutoModelForCausalLM, AutoProcessor

model_name = "microsoft/Florence-2-large"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading processor from {model_name}")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
print("Processor loaded successfully")

print(f"Loading model from {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch_dtype, trust_remote_code=True
).to(device)
print("Model loaded successfully")

print("Model and processor are ready for use")
