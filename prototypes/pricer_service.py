import modal
from modal import App, Image

# Setup - define our infrastructure with code!

app = modal.App("pricer-service")

image = Image.debian_slim().pip_install("torch", "transformers", "bitsandbytes", "accelerate", "peft")
secrets = [modal.Secret.from_name("hf-secret")]
GPU = "T4"

BASE_MODEL = "MightyOctopus/pricer-merged-model-A-v1"
FINETUNED_MODEL = "MightyOctopus/pricer-lora-ft-v3"
### Base model for tokenizer
TOK_BASE_MODEL = "meta-llama/Llama-3.1-8B"


QUESTION = "How much does this cost to the nearest dollar?"
PREFIX = "Price is $"

@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def price(description: str) -> float:
    import os
    import re
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import PeftModel

    prompt = f"{QUESTION}\n{description}\n{PREFIX}"

    # Quant Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    # Load model and tokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto"
    )

    fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)

    set_seed(42)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    attention_mask = torch.ones(inputs.shape, device="cuda")
    outputs = fine_tuned_model.generate(inputs, attention_mask=attention_mask, max_new_tokens=5, num_return_sequences=1)
    result = tokenizer.decode(outputs[0])

    contents = result.split("Price is $")[1]
    contents = contents.replace(',', '')
    match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
    return float(match.group()) if match else 0
