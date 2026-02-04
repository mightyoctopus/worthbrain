import modal
from modal import App, Volume, Image

app = modal.App("llama")
image = Image.debian_slim().pip_install("torch", "transformers", "bitsandbytes", "accelerate", "peft")
secrets = [modal.Secret.from_name("hf-secret")]
GPU = "T4"
### My own merged model used as the base model
MODEL_NAME = "MightyOctopus/pricer-merged-model-A-v1"

FINE_TUNED_MODEL = "MightyOctopus/pricer-lora-ft-v3"

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

    prompt = f"{QUESTION}\n\n{description}\n{PREFIX}"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(TOK_BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    fine_tuned_model = PeftModel.from_pretrained(
        base_model,
        FINE_TUNED_MODEL
    )

    set_seed(42)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    attention_mask = torch.ones(inputs.shape, device="cuda")
    outputs = fine_tuned_model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=5,
        num_return_sequences=1
    )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    contents = decoded_output.split("Price is $")[1]
    contents = contents.replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", contents)

    return float(match.group()) if match else 0.0