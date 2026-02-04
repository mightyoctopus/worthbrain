import modal
from modal import App, Volume, Image

app = modal.App("llama")
image = Image.debian_slim().pip_install("torch", "transformers", "bitsandbytes", "accelerate")
secrets = [modal.Secret.from_name("hf-secret")]
GPU = "T4"
### My own merged model used as the base model
MODEL_NAME = "MightyOctopus/pricer-merged-model-A-v1"

### Base model for tokenizer
TOK_BASE_MODEL = "meta-llama/Llama-3.1-8B"


@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def generate(prompt: str) -> str:
    import os
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

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
        device_map="auto"
    )

    set_seed(42)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    attention_mask = torch.ones(inputs.shape, device="cuda")

    outputs = base_model.generate(
        inputs=inputs,
        attention_mask=attention_mask,
        max_new_tokens=5,
        num_return_sequences=1
    )

    print(outputs)

    return tokenizer.decode(outputs[0])

