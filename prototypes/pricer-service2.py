import modal
from modal import App, Image, Volume

# Setup - define our infrastructure with code!

app = modal.App("pricer-service")

image = Image.debian_slim().pip_install("torch", "transformers", "bitsandbytes", "accelerate", "peft")
secrets = [modal.Secret.from_name("hf-secret")]
GPU = "T4"

BASE_MODEL = "MightyOctopus/pricer-merged-model-A-v1"
FINETUNED_MODEL = "MightyOctopus/pricer-lora-ft-v3"
### Base model for tokenizer
TOK_BASE_MODEL = "meta-llama/Llama-3.1-8B"

### Cache directory in the Modal image
CACHE_DIR = "/cache"

### Switch this to n (from 0) if wanting the modal server always up and running
MIN_CONTAINERS = 0

QUESTION = "How much does this cost to the nearest dollar?"
PREFIX = "Price is $"

### Create a volume for caching the model
hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)

@app.cls(
    image=image,
    secrets=secrets,
    gpu=GPU,
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    volumes={CACHE_DIR: hf_cache_volume}
)
class Pricer:

    ### @modal.enter() runs once per container and stored in-memory
    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        from peft import PeftModel

        # Quant Config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            TOK_BASE_MODEL,
            trust_remote_code=True,
            use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            trust_remote_code=True,
            device_map="auto"
        )
        self.base_model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.fine_tuned_model = PeftModel.from_pretrained(self.base_model, FINETUNED_MODEL)


    @modal.method()
    def price(self, description: str) -> float:
        import os
        import re
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        from peft import PeftModel

        set_seed(42)
        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        attention_mask = torch.ones(inputs.shape, device="cuda")
        outputs = self.fine_tuned_model.generate(inputs, attention_mask=attention_mask, max_new_tokens=5,
                                                 num_return_sequences=1)
        result = self.tokenizer.decode(outputs[0])

        contents = result.split("Price is $")[1]
        contents = contents.replace(',', '')
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        return float(match.group()) if match else 0