import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # eğitimde kullandığın base ile aynı olmalı
ADAPTER_DIR = "./law_rag_lora/checkpoint-200"              # sende bu klasör var

# Tokenizer: base'ten al (en güvenlisi)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Base model
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,   # bf16 eğittiysen torch.bfloat16 yap
    device_map="auto",
)

# LoRA adapter bind
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()

SYSTEM_TEXT = (
    "Sen Türk hukuku alanında yardımcı bir asistansın.\n"
    "- Yalnızca verilen CONTEXT'e dayanarak cevap ver.\n"
    "- Eğer context yeterli değilse: \"Bu sorunun cevabı verilen metinde bulunmuyor.\" de.\n"
    "- Uydurma yapma.\n"
)

def build_prompt(question: str, context: str = "") -> str:
    return f"""### SYSTEM
{SYSTEM_TEXT}

### CONTEXT
{context}

### SORU
{question}

### CEVAP
"""

@torch.no_grad()
def ask(question: str, context: str = "", max_new_tokens: int = 256) -> str:
    prompt = build_prompt(question, context)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("### CEVAP", 1)[-1].strip()

# --------- TESTLER ----------
ctx = """
1475 sayılı İş Kanunu'nun 14. maddesine göre,
işçinin iş sözleşmesinin belirli şartlarla sona ermesi halinde
kıdem tazminatına hak kazanılır.
"""

print(ask("Kıdem tazminatına ne zaman hak kazanılır?", ctx))

print("\n=== RAG VAR (context ile) ===")
#ctx = "Buraya RAG'den çektiğin ilgili kanun maddesi / metin parçasını yapıştır."
#print(ask("Kıdem tazminatına hangi durumlarda hak kazanılır?", ctx))
