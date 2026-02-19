import os
import math
import random
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer

# =========================
# CONFIG
# =========================
MODEL_ID = "ytu-ce-cosmos/Turkish-Gemma-9b-v0.1"
CSV_PATH = "turkish_law_dataset.csv"
OUT_DIR = "./outputs/turkish-gemma-9b-v0.1-law-rag-qlora"

# 16GB için güvenli preset
MAX_SEQ_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM = 8 
EPOCHS = 2

LR = 2e-5
SEED = 42
SCORE_MIN = 2

LOAD_4BIT = True
USE_BF16 = False  # QLoRA compute dtype genelde fp16 ile daha stabil; istersen True yapabilirsin.

# CUDA fragmentation için (opsiyonel)
# Windows CMD: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Linux/macOS: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =========================
# SEED
# =========================
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def safe_str(x):
    if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
        return ""
    return str(x)

# =========================
# LOAD CSV
# =========================
df = pd.read_csv(CSV_PATH)

if "Score" in df.columns:
    df = df[df["Score"] >= SCORE_MIN]

df["soru"] = df["soru"].astype(str)
df["cevap"] = df["cevap"].astype(str)
df = df[(df["soru"].str.len() > 0) & (df["cevap"].str.len() > 0)]

dataset = Dataset.from_pandas(df, preserve_index=False)

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

SYSTEM_TEXT = (
    "Sen Türk hukuku alanında yardımcı bir asistansın.\n"
    "- Yalnızca verilen CONTEXT'e dayanarak cevap ver.\n"
    "- Eğer context yeterli değilse: \"Bu sorunun cevabı verilen metinde bulunmuyor.\" de.\n"
    "- Uydurma yapma.\n"
)

def format_as_gemma_chat(example):
    context = safe_str(example.get("context", "")).strip()
    soru = safe_str(example.get("soru", "")).strip()
    cevap = safe_str(example.get("cevap", "")).strip()
    kaynak = safe_str(example.get("kaynak", "")).strip()

    user_content = (
        f"CONTEXT:\n{context}\n\n"
        f"SORU:\n{soru}\n\n"
        "Kurallar: Sadece CONTEXT'e dayan. Context yetmezse aynen şu cümleyi yaz:\n"
        "\"Bu sorunun cevabı verilen metinde bulunmuyor.\""
    )

    messages = [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": cevap + (f"\n\nKaynak: {kaynak}" if kaynak else "")},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(format_as_gemma_chat)

dataset = dataset.remove_columns([c for c in dataset.column_names if c != "text"])

splits = dataset.train_test_split(test_size=0.02, seed=SEED)
train_ds = splits["train"]
eval_ds = splits["test"]

print("\n--- SAMPLE TRAIN TEXT ---\n")
print(train_ds[0]["text"][:800])

tok = tokenizer(train_ds[0]["text"], truncation=False)
print("\nTokens:", len(tok["input_ids"]))
print("Truncated?", len(tok["input_ids"]) > MAX_SEQ_LEN)

# =========================
# MODEL (QLoRA)
# =========================
if torch.cuda.is_available() and LOAD_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16 if not USE_BF16 else torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
else:
    bnb_config = None

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# Train için kritik
model.config.use_cache = False
model.gradient_checkpointing_enable()

# ✅ 4-bit train hazırlığı (requires_grad hatasını çözen ana adım)
if bnb_config is not None:
    model = prepare_model_for_kbit_training(model)

# =========================
# LoRA (16GB için önce attention-only öneriyorum)
# =========================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj"],  # ✅ önce dar tut
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.train()

# Hızlı kontrol (debug)
print("is_loaded_in_4bit:", getattr(model, "is_loaded_in_4bit", None))
print("any trainable:", any(p.requires_grad for p in model.parameters()))

# =========================
# TRAINING ARGS
# =========================
# transformers sürüm uyumluluğu için:
eval_kwargs = {}
try:
    eval_kwargs["evaluation_strategy"] = "steps"
except Exception:
    eval_kwargs["eval_strategy"] = "steps"

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,   # ✅ bunu ekle
    eval_accumulation_steps=1,      # ✅ (opsiyonel) eval çıktısını biriktirmesin
    prediction_loss_only=True, 
    learning_rate=LR,
    warmup_ratio=0.03,

    fp16=torch.cuda.is_available() and (not USE_BF16),
    bf16=torch.cuda.is_available() and USE_BF16,

    max_grad_norm=0.6,
    logging_steps=10,

    eval_steps=15,
    save_steps=10,
    save_total_limit=2,

    report_to=["tensorboard"],
    logging_dir=f"{OUT_DIR}/logs",

    optim="paged_adamw_8bit" if (bnb_config is not None) else "adamw_torch",

    eval_strategy = "steps"
)
print("training_args: ", training_args)
# =========================
# TRAINER
# =========================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    formatting_func=lambda x: x["text"]
)

trainer.train()
trainer.model.save_pretrained(
    OUT_DIR,
    state_dict=trainer.model.state_dict(),
    safe_serialization=True
)

tokenizer.save_pretrained(OUT_DIR)

print("\n✅ Training finished. Adapter saved to:", OUT_DIR)
print(f"tensorboard --logdir {OUT_DIR}/logs")
