import os
import math
import random
import logging
import multiprocessing
from multiprocessing import freeze_support
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CUDA & OS ORTAM DEĞİŞKENLERİ
# ─────────────────────────────────────────────
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_ID  = "ytu-ce-cosmos/Turkish-Gemma-9b-v0.1"
CSV_PATH  = "turkish_law_dataset.csv"
RUN_NAME  = f"turkish-gemma-9b-law-rag-qlora-{datetime.now().strftime('%Y%m%d_%H%M')}"
OUT_DIR   = f"./outputs/{RUN_NAME}"

# ── Donanım ──────────────────────────────────
LOAD_4BIT = True
USE_BF16  = False           # A100/H100'de True; 16 GB RTX için False

# ── Sequence ─────────────────────────────────
MAX_SEQ_LEN = 512

# ── Batch / Gradient ─────────────────────────
BATCH_SIZE = 1
GRAD_ACCUM = 16             # efektif batch = 16

# ── Optimizer / LR ───────────────────────────
EPOCHS        = 3
LR            = 2e-4
LR_SCHEDULER  = "cosine"
WARMUP_RATIO  = 0.05
MAX_GRAD_NORM = 1.0

# ── Kayıt & Eval ─────────────────────────────
LOGGING_STEPS           = 10
EVAL_STEPS              = 50
SAVE_STEPS              = 50
SAVE_TOTAL_LIMIT        = 3
EARLY_STOPPING_PATIENCE = 5

# ── Veri ─────────────────────────────────────
SEED      = 42
SCORE_MIN = 2
TEST_SIZE = 0.02

# ── LoRA ─────────────────────────────────────
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ─────────────────────────────────────────────
# FONKSİYONLAR
# ─────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def log_dataset_stats(df: pd.DataFrame, label: str):
    log.info(f"[{label}] Satır sayısı  : {len(df)}")
    log.info(f"[{label}] Soru uzunluğu : ort={df['soru'].str.len().mean():.0f} "
             f"/ max={df['soru'].str.len().max()}")
    log.info(f"[{label}] Cevap uzunluğu: ort={df['cevap'].str.len().mean():.0f} "
             f"/ max={df['cevap'].str.len().max()}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    set_seed(SEED)

    # ── Veri ──────────────────────────────────
    log.info(f"CSV yükleniyor: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    if "Score" in df.columns:
        before = len(df)
        df = df[df["Score"] >= SCORE_MIN]
        log.info(f"Score filtresi: {before} → {len(df)} satır")

    for col in ["soru", "cevap"]:
        df[col] = df[col].apply(safe_str)

    df = df[(df["soru"].str.len() > 10) & (df["cevap"].str.len() > 10)]

    before = len(df)
    df = df.drop_duplicates(subset=["soru", "cevap"])
    log.info(f"Kopya kaldırma: {before} → {len(df)} satır")

    log_dataset_stats(df, "final")
    dataset = Dataset.from_pandas(df, preserve_index=False)

    # ── Tokenizer ─────────────────────────────
    log.info(f"Tokenizer yükleniyor: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    SYSTEM_TEXT = (
        "Sen Türk hukuku alanında uzman bir yapay zeka asistanısın.\n"
        "- Yalnızca verilen CONTEXT'e dayanarak cevap ver.\n"
        "- Eğer context yeterli değilse tam olarak şunu yaz: "
        "\"Bu sorunun cevabı verilen metinde bulunmuyor.\"\n"
        "- Asla uydurma yapma, tahmin yürütme.\n"
        "- Cevabını açık, düzgün Türkçe ile yaz.\n"
    )

    def format_as_gemma_chat(example):
        context = safe_str(example.get("context", ""))
        soru    = safe_str(example.get("soru", ""))
        cevap   = safe_str(example.get("cevap", ""))
        kaynak  = safe_str(example.get("kaynak", ""))

        user_content = (
            f"CONTEXT:\n{context}\n\n"
            f"SORU:\n{soru}\n\n"
            "Kural: Yalnızca CONTEXT'e dayan. "
            "Context yetmezse: \"Bu sorunun cevabı verilen metinde bulunmuyor.\""
        )

        assistant_content = cevap + (f"\n\nKaynak: {kaynak}" if kaynak else "")

        messages = [
            {"role": "system",    "content": SYSTEM_TEXT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )}

    log.info("Chat template uygulanıyor...")
    dataset = dataset.map(format_as_gemma_chat, num_proc=1)
    dataset = dataset.remove_columns([c for c in dataset.column_names if c != "text"])

    splits   = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    train_ds = splits["train"]
    eval_ds  = splits["test"]
    log.info(f"Eğitim: {len(train_ds)} | Eval: {len(eval_ds)}")

    sample_len = len(tokenizer(train_ds[0]["text"], truncation=False)["input_ids"])
    log.info(f"Örnek token uzunluğu: {sample_len} (MAX_SEQ_LEN={MAX_SEQ_LEN})")
    if sample_len > MAX_SEQ_LEN:
        log.warning("Örnek MAX_SEQ_LEN'den uzun, kırpılacak!")

    # ── Model — QLoRA ─────────────────────────
    COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

    if torch.cuda.is_available() and LOAD_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=True,
        )
        log.info("4-bit NF4 quantization aktif")
    else:
        bnb_config = None
        log.info("Quantization kapalı")

    log.info(f"Model yükleniyor: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=COMPUTE_DTYPE if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager",
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    # ── LoRA ──────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
        init_lora_weights="gaussian",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f"Eğitilebilir: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Training Arguments (SFTConfig) ───────────────────
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=OUT_DIR,
        run_name=RUN_NAME,
        num_train_epochs=EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,

        fp16=torch.cuda.is_available() and not USE_BF16,
        bf16=True if (torch.cuda.is_available() and USE_BF16) else None,

        optim="paged_adamw_8bit" if bnb_config is not None else "adamw_torch",
        learning_rate=LR,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        weight_decay=0.01,

        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        eval_accumulation_steps=4,
        prediction_loss_only=True,

        logging_dir=f"{OUT_DIR}/logs",
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        report_to=["tensorboard"],

        group_by_length=True,
        dataloader_num_workers=0,   # Windows için 0
        dataloader_pin_memory=True,

        seed=SEED,
        data_seed=SEED,

        # SFTConfig'e özgü parametreler (TRL 0.24.0)
        max_length=MAX_SEQ_LEN,       # eski adı: max_seq_length
        dataset_text_field="text",
    )

    # ── Trainer ───────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=1e-4,
            )
        ],
    )

    # ── Eğitim ────────────────────────────────
    log.info("=" * 60)
    log.info("EĞİTİM BAŞLIYOR")
    log.info(f"  Model      : {MODEL_ID}")
    log.info(f"  LoRA r/α   : {LORA_R} / {LORA_ALPHA}")
    log.info(f"  Seq Len    : {MAX_SEQ_LEN}")
    log.info(f"  Epochs     : {EPOCHS}")
    log.info(f"  LR         : {LR}  ({LR_SCHEDULER})")
    log.info(f"  Eff. Batch : {BATCH_SIZE * GRAD_ACCUM}")
    log.info(f"  Output     : {OUT_DIR}")
    log.info("=" * 60)

    train_result = trainer.train()

    # ── Kaydetme ──────────────────────────────
    log.info("Adapter kaydediliyor...")
    trainer.model.save_pretrained(OUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUT_DIR)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    log.info("=" * 60)
    log.info("✅ Eğitim tamamlandı.")
    log.info(f"   Adapter     : {OUT_DIR}")
    log.info(f"   TensorBoard : tensorboard --logdir {OUT_DIR}/logs")
    log.info("=" * 60)


if __name__ == "__main__":
    freeze_support()   # Windows'ta spawn/fork için zorunlu
    main()
