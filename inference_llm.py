import os
import torch
from multiprocessing import freeze_support
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_MODEL  = "ytu-ce-cosmos/Turkish-Gemma-9b-v0.1"

# outputs/ altındaki eğitim klasörünü göster
# Örnek: "./outputs/turkish-gemma-9b-law-rag-qlora-20260219_2356"
ADAPTER_DIR = "./outputs/turkish-gemma-9b-law-rag-qlora-20260219_2356"

# ── Quantization (train_llm.py ile birebir aynı) ─────────────
LOAD_4BIT = True
USE_BF16  = False

# ── Generation ───────────────────────────────────────────────
# Hukuki RAG için düşük temperature önerilir → daha belirleyici, daha az uydurma
MAX_NEW_TOKENS    = 512
DO_SAMPLE         = True
TEMPERATURE       = 0.2   # 0.7 çok yaratıcı; hukuk için 0.1-0.3 arası ideal
TOP_P             = 0.9
REPETITION_PENALTY = 1.15  # tekrarlayan ifadeleri önler

# ─────────────────────────────────────────────
# SİSTEM MESAJI — train_llm.py ile BİREBİR AYNI olmalı
# ─────────────────────────────────────────────
SYSTEM_TEXT = (
    "Sen Türk hukuku alanında uzman bir yapay zeka asistanısın.\n"
    "- Yalnızca verilen CONTEXT'e dayanarak cevap ver.\n"
    "- Eğer context yeterli değilse tam olarak şunu yaz: "
    "\"Bu sorunun cevabı verilen metinde bulunmuyor.\"\n"
    "- Asla uydurma yapma, tahmin yürütme.\n"
    "- Cevabını açık, düzgün Türkçe ile yaz.\n"
)

# ─────────────────────────────────────────────
# MODEL YÜKLEME
# ─────────────────────────────────────────────
def load_model():
    """Base model + LoRA adapter yükler, (model, tokenizer) döndürür."""

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

    print("Tokenizer yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # inference için left padding

    print("Base model yükleniyor (4-bit NF4)...")
    if torch.cuda.is_available() and LOAD_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=COMPUTE_DTYPE if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager",
    )

    print(f"LoRA adapter yükleniyor: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    model.config.use_cache = True   # inference'da cache açık olmalı (eğitimde kapalıydı)

    print("✅ Model hazır!\n")
    return model, tokenizer


# ─────────────────────────────────────────────
# PROMPT OLUŞTURMA — train_llm.py ile BİREBİR AYNI format
# ─────────────────────────────────────────────
def build_prompt(tokenizer, context: str, soru: str) -> str:
    """
    Eğitimde kullanılan format_as_gemma_chat() ile birebir aynı prompt üretir.
    Tek fark: add_generation_prompt=True (modelin cevaba başlaması için).
    """
    user_content = (
        f"CONTEXT:\n{context.strip()}\n\n"
        f"SORU:\n{soru.strip()}\n\n"
        "Kural: Yalnızca CONTEXT'e dayan. "
        "Context yetmezse: \"Bu sorunun cevabı verilen metinde bulunmuyor.\""
    )

    messages = [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user",   "content": user_content},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # <start_of_turn>model eklenir → model yazmaya başlar
    )


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def generate_response(
    model,
    tokenizer,
    context: str,
    soru: str,
) -> str:
    """
    Verilen context ve soruya göre cevap üretir.

    Args:
        model:     Yüklü PeftModel
        tokenizer: Tokenizer
        context:   Hukuki metin (kanun maddesi, karar vb.)
        soru:      Sorulan soru

    Returns:
        Modelin ürettiği cevap metni
    """
    prompt = build_prompt(tokenizer, context, soru)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():   # torch.no_grad()'dan daha hızlı
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Sadece yeni üretilen tokenları decode et (prompt hariç)
    generated_ids   = outputs[0][input_length:]
    generated_text  = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text.strip()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    model, tokenizer = load_model()

    # ── Örnek 1: Context içinde cevap var ─────
    print("=" * 65)
    print("ÖRNEK 1 — Cevap context'te mevcut")
    print("=" * 65)
    ctx1 = (
        "Türk Ceza Kanunu'nun 26. maddesine göre, meşru müdafaa durumunda "
        "suç teşkil eden bir fiil işleyen kişi cezalandırılmaz. Meşru müdafaa, "
        "kendisine veya başkasına yönelik haksız bir saldırıyı defetmek amacıyla "
        "gerçekleştirilen eylemdir. Savunmanın saldırıyla orantılı olması şarttır."
    )
    s1 = "Meşru müdafaa nedir ve kişiyi cezadan kurtarır mı?"
    print(f"Context : {ctx1}\n")
    print(f"Soru    : {s1}\n")
    print(f"Cevap   : {generate_response(model, tokenizer, ctx1, s1)}\n")

    # ── Örnek 2: Context yetersiz ─────────────
    print("=" * 65)
    print("ÖRNEK 2 — Context yetersiz (modelin 'bulunamıyor' demesi beklenir)")
    print("=" * 65)
    ctx2 = "Borçlar Kanunu'nun 1. maddesi, sözleşmenin kurulmasını düzenler."
    s2   = "İdare hukukunda tam yargı davası nedir?"
    print(f"Context : {ctx2}\n")
    print(f"Soru    : {s2}\n")
    print(f"Cevap   : {generate_response(model, tokenizer, ctx2, s2)}\n")

    # ── İnteraktif mod ────────────────────────
    print("=" * 65)
    print("İNTERAKTİF MOD  |  Çıkmak için boş bırak veya 'q' yaz")
    print("=" * 65)

    while True:
        try:
            print()
            ctx = input("Context (hukuki metin) : ").strip()
            if not ctx or ctx.lower() == "q":
                break

            soru = input("Soru                  : ").strip()
            if not soru or soru.lower() == "q":
                break

            print("\nÜretiliyor...\n")
            cevap = generate_response(model, tokenizer, ctx, soru)
            print(f"Cevap : {cevap}")
            print("-" * 65)

        except KeyboardInterrupt:
            print("\n\nÇıkılıyor...")
            break
        except Exception as e:
            print(f"\nHata: {e}\n")


if __name__ == "__main__":
    freeze_support()
    main()
