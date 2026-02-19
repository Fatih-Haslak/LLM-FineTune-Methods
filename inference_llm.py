import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# =========================
# CONFIG
# =========================
BASE_MODEL = "ytu-ce-cosmos/Turkish-Gemma-9b-v0.1"
ADAPTER_DIR = "./outputs/turkish-gemma-9b-v0.1-law-rag-qlora/checkpoint-100"

# Quantization ayarları (train_llm.py ile aynı)
LOAD_4BIT = True
USE_BF16 = False

# Generation parametreleri
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

# =========================
# MODEL & TOKENIZER YÜKLEME
# =========================
print("Tokenizer yükleniyor...")
# Tokenizer'ı base model'den yükle (adapter dizininde config sorunları olabilir)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Base model yükleniyor...")
# 4-bit quantization config (train_llm.py ile aynı)
if torch.cuda.is_available() and LOAD_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16 if not USE_BF16 else torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
else:
    bnb_config = None

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

print("LoRA adapter yükleniyor...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

print("✅ Model hazır!\n")

# =========================
# SİSTEM MESAJI (train_llm.py ile aynı)
# =========================
SYSTEM_TEXT = (
    "Sen Türk hukuku alanında yardımcı bir asistansın.\n"
    "- Yalnızca verilen CONTEXT'e dayanarak cevap ver.\n"
    "- Eğer context yeterli değilse: \"Bu sorunun cevabı verilen metinde bulunmuyor.\" de.\n"
    "- Uydurma yapma.\n"
)

# =========================
# INFERENCE FONKSİYONU
# =========================
def generate_response(context: str, soru: str, kaynak: str = ""):
    """
    Context ve soruya dayalı cevap üretir.
    
    Args:
        context: Verilen context metni
        soru: Sorulan soru
        kaynak: Opsiyonel kaynak bilgisi (sadece gösterim için)
    
    Returns:
        Modelin ürettiği cevap
    """
    user_content = (
        f"CONTEXT:\n{context}\n\n"
        f"SORU:\n{soru}\n\n"
        "Kurallar: Sadece CONTEXT'e dayan. Context yetmezse aynen şu cümleyi yaz:\n"
        "\"Bu sorunun cevabı verilen metinde bulunmuyor.\""
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user", "content": user_content},
    ]
    
    # Chat template uygula
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize et
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode et (sadece yeni tokenları)
    input_length = inputs["input_ids"].shape[1]
    generated_text = tokenizer.decode(
        outputs[0][input_length:], 
        skip_special_tokens=True
    )
    
    return generated_text.strip()

# =========================
# ÖRNEK KULLANIM
# =========================
if __name__ == "__main__":
    # Örnek 1
    print("=" * 60)
    print("ÖRNEK 1")
    print("=" * 60)
    context1 = """
    Türk Ceza Kanunu'nun 26. maddesine göre, meşru müdafaa durumunda 
    suç teşkil eden bir fiil işleyen kişi cezalandırılmaz. Meşru müdafaa, 
    kendisine veya başkasına yönelik haksız bir saldırıyı defetmek için 
    yapılan fiildir.
    """
    soru1 = "Meşru müdafaa nedir?"
    
    cevap1 = generate_response(context1, soru1)
    print(f"Context: {context1.strip()}")
    print(f"\nSoru: {soru1}")
    print(f"\nCevap: {cevap1}\n")
    
    # Örnek 2
    print("=" * 60)
    print("ÖRNEK 2")
    print("=" * 60)
    context2 = """
    Borçlar Kanunu'nun 1. maddesi, borç ilişkisinin temelini oluşturur.
    """
    soru2 = "Türk Ceza Kanunu'nun 5237 sayılı kanunun 1. maddesi ne diyor?"
    
    cevap2 = generate_response(context2, soru2)
    print(f"Context: {context2.strip()}")
    print(f"\nSoru: {soru2}")
    print(f"\nCevap: {cevap2}\n")
    
    # İnteraktif mod
    print("=" * 60)
    print("İNTERAKTİF MOD")
    print("=" * 60)
    print("Çıkmak için 'quit' yazın.\n")
    
    while True:
        try:
            context = input("Context: ").strip()
            if context.lower() == "quit":
                break
            
            soru = input("Soru: ").strip()
            if soru.lower() == "quit":
                break
            
            print("\nCevap üretiliyor...")
            cevap = generate_response(context, soru)
            print(f"\nCevap: {cevap}\n")
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nÇıkılıyor...")
            break
        except Exception as e:
            print(f"\nHata: {e}\n")
