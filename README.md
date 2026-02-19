# Turkish Law RAG — QLoRA Fine-Tuning

Türk hukuku alanında **RAG (Retrieval-Augmented Generation)** destekli soru-cevap sistemi için `ytu-ce-cosmos/Turkish-Gemma-9b-v0.1` modelinin **QLoRA** yöntemiyle ince ayar (fine-tuning) yapılması.

---

## Proje Özeti

Model, verilen bir **context** (kanun maddesi, mahkeme kararı vb.) ve **soru** çiftine dayanarak yalnızca o metinden cevap üretir. Context dışına çıkmaz, uydurma yapmaz.

```
CONTEXT: Türk Ceza Kanunu'nun 26. maddesine göre...
SORU:    Meşru müdafaa nedir?
CEVAP:   Meşru müdafaa, kendisine veya başkasına yönelik haksız...
```

---

## Dosya Yapısı

```
fine-tune/
│
├── train_llm.py                    # Ana eğitim scripti (QLoRA)
├── inference_llm.py                # Eğitilmiş modeli çalıştırma
├── turkish_law_dataset.csv         # Eğitim verisi (soru, cevap, context, kaynak)
├── training_parameters_guide.html  # Her parametrenin açıklandığı dokümantasyon
│
└── outputs/
    └── turkish-gemma-9b-law-rag-qlora-YYYYMMDD_HHMM/
        ├── adapter_model.safetensors
        ├── adapter_config.json
        ├── tokenizer.json
        └── logs/   ← TensorBoard logları
```

---

## Model & Yöntem

| Özellik | Değer |
|---|---|
| Base model | `ytu-ce-cosmos/Turkish-Gemma-9b-v0.1` |
| Fine-tuning yöntemi | QLoRA (4-bit NF4 + LoRA) |
| LoRA rank / alpha | 16 / 32 |
| LoRA hedef katmanlar | q, k, v, o, gate, up, down proj (7 katman) |
| Max sequence length | 512 token |
| Efektif batch size | 16 (batch=1 × grad_accum=16) |
| Learning rate | 2e-4 (cosine scheduler) |
| Epochs | 3 (early stopping ile) |
| GPU gereksinimleri | Minimum 16 GB VRAM (RTX 3090/4090) |

---

## Kurulum

```bash
pip install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.57.6 trl==0.24.0 peft==0.17.1
pip install bitsandbytes==0.48.2 accelerate==1.10.1
pip install datasets pandas numpy
```

> **Windows kullanıcıları:** `bitsandbytes-windows` paketi gerekebilir.

---

## Eğitim

### Dataset Formatı

`turkish_law_dataset.csv` dosyası şu sütunları içermelidir:

| Sütun | Açıklama | Zorunlu |
|---|---|---|
| `soru` | Hukuki soru | ✅ |
| `cevap` | Cevap metni | ✅ |
| `context` | İlgili kanun/karar metni | ✅ |
| `kaynak` | Kaynak bilgisi (kanun no vb.) | ➖ |
| `Score` | Kalite puanı (0-4) | ➖ |

### Eğitimi Başlat

```bash
python train_llm.py
```

Eğitim tamamlandığında adapter `./outputs/turkish-gemma-9b-law-rag-qlora-YYYYMMDD_HHMM/` klasörüne kaydedilir.

### TensorBoard ile İzleme

```bash
tensorboard --logdir ./outputs/turkish-gemma-9b-law-rag-qlora-YYYYMMDD_HHMM/logs
```

---

## Inference

### Script ile Çalıştırma

`inference_llm.py` içindeki `ADAPTER_DIR` değişkenini eğitim çıktı klasörüne göre güncelle:

```python
BASE_MODEL  = "ytu-ce-cosmos/Turkish-Gemma-9b-v0.1"
ADAPTER_DIR = "./outputs/turkish-gemma-9b-law-rag-qlora-YYYYMMDD_HHMM"
```

```bash
python inference_llm.py
```

### Python'dan Import

```python
from inference_llm import generate_response

context = """
Türk Ceza Kanunu'nun 26. maddesine göre, meşru müdafaa durumunda
suç teşkil eden bir fiil işleyen kişi cezalandırılmaz.
"""

cevap = generate_response(
    context=context,
    soru="Meşru müdafaa nedir?"
)
print(cevap)
```

### İnteraktif Mod

Script çalıştırıldığında örnekler gösterildikten sonra terminal üzerinden interaktif soru-cevap yapılabilir. Çıkmak için `quit` yaz.

---

## Eğitim Stratejisi

### QLoRA Neden?

9B parametreli model fp16'da ~18 GB VRAM ister. QLoRA ile:

- 4-bit NF4 quantization → model ~4.5 GB
- LoRA adaptörleri → ~50M eğitilebilir parametre (%0.5)
- Double quantization → ek ~0.4 GB tasarruf
- Gradient checkpointing → aktivasyon belleği ~%40 azalır

**Toplam: ~6 GB VRAM ile 9B model eğitimi**

### Veri Pipeline

```
CSV yükle
  → Score ≥ 2 filtresi
  → Boş / kısa örnekleri çıkar (len > 10)
  → Kopya kaldır
  → Gemma chat template formatına çevir
  → %98 train / %2 eval bölümle
```

### Erken Durdurma

5 ardışık eval adımında `eval_loss` iyileşmezse eğitim otomatik durur. En iyi checkpoint'e geri dönülür.

---

## Önemli Notlar

- **Windows'ta** `dataloader_num_workers=0` olarak kalmalıdır.
- **RTX GPU** kullanıyorsan `USE_BF16=False` bırak (BF16 desteği yok).
- **A100/H100** kullanıyorsan `USE_BF16=True` ve `LOAD_4BIT=False` yapabilirsin.
- Adapter checkpoint'i base model olmadan çalışmaz; inference sırasında her ikisi de gereklidir.
- Parametre açıklamaları için `training_parameters_guide.html` dosyasını tarayıcıda aç.

---

## Bağımlılıklar

| Paket | Versiyon |
|---|---|
| Python | 3.9+ |
| torch | 2.5.0 |
| transformers | 4.57.6 |
| trl | 0.24.0 |
| peft | 0.17.1 |
| bitsandbytes | 0.48.2 |
| accelerate | 1.10.1 |
| datasets | 4.5.0 |
| pandas | 2.2.3 |
