# Дообучение Whisper для радиоэфира

Полное руководство по адаптации Whisper под специфичный домен (экстренные службы, радиосвязь, русский язык).

## Оглавление

1. [Подходы без дообучения](#1-подходы-без-дообучения)
2. [LoRA fine-tuning (рекомендуется)](#2-lora-fine-tuning-рекомендуется)
3. [Полный fine-tune](#3-полный-fine-tune)
4. [Подготовка данных](#4-подготовка-данных)
5. [Аугментация под радиоусловия](#5-аугментация-под-радиоусловия)
6. [Требования к железу](#6-требования-к-железу)
7. [Конвертация для продакшна](#7-конвертация-для-продакшна)
8. [Альтернативные подходы](#8-альтернативные-подходы)
9. [Ресурсы и ссылки](#9-ресурсы-и-ссылки)

---

## 1. Подходы без дообучения

### Prompt engineering

Самый быстрый старт. Whisper принимает `initial_prompt` — до 224 токенов доменной лексики. Модель подстраивает стиль и терминологию под промпт.

```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda")

segments, _ = model.transcribe(
    "radio.wav",
    language="ru",
    initial_prompt="Центральная, говорит Сокол-один, принял вызов. "
                   "Выезжаем на Ленина двадцать три. Десять-семьдесят пять. "
                   "БПЛА, МЧС, диспетчер, караул, пожарный расчёт."
)
```

**Правила:**
- Встраивать термины в естественные предложения (не списком)
- Длинные промпты надёжнее коротких
- Whisper эмулирует *стиль* промпта, а не следует инструкциям

**Ограничения:** адаптирует только лексику/стиль, не акустическую модель.

### Через HuggingFace Transformers

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")

result = pipe(
    "audio.wav",
    generate_kwargs={
        "language": "russian",
        "prompt_ids": processor.tokenizer.encode(
            "ЦОД, БПЛА, МЧС, позывной Сокол, диспетчер, караул"
        ),
    }
)
```

### Vocabulary boosting (логит-байасинг)

Не поддерживается нативно, но можно через custom LogitsProcessor:

```python
from transformers import LogitsProcessor

class VocabularyBoostProcessor(LogitsProcessor):
    def __init__(self, boost_token_ids, boost_factor=2.0):
        self.boost_token_ids = boost_token_ids
        self.boost_factor = boost_factor

    def __call__(self, input_ids, scores):
        for token_id in self.boost_token_ids:
            scores[:, token_id] *= self.boost_factor
        return scores
```

---

## 2. LoRA fine-tuning (рекомендуется)

Обучается ~1% параметров модели. Требует **8–10 GB VRAM** вместо 40+ GB. Размер адаптера ~63–80 МБ вместо 6.7 ГБ полной модели.

### Полный пример

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset, Audio
import torch

# --- Загрузка модели в INT8 ---
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3",
    quantization_config=quantization_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

# --- LoRA ---
lora_config = LoraConfig(
    r=32,                    # ранг (8–64, 32 — хороший баланс)
    lora_alpha=64,           # обычно 2*r
    target_modules=[         # какие слои адаптировать
        "q_proj", "v_proj",  # минимум
        "k_proj", "out_proj" # рекомендуется для лучшего качества
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: ~15M || all params: ~1.55B || trainable%: ~1%

# --- Процессор ---
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-large-v3",
    language="russian",
    task="transcribe",
)

# --- Данные ---
dataset = load_dataset("google/fleurs", "ru_ru")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names)

# --- Параметры обучения ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-radio-ru-lora",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,         # для LoRA lr выше (vs 1e-5 для full)
    warmup_steps=500,
    max_steps=5000,
    fp16=True,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    remove_unused_columns=False,
    report_to=["tensorboard"],
)

# --- Обучение ---
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor.feature_extractor,
)

trainer.train()
```

### Выбор target_modules

| Конфигурация | Модули | Параметры | Когда использовать |
|---|---|---|---|
| Минимальная | `q_proj`, `v_proj` | ~8M | Быстрые эксперименты |
| Рекомендуемая | + `k_proj`, `out_proj` | ~15M | Основной вариант |
| Максимальная | + `fc1`, `fc2` | ~30M | Максимальная адаптация |

### Гиперпараметры LoRA

| Параметр | Значение | Примечание |
|---|---|---|
| `r` (ранг) | 16–32 | Выше = больше ёмкость |
| `lora_alpha` | 2×r | Масштабирующий фактор |
| `lora_dropout` | 0.05 | Регуляризация |
| `learning_rate` | 1e-3 — 5e-4 | В 10–100× выше чем full fine-tune |
| `bias` | "none" | Bias замораживаем |
| `task_type` | "SEQ_2_SEQ_LM" | Критично для encoder-decoder |

### AdaLoRA (адаптивный ранг)

Автоматически определяет оптимальный ранг для каждого слоя:

```python
from peft import AdaLoraConfig

adalora_config = AdaLoraConfig(
    init_r=64,
    target_r=16,
    beta1=0.85,
    beta2=0.85,
    tinit=200,
    tfinal=1000,
    deltaT=10,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    task_type="SEQ_2_SEQ_LM",
)
```

### Сохранение и загрузка адаптера

```python
# Сохранение (~63 МБ)
model.save_pretrained("whisper-russian-lora-adapter")

# Загрузка
from peft import PeftModel
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
model = PeftModel.from_pretrained(base_model, "whisper-russian-lora-adapter")

# Merge в базовую модель (убирает overhead при инференсе)
model = model.merge_and_unload()
model.save_pretrained("whisper-large-v3-russian-merged")
```

---

## 3. Полный fine-tune

Требует значительно больше ресурсов, но даёт максимальное качество.

```python
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
model.generation_config.language = "russian"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-radio-ru-full",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,    # экономия памяти
    fp16=True,
    optim="adamw_bnb_8bit",        # 8-bit Adam
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=10000,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=2000,
    eval_steps=2000,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)
```

---

## 4. Подготовка данных

### Требования к аудио

- **Частота дискретизации:** 16000 Гц (обязательно)
- **Каналы:** моно
- **Формат:** WAV (предпочтительно); MP3/FLAC конвертируются внутренне
- **Длительность сегмента:** до 30 секунд (окно Whisper)
- **Минимум данных:** 10–50 часов для заметного улучшения, 100+ часов — идеально

### Структура кастомного датасета

Для `vasistalodagala/whisper-finetune`:

```
# audio_paths.txt
utt_001 /path/to/audio/001.wav
utt_002 /path/to/audio/002.wav

# text.txt
utt_001 Центральная, говорит Сокол-один, принял вызов
utt_002 Подтверждаю, выезжаем на Ленина двадцать три
```

Для HuggingFace Datasets:

```python
from datasets import Dataset, Audio

dataset = Dataset.from_dict({
    "audio": ["/path/to/audio1.wav", "/path/to/audio2.wav"],
    "sentence": ["транскрипция 1", "транскрипция 2"],
}).cast_column("audio", Audio(sampling_rate=16000))
```

### Фильтрация

```python
MAX_DURATION = 30.0
MIN_DURATION = 1.0

def filter_by_duration(example):
    duration = len(example["audio"]["array"]) / example["audio"]["sampling_rate"]
    return MIN_DURATION < duration < MAX_DURATION

dataset = dataset.filter(filter_by_duration)
```

### Препроцессинг аудио (sox)

```bash
# Нормализация и компрессия (радио-качество)
sox record.wav -r 16k record-normalized.wav \
    norm -0.5 compand 0.3,1 -90,-90,-70,-70,-60,-20,0,0 -5 0 0.2

# Очень шумное радио (8 кГц upsampled)
sox record.wav -r 8000 record-normalized.wav \
    norm -0.5 compand 0.3,1 -90,-90,-70,-50,-40,-15,0,0 -7 0 0.15
```

---

## 5. Аугментация под радиоусловия

**Критично: НЕ чистить аудио от шума перед обучением.** Модель должна учиться транскрибировать сквозь помехи.

### audiomentations

```python
from audiomentations import (
    Compose, AddGaussianSNR, BandPassFilter,
    ClippingDistortion, AddBackgroundNoise,
)

radio_augment = Compose([
    # Полосовой фильтр (диапазон радиосвязи ~300–3400 Гц)
    BandPassFilter(min_center_freq=300, max_center_freq=3400, p=0.8),
    # Статический шум
    AddGaussianSNR(min_snr_db=5, max_snr_db=20, p=0.7),
    # Клиппинг от перегруза
    ClippingDistortion(
        min_percentile_threshold=0, max_percentile_threshold=20, p=0.3
    ),
    # Фоновый шум (двигатель, ветер)
    AddBackgroundNoise(
        sounds_path="/path/to/background_noises/",
        min_snr_db=3, max_snr_db=15, p=0.5,
    ),
])

# Применение при подготовке датасета
import random

def prepare_dataset_with_augmentation(batch):
    audio = batch["audio"]["array"]
    sr = batch["audio"]["sampling_rate"]

    if random.random() > 0.5:
        audio = radio_augment(samples=audio, sample_rate=sr)

    batch["input_features"] = processor.feature_extractor(
        audio, sampling_rate=sr
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch
```

### SpecAugment (встроенный в Whisper)

```python
model.config.apply_spec_augment = True
model.config.mask_time_prob = 0.05
model.config.mask_time_length = 10
model.config.mask_feature_prob = 0.02
model.config.mask_feature_length = 10
```

### Demucs для тяжёлых случаев (препроцессинг при инференсе)

```bash
pip install demucs
python -m demucs --two-stems=vocals noisy_radio.wav
# → separated/htdemucs/noisy_radio/vocals.wav
```

---

## 6. Требования к железу

### Полный fine-tune

| Модель | Параметры | Мин. VRAM | Рекомендуемо | Время (100ч данных) |
|---|---|---|---|---|
| tiny | 39M | 4 ГБ | 8 ГБ (T4) | 2–4 ч |
| base | 74M | 8 ГБ | 16 ГБ (V100) | 4–8 ч |
| small | 244M | 16 ГБ | 16 ГБ (V100) | 8–16 ч |
| medium | 769M | 24 ГБ | 40 ГБ (A100) | 24–48 ч |
| large-v3 | 1.55B | 40 ГБ | 2×80 ГБ (A100) | 60+ ч |

### LoRA + INT8

| Модель | Full VRAM | LoRA + INT8 VRAM | Размер чекпоинта |
|---|---|---|---|
| medium | 24 ГБ | < 6 ГБ | ~30 МБ |
| large-v2 | 40+ ГБ | < 8 ГБ | 63 МБ |
| large-v3 | 40+ ГБ | < 10 ГБ | ~80 МБ |

---

## 7. Конвертация для продакшна

### HuggingFace → faster-whisper (CTranslate2)

```bash
pip install ctranslate2

ct2-whisper-converter \
    --model ./whisper-radio-ru-merged \
    --output_dir ./whisper-ct2-radio-ru
```

### HuggingFace → whisper.cpp (GGML)

```bash
python whisper.cpp/models/convert-pt-to-ggml.py \
    ./whisper-radio-ru-merged \
    ./whisper-ggml-radio-ru
```

### Инференс через faster-whisper (4× ускорение)

```python
from faster_whisper import WhisperModel

model = WhisperModel("./whisper-ct2-radio-ru", device="cuda", compute_type="float16")

segments, info = model.transcribe(
    "radio_call.wav",
    language="ru",
    initial_prompt="Центральная, Сокол-один, принял.",
    beam_size=5,
    vad_filter=True,
)

for segment in segments:
    print(f"[{segment.start:.1f}s] {segment.text}")
```

---

## 8. Альтернативные подходы

### LLM-постобработка

Исправление ошибок ASR через LLM:

```python
def post_process(raw_text):
    prompt = f"""Исправь ошибки ASR в транскрипции радиоэфира экстренных служб.
Известные позывные: Сокол-1, Орёл-3, Центральная.
Известные адреса: Ленина 23, Пушкина 45.

Сырая транскрипция: {raw_text}
Исправленная:"""
    # → LLM API call
```

### Model merging (TIES)

Объединение нескольких дообученных моделей без дополнительного обучения:

```yaml
# mergekit конфигурация
method: ties
parameters:
  ties_density: 0.9
  encoder_weights: [0.8, 0.2]
  decoder_weights: [0.2, 0.8]
```

### Multi-model pipeline (снижение галлюцинаций)

1. **Silero VAD / WebRTC VAD** — сегментация и детекция речи
2. **AST (Audio Spectrogram Transformer)** — фильтрация не-речи
3. **Whisper** — финальная транскрипция

---

## 9. Ресурсы и ссылки

### Гайды
- [HuggingFace Blog: Fine-Tune Whisper](https://huggingface.co/blog/fine-tune-whisper)
- [HF Transformers examples](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition)
- [PEFT INT8 + LoRA examples](https://github.com/huggingface/peft/tree/main/examples/int8_training)
- [HF Community Whisper Event](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event)

### Инструменты
- [vasistalodagala/whisper-finetune](https://github.com/vasistalodagala/whisper-finetune) — standalone fine-tuning
- [jumon/whisper-finetuning](https://github.com/jumon/whisper-finetuning) — с поддержкой таймстампов
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 инференс (4×)
- [ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp) — C/C++ инференс
- [Vaibhavs10/insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) — batch CLI
- [speechbrain/speechbrain](https://github.com/speechbrain/speechbrain) — полный ASR toolkit
- [iver56/audiomentations](https://github.com/iver56/audiomentations) — аугментация аудио
