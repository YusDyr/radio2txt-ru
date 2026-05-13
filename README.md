# Radio-to-Text Transcription Research

Исследование и каталог инструментов для автоматической транскрипции радиоэфира экстренных служб (пожарные, полиция, скорая) в текст, с фокусом на русский язык.

## Содержание

- [Проекты и инструменты](docs/projects.md) — каталог open-source проектов
- [Дообучение Whisper](docs/whisper-finetuning.md) — полное руководство по fine-tuning
- [Архитектура пайплайна](docs/pipeline-architecture.md) — как собрать систему от антенны до текста
- [Готовые русские модели](docs/russian-models.md) — предобученные Whisper-модели для русского языка

## Минимальный стек

```
RTL-SDR → rtl_fm (демодуляция) → sox (нормализация) → faster-whisper (русская модель) → текст
```

## Быстрый старт

```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda")

segments, _ = model.transcribe(
    "radio.wav",
    language="ru",
    initial_prompt="Центральная, говорит Сокол-один, принял вызов. "
                   "Выезжаем на Ленина двадцать три."
)

for segment in segments:
    print(f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}")
```

## Статус

Исследовательский проект. Русскоязычных open-source решений для радиоэфира пока не существует — все компоненты есть, но их нужно собрать вместе.
