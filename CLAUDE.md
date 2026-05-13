# Radio Transcription Research

Исследовательский проект: автоматическая транскрипция радиоэфира экстренных служб (пожарные, МЧС, полиция) в текст. Фокус — русский язык.

## Контекст

- Русскоязычных open-source решений для радиоэфира **не существует** — это незанятая ниша
- Все компоненты (SDR-приём, VAD, ASR) существуют по отдельности и их нужно собрать в пайплайн
- Экосистема англоязычных проектов сконцентрирована вокруг trunk-recorder и SDRTrunk

## Ключевые решения

- **Базовая ASR-модель:** `antony66/whisper-large-v3-russian` (WER 6.39%) — лучшая для русского
- **Дообучение:** LoRA (8–10 GB VRAM) предпочтительнее full fine-tune (40+ GB)
- **Инференс:** faster-whisper (CTranslate2) — 4× ускорение над стандартным Whisper
- **Приём в РФ:** аналоговые пожарные частоты 148–174 МГц FM, RTL-SDR достаточно
- **Данные для обучения:** НЕ чистить шум — модель должна учиться транскрибировать сквозь помехи

## Структура документации

| Файл | Содержание |
|------|-----------|
| [docs/projects.md](docs/projects.md) | Каталог 30+ open-source проектов с URL, стеком, статусом активности |
| [docs/whisper-finetuning.md](docs/whisper-finetuning.md) | Полный гайд: LoRA, full fine-tune, prompt engineering, аугментация, железо, конвертация |
| [docs/pipeline-architecture.md](docs/pipeline-architecture.md) | 4 варианта архитектуры (от минимального до production), Python-скелет, частоты РФ |
| [docs/russian-models.md](docs/russian-models.md) | Предобученные русские Whisper-модели, датасеты, рекомендации |
| [docs/sdr-hardware.md](docs/sdr-hardware.md) | SDR-приёмник MSi2500+MSi001: ESD-защита, моды, апгрейды, ссылки на схемы |
| [docs/russian-resources.md](docs/russian-resources.md) | Русскоязычные магазины, форумы, альтернативы: где купить компоненты в РФ |

## Топ-проекты для референса

- **OpenScanner** (revtex/OpenScanner) — лучшее all-in-one: сканер + транскрипция, Go, 15 языков
- **trunk-transcribe** (CrimeIsDown) — production-grade пайплайн trunk-recorder + Whisper
- **Scanner-map** (poisonednumber) — транскрипция + геолокация на карте
- **imbe-asr** (trunk-reporter) — уникальный: ASR напрямую из IMBE-кодека без реконструкции аудио

## Железо

SDR-приёмник — клон RSP1 на MSi2500/MSi001 (4 антенных входа, Aliexpress). Критическая доработка: антенные входы НЕ защищены от ESD — нужно напаять TVS-диоды (PESD5V0S1BA, 0.25 пФ, SOD-323) на каждый SMA. Подробности в [docs/sdr-hardware.md](docs/sdr-hardware.md).

## Рекомендуемый пайплайн

```
RTL-SDR → rtl_fm (FM 148–174 МГц) → Silero VAD → sox (16кГц моно) → faster-whisper (русская модель + LoRA) → LLM постобработка → текст/алерты
```

## При добавлении новых материалов

- Сохранять в `docs/` как отдельные тематические файлы
- Обновлять таблицу в этом файле
- Рабочий код и примеры — в `examples/` (пока не создана)
