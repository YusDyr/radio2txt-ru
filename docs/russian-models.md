# Готовые русские Whisper-модели

## Предобученные модели на HuggingFace

### antony66/whisper-large-v3-russian (рекомендуется как база)

- **URL:** https://huggingface.co/antony66/whisper-large-v3-russian
- **WER:** 6.39% (Common Voice 17.0)
- **Данные:** Common Voice 17.0, 225k строк
- **Обучение:** 2×A100 80GB, 60+ часов
- **Лучший выбор** для общей русской речи и как стартовая точка для дообучения

### bond005/whisper-large-v3-ru-podlodka

- **URL:** https://huggingface.co/bond005/whisper-large-v3-ru-podlodka
- **WER:** 9.80%
- **Данные:** rulibrispeech + podlodka_speech + taiga_speech_v2
- **Оптимизирован** для лекций и интервью

### Apel-sin/whisper-large-v3-russian-ties-podlodka-v1.2

- **URL:** https://huggingface.co/Apel-sin/whisper-large-v3-russian-ties-podlodka-v1.2
- **Метод:** TIES merge двух моделей выше
- **Оптимизирован** для телефонных разговоров

### dvislobokov/whisper-large-v3-turbo-russian

- **URL:** https://huggingface.co/dvislobokov/whisper-large-v3-turbo-russian
- **Вариант:** Turbo (быстрее, чуть ниже качество)

---

## Датасеты для дообучения на русском

| Датасет | Сэмплы | Лицензия | Примечание |
|---|---|---|---|
| Mozilla Common Voice 17.0 (ru) | 225k+ | CC-0 | Теперь на Mozilla Data Collective |
| Google FLEURS (ru_ru) | ~1000 train | CC-BY-4.0 | 16 кГц WAV |
| bond005/rulibrispeech | 57.2k | — | Русский аналог LibriSpeech |
| bond005/podlodka_speech | 107 | — | Подкасты |
| bond005/taiga_speech_v2 | 12.3k | — | Web-sourced |

---

## Другие русские ASR-движки

### Vosk (alphacephei)

- **URL:** https://alphacephei.com/vosk/
- **Модели:** есть лёгкие русские модели (50 МБ и 1.5 ГБ)
- **Плюсы:** офлайн, низкие требования, real-time на CPU
- **Минусы:** качество ниже Whisper large, нет простого fine-tune

### GigaAM (SberDevices)

- **URL:** https://github.com/salute-developers/GigaAM
- **Описание:** Open-source ASR от Сбера, обученная на русском
- **Плюсы:** изначально русскоязычная

---

## Рекомендация для радиоэфира

1. **Начать с** `antony66/whisper-large-v3-russian` — лучший WER на русском
2. **LoRA дообучение** на 10–50 часах реальных радиозаписей с ручной транскрипцией
3. **Prompt engineering** с пожарной/МЧС лексикой для immediate improvement
4. **Конвертация** в faster-whisper (CTranslate2) для 4× ускорения в продакшне
