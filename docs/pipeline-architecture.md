# Архитектура пайплайна: от антенны до текста

## Общая схема

```
┌─────────┐    ┌──────────┐    ┌───────────┐    ┌─────────┐    ┌──────────┐
│ RTL-SDR │───▶│ Демод.   │───▶│ VAD +     │───▶│ Whisper  │───▶│ Вывод    │
│ антенна │    │ rtl_fm / │    │ нормализ. │    │ ASR     │    │ текст/   │
│         │    │ trunk-rec│    │ sox/silero │    │         │    │ алерты   │
└─────────┘    └──────────┘    └───────────┘    └─────────┘    └──────────┘
```

## Вариант 1: Минимальный (аналоговое FM-радио)

Для аналоговых пожарных частот в РФ (148–174 МГц).

### Железо

- **RTL-SDR** USB-донгл (~$25): RTL-SDR Blog V4, Nooelec NESDR SMArt
- Антенна: штатная телескопическая или Nagoya NA-771 (лучше приём)

### Софт

```bash
# 1. Приём и демодуляция FM
rtl_fm -f 154.000M -M fm -s 12500 -r 16000 -l 0 - | \
# 2. Нормализация
sox -t raw -r 16000 -e signed -b 16 -c 1 - -t wav - norm -0.5 | \
# 3. Запись сегментов (по 30 сек)
sox - segment_%03d.wav trim 0 30 : newfile : restart
```

```python
# 4. Транскрипция
from faster_whisper import WhisperModel
import glob

model = WhisperModel("large-v3", device="cuda")

for f in sorted(glob.glob("segment_*.wav")):
    segments, _ = model.transcribe(
        f, language="ru",
        initial_prompt="Центральная, принял вызов, выезжаем.",
        vad_filter=True,
    )
    for seg in segments:
        print(f"[{f}] [{seg.start:.1f}s] {seg.text}")
```

### Ограничения

- Нет автоматической записи по активности (VOX)
- Одна частота за раз
- Нет интеграции с talkgroup

---

## Вариант 2: Trunk-recorder (P25 / цифровые системы)

Для транковых и P25-систем. Автоматически записывает звонки по talkgroup.

### Docker Compose

```yaml
version: "3.8"

services:
  trunk-recorder:
    image: robotastic/trunk-recorder:latest
    devices:
      - /dev/bus/usb:/dev/bus/usb
    volumes:
      - ./config.json:/app/config.json
      - ./recordings:/app/recordings
    restart: unless-stopped

  transcriber:
    image: ghcr.io/crimeisdown/trunk-transcribe:latest
    environment:
      - WHISPER_MODEL=large-v3
      - WHISPER_LANGUAGE=ru
    volumes:
      - ./recordings:/recordings
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    depends_on:
      - trunk-recorder
```

### Альтернатива: tr-stack (всё в одном)

```bash
git clone https://github.com/trunk-reporter/tr-stack
cd tr-stack
docker compose up -d
```

---

## Вариант 3: OpenScanner (рекомендуемый all-in-one)

Единый Go-бинарник с встроенной транскрипцией.

```bash
git clone https://github.com/revtex/OpenScanner
cd OpenScanner
# Сборка с GPU-поддержкой
make build-cuda
# Или скачать готовый релиз
```

Поддерживает:
- Встроенный Whisper (go-whisper / whisper.cpp)
- GPU: CUDA, Intel iGPU, AMD ROCm
- 15 языков включая русский
- Live WebSocket feed
- RBAC и публичные ссылки

---

## Вариант 4: Custom pipeline для русского пожарного эфира

### Компоненты

```
RTL-SDR
  │
  ▼
rtl_fm (демодуляция FM 154 МГц)
  │
  ▼
Silero VAD (детекция речи, отсечение тишины)
  │
  ▼
sox (нормализация, 16 кГц моно)
  │
  ▼
faster-whisper (antony66/whisper-large-v3-russian или свой LoRA)
  │
  ▼
LLM пост-обработка (опционально: исправление жаргона)
  │
  ├──▶ Текстовый лог (файл / БД)
  ├──▶ Telegram-бот (алерты по ключевым словам)
  └──▶ Веб-интерфейс (поиск по транскриптам)
```

### Python-скелет

```python
import subprocess
import numpy as np
import torch
from faster_whisper import WhisperModel

# --- Silero VAD ---
vad_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad"
)
get_speech_timestamps = utils[0]

# --- Whisper ---
whisper = WhisperModel(
    "antony66/whisper-large-v3-russian",
    device="cuda",
    compute_type="float16",
)

PROMPT = (
    "Центральная, говорит Сокол-один, принял вызов. "
    "Пожарный расчёт выезжает. Адрес: Ленина двадцать три."
)

def transcribe_segment(audio_array: np.ndarray, sr: int = 16000) -> str:
    # VAD
    speech_timestamps = get_speech_timestamps(
        torch.from_numpy(audio_array).float(), vad_model, sampling_rate=sr
    )
    if not speech_timestamps:
        return ""

    segments, _ = whisper.transcribe(
        audio_array,
        language="ru",
        initial_prompt=PROMPT,
        beam_size=5,
        vad_filter=False,  # уже отфильтровали
    )
    return " ".join(seg.text for seg in segments)


def stream_from_rtl_sdr(freq_mhz: float = 154.0):
    """Стрим аудио с RTL-SDR через rtl_fm."""
    proc = subprocess.Popen(
        [
            "rtl_fm",
            "-f", f"{freq_mhz}M",
            "-M", "fm",
            "-s", "12500",
            "-r", "16000",
            "-l", "0",
            "-",
        ],
        stdout=subprocess.PIPE,
    )

    CHUNK = 16000 * 30  # 30 секунд
    buffer = b""

    while True:
        data = proc.stdout.read(CHUNK * 2)  # 16-bit = 2 bytes per sample
        if not data:
            break
        buffer += data
        if len(buffer) >= CHUNK * 2:
            audio = np.frombuffer(buffer[:CHUNK * 2], dtype=np.int16)
            audio = audio.astype(np.float32) / 32768.0
            buffer = buffer[CHUNK * 2:]

            text = transcribe_segment(audio)
            if text.strip():
                print(f">>> {text}")
                # TODO: отправка в Telegram, запись в БД


if __name__ == "__main__":
    stream_from_rtl_sdr(154.0)
```

---

## Приём радио в РФ: частоты и стандарты

### Аналоговые частоты экстренных служб

| Служба | Диапазон | Модуляция |
|---|---|---|
| Пожарные / МЧС | 148–174 МГц | FM (узкополосная, 12.5 кГц) |
| Скорая помощь | 148–174 МГц | FM |
| Полиция (устаревшие) | 148–174 МГц | FM |

### Цифровые стандарты

| Стандарт | Описание | Декодирование |
|---|---|---|
| TETRA | Внедряется для МВД/МЧС | osmo-tetra, tetra-kit (сложно) |
| DMR | Некоторые ведомства | DSD+, SDRTrunk |
| P25 | Практически не используется в РФ | trunk-recorder, SDRTrunk |

### Правовые аспекты

Приём радиосигналов в РФ для личного использования — серая зона. Использование/распространение перехваченных переговоров может нарушать закон о связи (ФЗ-126) и о персональных данных. Публикация транскриптов — юридический риск.
