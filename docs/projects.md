# Каталог проектов: радиоэфир → текст

Последнее обновление: 2026-05-13

## Production-ready решения

### CrimeIsDown/trunk-transcribe
- **URL:** https://github.com/CrimeIsDown/trunk-transcribe
- **Стек:** Python, FastAPI, Celery, RabbitMQ, Meilisearch, PostgreSQL, Docker, Whisper
- **Описание:** Флагманский проект. Транскрипция записей trunk-recorder с полнотекстовым поиском, алертами по ключевым словам с гео-фильтрацией, Telegram-уведомлениями. Работает на CrimeIsDown.com (Чикаго)
- **Фичи:** GPU autoscaling через Vast.ai, AI-панель анализа, React/TypeScript фронтенд
- **Звёзды:** ~40 | **Статус:** активно развивается

### revtex/OpenScanner
- **URL:** https://github.com/revtex/OpenScanner
- **Стек:** Go, Gin, SQLite, React 18, TypeScript, go-whisper (whisper.cpp), FFmpeg
- **Описание:** **Лучшее «всё-в-одном» решение.** Замена rdio-scanner со встроенной транскрипцией. Единый Go-бинарник
- **Фичи:** GPU (CUDA, Intel iGPU, AMD ROCm), speaker diarization, 15 языков + автодетект, live WebSocket, hold/avoid, закладки, RBAC, публичные ссылки
- **Звёзды:** 381 коммитов, 9 релизов, v1.3.2 | **Статус:** очень активно

### poisonednumber/Scanner-map
- **URL:** https://github.com/poisonednumber/Scanner-map
- **Стек:** Node.js, Python, faster-whisper, Google Maps/LocationIQ, Leaflet, S3
- **Описание:** Real-time система маппинга вызовов. Транскрибирует радио, извлекает адреса, отображает на интерактивной карте
- **Фичи:** Two-tone detection для пожарных/EMS, "Ask AI" чат, тепловые карты, Discord-интеграция
- **Звёзды:** ~37 | **Статус:** очень активно

### sean-reid/blotter
- **URL:** https://github.com/sean-reid/blotter
- **Стек:** faster-whisper (large-v3, CUDA), Google Cloud NLP/Places, Ollama (Qwen 2.5 7B), React 19, MapLibre GL, PostgreSQL 16, Redis, RunPod GPU
- **Описание:** Real-time карта полицейских сканеров для 21 мегаполиса США. LLM-саммари, извлечение police-кодов
- **Звёзды:** ~3 | **Статус:** активно

---

## Специализированные проекты

### trunk-reporter/imbe-asr
- **URL:** https://github.com/trunk-reporter/imbe-asr
- **Стек:** PyTorch, Conformer-CTC (290M params), ONNX Runtime, KenLM, FastAPI
- **Описание:** **Уникальный подход** — ASR напрямую из параметров IMBE-вокодера P25, без реконструкции аудио. WER 3.35% на LibriSpeech-IMBE, 19.2% на реальном P25
- **Фичи:** 3 размера модели, 5.8ms на 10с аудио (GPU), деплой на RPi5, OpenAI-совместимый API
- **Звёзды:** ~3 | **Статус:** активная разработка

### trunk-reporter/tr-stack
- **URL:** https://github.com/trunk-reporter/tr-stack
- **Стек:** Docker Compose, PostgreSQL 17, Mosquitto MQTT, Caddy
- **Описание:** Полный P25-стек в одном `docker compose up`: trunk-recorder + tr-engine + tr-dashboard + imbe-asr. Поддержка ARM64/GPU
- **Звёзды:** ~3 | **Статус:** активно

### lilhoser/pizzawave
- **URL:** https://github.com/lilhoser/pizzawave
- **Стек:** C#, .NET 9.0, Avalonia UI, whisper.net (whisper.cpp)
- **Описание:** Кроссплатформенное десктоп-приложение для пожарных/EMS. Keyword-мониторинг, импорт из RadioReference
- **Фичи:** Real-time транскрипция, talkgroup mapping, SFTP-архивы, email-нотификации
- **Звёзды:** ~3 | **Статус:** активно

### swiftraccoon/sdrtrunk-transcriber
- **URL:** https://github.com/swiftraccoon/sdrtrunk-transcriber
- **Стек:** Python, faster-whisper, SQLite, Gmail SMTP
- **Описание:** Транскрипция MP3 из SDRTrunk, авто-организация по talkgroup, извлечение dispatch-информации (10-коды, сигналы, позывные)
- **Звёзды:** ~11 | **Статус:** умеренно активно

### swiftraccoon/cpp-sdrtrunk-transcriber
- **URL:** https://github.com/swiftraccoon/cpp-sdrtrunk-transcriber
- **Стек:** C++ (C++23), CMake, whisper.cpp/OpenAI API, YAML
- **Описание:** C++ версия с параллельной обработкой и **кастомными словарями терминологии per-talkgroup**
- **Звёзды:** ~8 | **Статус:** умеренно активно

### NotJoeMartinez/copcrawler-whisper
- **URL:** https://github.com/NotJoeMartinez/copcrawler-whisper
- **Описание:** Fine-tuned Whisper модель специально для полицейского сканера. Связан с copcrawler.com — поисковик по транскриптам
- **Звёзды:** ~2 | **Статус:** экспериментальный

---

## Простые / нишевые

| Проект | URL | Описание |
|--------|-----|----------|
| simple-tr-transcription | https://github.com/cschmittiey/simple-tr-transcription | Минималистичная обёртка trunk-recorder + faster-whisper + MQTT |
| RadioScribe | https://github.com/terminalrally/RadioScribe | Транскрипция talkgroups OpenMHz с keyword detection |
| faster-whisper-radio-transcriber | https://github.com/arguingbananas/faster-whisper-radio-transcriber | Стриминг HLS/MP3/system audio + транскрипция + Ollama LLM |
| trunk-recorder-transcribe-dashboard | https://github.com/MrMortalMonkey/trunk-recorder-transcribe-dashboard | Real-time дашборд с WebSockets и DeepInfra API |
| Radio-Scanner-Transcription-Pipeline | https://github.com/Clarkson-Applied-Data-Science/Radio-Scanner-Transcription-Pipeline | Академический пайплайн (Clarkson University), Whisper + NDJSON |
| icad_transcribe | https://github.com/TheGreatCodeholio/icad_transcribe | PWA с аудио-препроцессингом для повышения точности |
| Live_Scanner_Transcriptions | https://github.com/Damoxy/Live_Scanner_Transcriptions | spaCy NLP + LLM извлечение из транскриптов RunPod |

---

## Авиация (ATC)

| Проект | URL | Описание |
|--------|-----|----------|
| Vetrenica | https://github.com/wxn0brP/Vetrenica | Real-time ATC транскрипция аэропорта Варшавы, Faster Whisper + Ollama |
| atc-transcription | https://github.com/aether-raid/atc-transcription | Академическая работа: ASR для акцентированных ATC коммуникаций (ICMCIS 2025) |

---

## Инфраструктурная база (без транскрипции)

| Проект | Звёзды | URL | Описание |
|--------|--------|-----|----------|
| ground-station | 4400 | https://github.com/sgoudelis/ground-station | Спутниковый мониторинг + SDR + Gemini/Deepgram ASR |
| sdrtrunk | 2085 | https://github.com/DSheirer/sdrtrunk | Java-декодер для SDR (P25, DMR, NBFM, AM) |
| trunk-recorder | 1100 | https://github.com/robotastic/trunk-recorder | Ядро записи P25/транковых систем. C++, GNU Radio |
| rdio-scanner | 604 | https://github.com/chuot/rdio-scanner | Go-сканер UI (single binary) |
| trunk-server (OpenMHz) | 156 | https://github.com/openmhz/trunk-server | Софт за OpenMHz.com |
| trunk-player | 85 | https://github.com/ScanOC/trunk-player | Django-плеер записей |

---

## Плагины trunk-recorder

| Плагин | URL | Описание |
|--------|-----|----------|
| callstream | https://github.com/lilhoser/callstream | Стриминг аудио на удалённый сервер (питает pizzawave) |
| tr-plugin-dvcf | https://github.com/trunk-reporter/tr-plugin-dvcf | Захват IMBE/AMBE кодек-фреймов → MQTT (питает imbe-asr) |

---

## Русскоязычные проекты

**Не найдено ни одного** готового проекта для русского радиоэфира. Все Whisper-проекты выше поддерживают русский через multilingual-модели, но специализированной интеграции нет.

Это — ниша для нового проекта.
