# Limi

Limi — голосовой ассистент для дома с анимированным лицом на экране.
Скажи «Лими» (или «Limi»), задай вопрос, он ответит голосом и покажет эмоцию на лице.
Работает на ПК для разработки и на Raspberry Pi с маленьким экраном, микрофоном и колонкой.
Музыка пока не реализована (есть заглушка).

## Железо

- **Разработка:** любой ПК с Windows или Linux, микрофон и колонки.
- **Продакшен:** Raspberry Pi 4/5 (64-bit OS), маленький дисплей, USB-микрофон и USB-колонка.

## Установка на Windows

1. Установи Python 3.11 или новее с [python.org](https://www.python.org/downloads/). При установке отметь **Add python.exe to PATH**.
2. Открой PowerShell в папке проекта и создай виртуальное окружение:
   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   Если PowerShell ругается на скрипты: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
3. Поставь зависимости:
   ```powershell
   pip install -r requirements.txt
   ```
   `sounddevice` на Windows ставится с готовыми библиотеками PortAudio. `faster-whisper` при первом запуске сам скачает модель распознавания.
4. Скопируй конфиг:
   ```powershell
   copy config.yaml.example config.yaml
   ```
5. Если ты используешь `llm.provider: local_gguf`, см. раздел про GGUF ниже. Для облачного API см. раздел про API.
6. Для первого запуска без больших моделей поставь в `config.yaml`:
   - `stt.provider: google` (нужен интернет),
   - `tts.provider: gtts` (нужен интернет), если не хочешь ставить xVibePocketTTS или PyTorch.

Опциональные пакеты:
- `llama-cpp-python` на Windows собирается из исходников и требует Visual Studio Build Tools (компонент «Разработка классических приложений на C++»). Для первого запуска его можно пропустить.
- **Голос по умолчанию: xVibePocketTTS** (`tts.provider: xvibe_local`). Русская модель от GenVoice, работает на CPU, умеет клонировать голос по WAV-образцу 1–30 с. Ставится из git, на PyPI её нет. Нужен Python 3.10–3.13 (на 3.13 проверено) и интернет при первом запуске, чтобы скачать модель с Hugging Face:
  ```powershell
  pip install git+https://github.com/GenVoice/xVibePocketTTS
  ```
  Установка тянет PyTorch (несколько сотен МБ). Голос выбирается в `tts.xvibe_voice`: `male_deep`, `female_silky`, `male_authoritative` или путь к своему WAV.
- Альтернатива: `tts.provider: piper_local` (лёгкий, подходит для Pi). `piper-tts` ставится через `pip install piper-tts`, голос `.onnx` и `.onnx.json` положи в `models/tts/`.

## Установка на Linux / Raspberry Pi

```bash
sudo apt install libportaudio2 libsdl2-dev python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-pi.txt     # на Pi; на ПК используй requirements.txt
cp config.yaml.example config.yaml
```

На Pi рекомендуется `stt.whisper_model: tiny` или `base`, и `llm.provider: api`. Локальная модель GGUF на Pi 4 будет очень медленной.

## Локальная модель GGUF

1. Положи файл `.gguf` в папку `models/`, например `models/llm/model.gguf`. Limi сам модели не скачивает.
2. Укажи путь в `config.yaml`:
   ```yaml
   llm:
     provider: local_gguf
     local_gguf:
       model_path: models/llm/model.gguf
       n_ctx: 4096
       n_gpu_layers: 0
   ```
3. Если файла нет, Limi покажет ошибку с ожидаемым путём и не запустится.

## API провайдер (OpenAI-совместимый или Anthropic)

Ключ можно задать в `config.yaml` или в переменной окружения. Переменная имеет приоритет.

**OpenAI-совместимый** (OpenAI, DeepSeek, OpenRouter, Ollama, LM Studio и т. п.):
```yaml
llm:
  provider: api
  api:
    base_url: https://api.openai.com/v1   # адрес сервиса
    model: gpt-4o-mini                     # имя модели
    api_key: ""                            # лучше оставить пустым
```
Ключ задай переменной `LIMI_API_KEY`:
- PowerShell: `$env:LIMI_API_KEY = "твой_ключ"`
- Linux / Pi: `export LIMI_API_KEY="твой_ключ"`

**OpenRouter:** `base_url: https://openrouter.ai/api/v1`, в `model` имя модели с сайта openrouter.ai (например `openai/gpt-4o-mini`). Ключ можно задать в `OPENROUTER_API_KEY`:
- PowerShell: `$env:OPENROUTER_API_KEY = "твой_ключ"`

**Anthropic:** `llm.provider: anthropic`, поле `llm.anthropic.model`, ключ в `ANTHROPIC_API_KEY`.

Ключи нигде не пишутся в лог (маскируются). Не коммить `config.yaml` и `.env`: они в `.gitignore`.

## VPN / прокси (Happ, v2rayN, Clash)

Если OpenRouter или поиск не грузят при включённом VPN, Limi нужно отправлять запросы через локальный порт клиента:

1. Включи VPN-клиент и посмотри в его настройках локальный адрес, обычно `127.0.0.1` и порт.
2. В `config.yaml` укажи:
   ```yaml
   network:
     proxy: "http://127.0.0.1:10809"
   ```
   Или задай переменную окружения, она важнее: `$env:LIMI_PROXY = "http://127.0.0.1:10809"`.
3. Если клиент даёт только SOCKS5, порт будет вида `10808`. Тогда используй `socks5://127.0.0.1:10808` и установи поддержку SOCKS: `pip install "httpx[socks]"`.
4. Ошибка `Unknown scheme proxy 127.0.0.1:10808` означает, что в переменных окружения нет `http://` или `socks5://`. Limi теперь сам дописывает `http://`, но лучше указывать схему явно.

Порт `10809` обычно HTTP, `10808` обычно SOCKS5 (так в v2rayN). Точные порты смотри в своём клиенте. Адреса `localhost` и `127.0.0.1` в обход прокси, поэтому локальный Ollama или LM Studio работают.

## Системный промпт

Характер ассистента лежит в `prompts/system.txt`. Его можно править без изменения кода.

## Замена звуков

Все звуки лежат в `sounds/`. Имена фиксированные:

| Файл | Когда играет |
|---|---|
| `wake.wav` | услышал wake word |
| `ack.wav` | запрос принят |
| `sleep.wav` | уходит в сон |
| `error.wav` | что-то сломалось |
| `confirm.wav` | спрашивает подтверждение забывания |
| `done.wav` | память стерта |
| `mood_happy.wav`, `mood_sad.wav`, `mood_angry.wav` | необязательные, при смене настроения |

Формат: WAV, 16 kHz или 44.1 kHz, моно. Просто замени файл на свой с тем же именем. Если файла нет, Limi работает без него.

## Запуск

```bash
python -m limi.main
```

Проверка лица без микрофона и моделей (циклически показывает все состояния и настроения):
```bash
python -m limi.main --demo-face
```

Окно по умолчанию 480x320. Для Pi включи `display.fullscreen: true`.

## Память

- Факты и история хранятся в `data/limi.db`. Папка `data/` в `.gitignore`.
- Посмотреть, что помнит Limi:
  ```bash
  python -m limi.memory list
  ```
- Стереть всё:
  ```bash
  python -m limi.memory wipe
  ```
- Голосом: скажи «забудь меня» или «удали диалог», а потом «да». Без явного «да» ничего не удаляется.

## Известные ограничения

- **Музыка не реализована.** Инструменты `music_play`, `music_like`, `music_dislike` возвращают «music not implemented yet». Заглушка в `limi/music/stub.py`, туда будет подключаться реальный провайдер.
- **Точность wake word.** Режим `stt` (поиск слова в распознанном тексте) ловит слово хуже и тратит больше ресурсов. `openwakeword` точнее, но ставится отдельно. Ложные срабатывания и пропуски возможны, особенно в шумной комнате.
- **Производительность Pi.** Whisper `small` и локальные LLM на Pi 4 работают медленно. Используй `tiny`/`base` и API.
- **Не все части проверены** на реальном железе: микрофон, TTS-воспроизведение и интеграция с провайдерами требуют ручной проверки.

## Тесты

```bash
pip install pytest
pytest
```
