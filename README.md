# Агент для роботи з даними на базі Streamlit + Milvus + Ollama

## Що це за застосунок

Цей репозиторій містить інтерактивний веб-додаток на Streamlit, який дозволяє завантажувати документи, відео, зображення та аудіо, автоматично перетворювати їх на текстові фрагменти, індексувати у Milvus і спілкуватися з даними через чат-інтерфейс з підтримкою гібридного пошуку (BM25 + dense-вектори BGE-M3) та генеративних LLM-відповідей.  

Код організовано у чотири файли:

| Файл | Призначення |
|------|-------------|
| **app.py** | фронтенд на Streamlit, маршрути режимів, керування станом |
| **functions.py** | підготовка даних, інтеграція Whisper, Ollama, побудова JSON для Milvus |
| **rag.py** | класи *HybridRetriever* для документів, відео, зображень і аудіо |
| **config.py** | глобальні константи, системні промпти, об’єкт `ChatOllama` |

## Передумови

* Python ³ˑ⁸ або новіший  
* Docker Desktop (Mac / Windows) або Docker Engine + Docker Compose (Linux)  
* Git  
* FFmpeg (потрібен yt-dlp)  
* 8 ГБ RAM; для великих моделей бажана GPU-карта

## Клонування та локальний запуск

# Крок 1 — клонувати репозиторій**  
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>

# Крок 2 — створити віртуальне середовище і встановити залежності**

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt


# Крок 3 — запустити Milvus (Docker Compose)**

**Інструкції для macOS, Windows і Linux однакові: завантажте готовий docker-compose.yml, підніміть сервіс і перевірте стан.**

curl -L https://github.com/milvus-io/milvus/releases/download/v2.3.21/milvus-standalone-docker-compose.yml \
     -o docker-compose.yml
docker compose up -d
docker compose ps        # має з’явитися milvus-standalone running

# Крок 4 — (необов’язково) встановити Attu GUI для Milvus

**Завантажте Attu Desktop із сайту Milvus, запустіть і створіть підключення до http://127.0.0.1:19530.
Далі ви зможете оглядати колекції, додавати документи вручну та виконувати запити візуально.**

milvus.io

# Крок 5 — встановити Ollama

**macOS — завантажте .dmg із сайту й перетягніть у Applications.** 

**Linux — одна команда:**

curl -fsSL https://ollama.com/install.sh | sh

**Windows — скачайте інсталятор із розділу Download → Windows та запустіть.** 

Після встановлення обов’язково завантажте потрібні моделі:

ollama pull gemma3:4b
ollama pull INSAIT-Institute/MamayLM-Gemma-2-9B-IT-v0.1   # українськомовна

# Крок 6 — запустити Streamlit-додаток

streamlit run app.py

Відкрийте http://localhost:8501 у браузері.

# Використання

Після старту в боковій панелі введіть назву колекції (якщо вона не існує, код створить її автоматично). Оберіть режим:

«Обробка документу» — завантажте PDF, DOCX, XLSX, CSV або MD. Перегляньте прев’ю, натисніть Почати чат для спілкування з контентом.

«Обробка відео» — вставте YouTube URL, виберіть модель Whisper, натисніть Обробити. Після транскрипції доступне спілкування.

«Обробка зображення» — завантажте файли, дочекайтеся автогенерації опису, далі чат.

«Обробка аудіо» — завантажте аудіофайл, виконайте транскрипцію, спілкуйтеся.

«Почати діалог» — універсальний чат з усією колекцією.

Запити проходять через HybridRetriever, який об’єднує BM25 та dense-вектори BGE-M3, після чого LLM (Gemma 3 або MamayLM-Gemma) формує відповідь з посиланням на знайдений контекст.

# Корисні команди Milvus / Docker

docker compose logs milvus-standalone      # перегляд логів
docker compose stop && docker compose rm   # зупинка та видалення контейнерів

# Оновлення та резервне копіювання

Milvus зберігає дані в томах volumes/; для резервної копії достатньо скопіювати папку milvus_standalone_data. При оновленні версії Milvus замініть docker-compose.yml на свіжий, збережіть томи і перезапустіть контейнери.

# Поширені помилки

Connection refused :19530 — контейнер Milvus не запущений або порт зайнятий. Переконайтеся, що docker compose ps показує running.

CUDA не знайдено — Ollama спробує використати GPU; якщо ви на CPU-машині, ігноруйте попередження або вимкніть GPU-режим.

KeyError context під час чату — переконайтеся, що функція create_prompt() у коді використовує змінні {context} і {question}.

