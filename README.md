# Агент для роботи з даними на базі Streamlit + Milvus + Ollama

## Що це за застосунок

Цей репозиторій містить інтерактивний веб-додаток на Streamlit, який дозволяє завантажувати документи, відео, зображення та аудіо, автоматично перетворювати їх на текстові фрагменти, індексувати у Milvus і спілкуватися з даними через чат-інтерфейс з підтримкою гібридного пошуку (BM25 + dense-вектори BGE-M3) та генеративних LLM-відповідей.  

Код організовано у чотири файли:

| Файл | Призначення |
|------|-------------|
| **app.py** | фронтенд на Streamlit, маршрути режимів, керування станом |
| **functions.py** | підготовка даних, інтеграція Whisper, Ollama, побудова JSON для Milvus |
| **rag.py** | класи *HybridRetriever* для документів, відео, зображень і аудіо |
| **config.py** | глобальні константи, системні промпти, об'єкт `ChatOllama` |

## Передумови

* Python ³ˑ⁸ або новіший  
* Docker Desktop (Mac / Windows) або Docker Engine + Docker Compose (Linux)  
* Git  
* FFmpeg (потрібен yt-dlp)  
* 8 ГБ RAM; для великих моделей бажана GPU-карта

## Клонування та локальний запуск

### Крок 1 — клонувати репозиторій**

<pre> git clone https://github.com/OleksandrKorenkovUA/airagagent
     cd airagagent </pre>

### Крок 2 — створити віртуальне середовище і встановити залежності**

<pre> 
python -m venv .venv
source .venv/bin/activate 
</pre>
Для Windows PowerShell спочатку встановлення python через Start-Process "https://www.python.org/downloads/windows/" 

# Windows: .venv\Scripts\activate

pip install -r requirements.txt


### Крок 3 — запустити Milvus (Docker Compose)**

**Інструкції для macOS, Windows і Linux однакові: завантажте готовий docker-compose.yml, підніміть сервіс і перевірте стан.**

Для  Milvus in Docker (Windows) спочатку треба Install Docker Desktop on Windows (https://docs.docker.com/desktop/setup/install/windows-install/), далі відкрий Docker Desktop від імені адміністратора, клацнувши правою кнопкою миші та обравши Запустити від імені адміністратора. Завантаж інсталяційний скрипт і збережи його під назвою standalone.bat. 

C:\>Invoke-WebRequest https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat -OutFile standalone.bat

Запусти завантажений скрипт, щоб запустити Milvus як контейнер Docker.

C:\>standalone.bat start
Wait for Milvus starting...
Start successfully.
To change the default Milvus configuration, edit user.yaml and restart the service.

Контейнер Docker з назвою milvus-standalone запущено на порту 19530. Разом із Milvus у тому ж контейнері встановлено вбудований etcd, який працює на порту 2379. Його файл конфігурації змаплено до embedEtcd.yaml у поточній теці. Том із даними Milvus змаплено до volumes/milvus у поточній теці. Ти можеш керувати контейнером Milvus і збереженими даними за допомогою таких команд.

Детальніше тут https://milvus.io/docs/install_standalone-windows.md

**Запуск Milvus за допомогою Docker Compose**

Після встановлення Docker Desktop на Windows ти можеш отримати доступ до Docker CLI через PowerShell або командний рядок Windows у режимі адміністратора. Виконати Docker Compose для запуску Milvus можна з PowerShell, командного рядка Windows або середовища WSL 2. 

Із PowerShell або командного рядка Windows:
Відкрий Docker Desktop у режимі адміністратора, клацнувши правою кнопкою миші й обравши Запустити від імені адміністратора.
Виконай у PowerShell або командному рядку Windows такі команди, щоб завантажити файл конфігурації Docker Compose для Milvus Standalone і запустити Milvus:

# Download the configuration file and rename it as docker-compose.yml
C:\>Invoke-WebRequest https://github.com/milvus-io/milvus/releases/download/v2.4.15/milvus-standalone-docker-compose.yml -OutFile docker-compose.yml

# Start Milvus
C:\>docker compose up -d
Creating milvus-etcd  ... done
Creating milvus-minio ... done
Creating milvus-standalone ... done

Залежно від швидкості з’єднання, завантаження образів для інсталяції Milvus може зайняти деякий час. Після запуску контейнерів milvus-standalone, milvus-minio та milvus-etcd ти побачиш наступне: 
Контейнер milvus-etcd не відкриває порти для хосту, а його дані зберігаються у теці volumes/etcd у поточній директорії.
Контейнер milvus-minio доступний локально на портах 9090 і 9091, використовує стандартні облікові дані автентифікації, а його дані зберігаються у volumes/minio.
Контейнер milvus-standalone працює локально на порту 19530 з налаштуваннями за замовчуванням, зберігає свої дані у volumes/milvus.

**Варіант (для Linux/macOS)**

curl -L https://github.com/milvus-io/milvus/releases/download/v2.3.21/milvus-standalone-docker-compose.yml \
     -o docker-compose.yml

docker compose up -d

docker compose ps        # має з'явитися milvus-standalone running

### Крок 4 — (необов'язково) встановити Attu GUI для Milvus

**Завантажте Attu Desktop із сайту Milvus, запустіть і створіть підключення до http://127.0.0.1:19530.
Далі ви зможете оглядати колекції, додавати документи вручну та виконувати запити візуально.**


### Крок 5 — встановити Ollama

**macOS — завантажте .dmg із сайту й перетягніть у Applications.** 

**Linux — одна команда:**

curl -fsSL https://ollama.com/install.sh | sh

**Windows — скачайте інсталятор із розділу Download → Windows та запустіть.** 

Після встановлення обов'язково завантажте потрібні моделі:

ollama pull gemma3:4b

ollama pull INSAIT-Institute/MamayLM-Gemma-2-9B-IT-v0.1   # українськомовна

### Крок 6 — запустити Streamlit-додаток

streamlit run app.py

Відкрийте http://localhost:8501 у браузері.

### Використання

Після старту в боковій панелі введіть назву колекції (якщо вона не існує, код створить її автоматично). Оберіть режим:

####«Обробка документу» — завантажте PDF, DOCX, XLSX, CSV або MD. 

Перегляньте прев'ю, натисніть Почати чат для спілкування з контентом.

####«Обробка відео» — вставте YouTube URL, виберіть розмір моделі Whisper, натисніть "Обробити". Після транскрипції доступне спілкування.

####«Обробка зображення» — завантажте файли, дочекайтеся автогенерації опису, далі чат.

####«Обробка аудіо» — завантажте аудіофайл, виконайте транскрипцію, спілкуйтеся.

####«Почати діалог» — універсальний чат з усією колекцією.

Запити проходять через HybridRetriever, який об'єднує BM25 та dense-вектори BGE-M3, після чого LLM (Gemma 3 або MamayLM-Gemma) формує відповідь з посиланням на знайдений контекст.

### Корисні команди Milvus / Docker

docker compose logs milvus-standalone      # перегляд логів

docker compose stop && docker compose rm   # зупинка та видалення контейнерів

### Оновлення та резервне копіювання

Milvus зберігає дані в томах volumes/; для резервної копії достатньо скопіювати папку milvus_standalone_data. При оновленні версії Milvus замініть docker-compose.yml на свіжий, збережіть томи і перезапустіть контейнери.

### Поширені помилки

Connection refused :19530 — контейнер Milvus не запущений або порт зайнятий. Переконайтеся, що docker compose ps показує running.

CUDA не знайдено — Ollama спробує використати GPU; якщо ви на CPU-машині, ігноруйте попередження або вимкніть GPU-режим.

KeyError context під час чату — переконайтеся, що функція create_prompt() у коді використовує змінні {context} і {question}.

## Локальна база даних SQLite

Додаток використовує SQLite для зберігання всіх запитів користувачів та відповідей моделей. Це дозволяє:

- Зберігати історію взаємодії з моделлю
- Аналізувати запити користувачів
- Використовувати дані для подальшого навчання моделі
- Відстежувати ефективність роботи системи

База даних автоматично створюється при першому запуску додатку в директорії `data/` з назвою `chat_history.db`. Структура бази даних включає:

- Таблиця `chat_history`:
  - `id` - унікальний ідентифікатор запису
  - `timestamp` - час створення запису
  - `user_query` - запит користувача
  - `model_response` - відповідь моделі
  - `context` - використаний контекст для формування відповіді
  - `model_name` - назва використаної моделі
  - `collection_name` - назва колекції, з якою працював користувач

Для роботи з базою даних використовується стандартний модуль Python `sqlite3`. Всі запити та відповіді автоматично зберігаються під час роботи додатку.
