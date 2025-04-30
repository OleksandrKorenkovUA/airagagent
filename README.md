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
cd airagagent</pre>

### Крок 2 — створити віртуальне середовище і встановити залежності**

<pre> python -m venv .venv
source .venv/bin/activate </pre>
Для Windows PowerShell спочатку встановлення python через Start-Process - [Документація Docker](https://www.python.org/downloads/windows)

# Windows: .venv\Scripts\activate

<pre>pip install -r requirements.txt</pre>

### Крок 3 — запустити Milvus (Docker Compose)**

**Інструкції для macOS, Windows і Linux однакові: завантажте готовий docker-compose.yml, підніміть сервіс і перевірте стан.**

Для  Milvus in Docker (Windows) спочатку треба [Install Docker Desktop on Windows](https://docs.docker.com/desktop/release-notes/#4390), далі відкрий Docker Desktop від імені адміністратора, клацнувши правою кнопкою миші та обравши Запустити від імені адміністратора. Завантаж інсталяційний скрипт і збережи його під назвою standalone.bat. 

<pre>Invoke-WebRequest https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat -OutFile standalone.bat</pre>

Запусти завантажений скрипт, щоб запустити Milvus як контейнер Docker.

<pre>.\standalone.bat start
Wait for Milvus starting...
Start successfully.
To change the default Milvus configuration, edit user.yaml and restart the service.</pre>

Контейнер Docker з назвою milvus-standalone запущено на порту 19530. Разом із Milvus у тому ж контейнері встановлено вбудований etcd, який працює на порту 2379. Його файл конфігурації змаплено до embedEtcd.yaml у поточній теці. Том із даними Milvus змаплено до volumes/milvus у поточній теці. Ти можеш керувати контейнером Milvus і збереженими даними за допомогою таких команд.

Детальніше тут https://milvus.io/docs/install_standalone-windows.md

**Запуск Milvus за допомогою Docker Compose**

Після встановлення Docker Desktop на Windows ти можеш отримати доступ до Docker CLI через PowerShell або командний рядок Windows у режимі адміністратора. Виконати Docker Compose для запуску Milvus можна з PowerShell, командного рядка Windows або середовища WSL 2. 

Із PowerShell або командного рядка Windows:
Відкрий Docker Desktop у режимі адміністратора, клацнувши правою кнопкою миші й обравши Запустити від імені адміністратора.
Виконай у PowerShell або командному рядку Windows такі команди, щоб завантажити файл конфігурації Docker Compose для Milvus Standalone і запустити Milvus:

# Download the configuration file and rename it as docker-compose.yml
<pre>Invoke-WebRequest https://github.com/milvus-io/milvus/releases/download/v2.4.15/milvus-standalone-docker-compose.yml -OutFile docker-compose.yml</pre>

# Start Milvus
<pre>docker compose up -d
Creating milvus-etcd  ... done
Creating milvus-minio ... done
Creating milvus-standalone ... done</pre>

Залежно від швидкості з’єднання, завантаження образів для інсталяції Milvus може зайняти деякий час. Після запуску контейнерів milvus-standalone, milvus-minio та milvus-etcd ти побачиш наступне: 
Контейнер milvus-etcd не відкриває порти для хосту, а його дані зберігаються у теці volumes/etcd у поточній директорії.
Контейнер milvus-minio доступний локально на портах 9090 і 9091, використовує стандартні облікові дані автентифікації, а його дані зберігаються у volumes/minio.
Контейнер milvus-standalone працює локально на порту 19530 з налаштуваннями за замовчуванням, зберігає свої дані у volumes/milvus.

**Варіант (для Linux/macOS)**

<pre>curl -L https://github.com/milvus-io/milvus/releases/download/v2.3.21/milvus-standalone-docker-compose.yml \
     -o docker-compose.yml

docker compose up -d

docker compose ps        # має з'явитися milvus-standalone running</pre>

### Крок 4 — (необов'язково) встановити Attu GUI для Milvus

Завантажте Attu Desktop із сайту Milvus, запустіть і створіть підключення до http://127.0.0.1:19530.
Далі ви зможете оглядати колекції, додавати документи вручну та виконувати запити візуально.

Attu — це універсальний інструмент з відкритим вихідним кодом для адміністрування Milvus. Він оснащений інтуїтивно зрозумілим графічним інтерфейсом користувача (GUI), що дозволяє легко взаємодіяти з вашими базами даних. За кілька кліків ви можете візуалізувати стан кластера, керувати метаданими, виконувати запити до даних та багато іншого.

**Встановлення десктопного застосунку**

Завантажте десктопну версію Attu, перейшовши на сторінку випусків [Releases](https://github.com/zilliztech/attu/releases) Attu на GitHub. Виберіть версію, сумісну з вашою операційною системою, і виконайте інструкції зі встановлення.


### Крок 5 — встановити Ollama

**macOS — завантажте .dmg із сайту й перетягніть у Applications.** 

**Linux — одна команда:**

<pre>curl -fsSL https://ollama.com/install.sh | sh</pre>

**Windows — скачайте інсталятор із розділу Download → Windows та запустіть.** 

Після встановлення обов'язково завантажте потрібні моделі:

<pre>ollama pull gemma3:4b

ollama pull INSAIT-Institute/MamayLM-Gemma-2-9B-IT-v0.1   # українськомовна</pre>

**!!!Якщо при завантаженні моделей з Ollama ви стикаєтесь з ***Error: llama runner process has terminated: GGML_ASSERT(tensor->op == GGML_OP_UNARY) failed***, спробуйте встановити іншу версію Docker. Декілька користувачів помітили, що в Docker 4.41 Ollama починає падати із цим же GGML_ASSERT, а зворотне оновлення Docker Desktop до 4.40.0 вирішує проблему.**

### Крок 6 - Встановлення бібліотеки Milvus Model Lib

Бібліотека **milvus-model** забезпечує інтеграцію з поширеними моделями для отримання ембедингів та переранжування (reranker) у Milvus — високопродуктивній open-source векторній базі даних, створеній для AI-додатків. Пакет **milvus-model** підключається як залежність до **pymilvus**, офіційного Python SDK для Milvus.

**milvus-model** підтримує моделі ембедингів і reranker-моделі від провайдерів сервісів, таких як OpenAI, Voyage AI, Cohere, а також open-source рішення через SentenceTransformers або Hugging Face Text Embeddings Inference (TEI).

Бібліотека **milvus-model** сумісна з Python 3.8 і вище.

Якщо ви вже використовуєте **pymilvus**, можна встановити **milvus-model** через опціональний компонент `model`:

<pre> pip install "pymilvus[model]"
# або для zsh:
pip install pymilvus\[model\]</pre>

!!! Якщо після цього все одно виникає помилка **ModuleNotFoundError: No module named 'milvus_model'**, потрібно окремо встановити 

<pre>pip install milvus-model</pre>

### Крок 7 — запустити Streamlit-додаток

<pre>streamlit run app.py

Відкрийте http://localhost:8501 у браузері.</pre>

### Треба почекати

Після запуску має пройти певний час поки завантажаться моделі та різні пакети. Спочатку ваш скрипт автоматично інсталює три Python-пакети — ***datasets (для роботи з наборами даних), peft (для параметрично-ефективного донавчання) та FlagEmbedding (забезпечує функції ембедингів)*** — і підтверджує їх успішне встановлення. Далі починається завантаження артефактів самої моделі з Hugging Face Hub: першим завантажуються конфігураційні файли токенізатора (наприклад, tokenizer_config.json, tokenizer.json та sentencepiece.bpe.model), які описують, як розбивати текст на підсловні одиниці й відображати їх у індекси. Одночасно тягнуться й файли, пов’язані зі спеціальними токенами (special_tokens_map.json), а також підпапки з різноманітними конфігураціями модулів (***config.json, config_sentence_transformers.json*** тощо) та інколи ілюстративні файли (зображення .jpg, .webp, .DS_Store як артефакт macOS). Після цього починається завантаження самих ваг моделі: спочатку порівняно невеликі файли формату ***ONNX (model.onnx) та PyTorch-модулів (sparse_linear.pt, colbert_linear.pt)***, а потім великий бінарник з вагою нейромережі (pytorch_model.bin обсягом кілька гігабайт). Весь цей набір файлів потрібен, щоб локально ініціалізувати та виконувати ембединг- і ранкінг-модулі під час роботи вашого додатка.

### Використання

Після старту в боковій панелі введіть назву колекції (якщо вона не існує, код створить її автоматично). Оберіть режим:

- «Обробка документу» — завантажте PDF, DOCX, XLSX, CSV або MD. 

Перегляньте прев'ю, натисніть Почати чат для спілкування з контентом.

- «Обробка відео» — вставте YouTube URL, виберіть розмір моделі Whisper, натисніть "Обробити". Після транскрипції доступне спілкування.

- «Обробка зображення» — завантажте файли, дочекайтеся автогенерації опису, далі чат.

- «Обробка аудіо» — завантажте аудіофайл, виконайте транскрипцію, спілкуйтеся.

- «Почати діалог» — універсальний чат з усією колекцією.

Запити проходять через HybridRetriever, який об'єднує BM25 та dense-вектори BGE-M3, після чого LLM (Gemma 3 або MamayLM-Gemma) формує відповідь з посиланням на знайдений контекст.

### Корисні команди Milvus / Docker

<pre>docker compose logs milvus-standalone      # перегляд логів

docker compose stop && docker compose rm   # зупинка та видалення контейнерів</pre>

### Оновлення та резервне копіювання

Milvus зберігає дані в томах volumes/; 
для резервної копії достатньо скопіювати папку milvus_standalone_data. 
При оновленні версії Milvus замініть docker-compose.yml на свіжий, збережіть томи і перезапустіть контейнери.

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
