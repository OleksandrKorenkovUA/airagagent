# Імпорт необхідних бібліотек
from transformers import AutoModelForCausalLM, GenerationConfig
import streamlit as st  # Для створення інтерактивного веб-інтерфейсу
import os  # Для роботи з файловою системою та операційною системою
import uuid  # Для генерації унікальних ідентифікаторів для файлів та сесій
import logging  # Для логування подій та помилок у програмі
from functions import *  # Імпорт усіх допоміжних функцій з модуля functions
import whisper  # Бібліотека для розпізнавання мови в аудіофайлах
from rag import *  # Імпорт усіх функцій для роботи з системою пошуку та отримання інформації (RAG) з Milvus
from pymilvus import MilvusClient, DataType, connections, FieldSchema, MilvusException  # Компоненти для роботи з векторною базою даних Milvus
from datetime import datetime  # Для роботи з датами та часом, зокрема для логування

from IPython.display import Markdown, display  # Для відображення форматованого тексту в Jupyter Notebook
import pandas as pd  # Для роботи з табличними даними
import yt_dlp  # Для завантаження відео з YouTube та інших платформ
import hashlib  # Для створення хешів файлів для унікальної ідентифікації
from config import *  # Імпорт усіх змінних та констант з конфігураційного файлу
from docx import Document as DocxDocument  # Для роботи з документами формату .docx
from PIL import Image  # Для обробки та аналізу зображень

# Ініціалізація режиму роботи додатку, якщо він ще не встановлений
if "mode" not in st.session_state: st.session_state.mode = "menu"
# Створення директорії для збереження даних, якщо вона не існує
if not os.path.exists("data"):
    os.makedirs("data")

# Ініціалізація змінних стану сесії
# Кожна змінна відповідає за певний стан інтерфейсу або зберігає дані між взаємодіями
if "database_selected" not in st.session_state:
    st.session_state.database_selected = False  # Прапорець вибору бази даних
if "video_name" not in st.session_state:
    st.session_state.video_name = False
if 'last_video_json' not in st.session_state:
    st.session_state.last_video_json = {}  # Зберігає метадані останнього обробленого відео
if 'processed_video' not in st.session_state:
    st.session_state.video_processed = {}  # Зберігає інформацію про оброблене відео
if 'process_triggered' not in st.session_state:
    st.session_state['process_triggered'] = False  # Прапорець запуску процесу обробки
if 'chat_triggered' not in st.session_state:
    st.session_state['chat_triggered'] = False  # Прапорець активації режиму чату
if "document_selected" not in st.session_state:
    st.session_state.document_selected = False  # Прапорець вибору документа для обробки
if "video_selected" not in st.session_state:
    st.session_state.video_selected = False  # Прапорець вибору відео для обробки
if "image_selected" not in st.session_state:
    st.session_state.image_selected = False  # Прапорець вибору зображення для обробки
if "audio_selected" not in st.session_state:
    st.session_state.audio_selected = False  # Прапорець вибору аудіо для обробки

if "messages" not in st.session_state:
    st.session_state.messages = []  # Зберігає історію повідомлень у чаті
if 'start_chat' not in st.session_state:
    st.session_state.start_chat = False  # Прапорець початку чату
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # Зберігає повну історію чату для аналізу
if 'document_chunks' not in st.session_state:
    st.session_state.document_chunks = []  # Зберігає фрагменти тексту з документа
if 'video_chunks' not in st.session_state:
    st.session_state.video_chunks = []  # Зберігає фрагменти тексту з відео
if 'document_collection' not in st.session_state:
    st.session_state.document_collection = None  # Зберігає колекцію документів у Milvus
if 'video_collection' not in st.session_state:
    st.session_state.video_collection = None  # Зберігає колекцію відео у Milvus
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False  # Прапорець завершення обробки файлу
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "menu"  # Поточний режим роботи додатку

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False  # Прапорець завершення обробки документа

if "audio_processed" not in st.session_state:
    st.session_state.audio_processed = False  # Прапорець завершення обробки аудіо

if "image_processed" not in st.session_state:
    st.session_state.image_processed = False  # Прапорець завершення обробки зображення
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False  # Прапорець завершення обробки відео
if "description" not in st.session_state:
    st.session_state.description = False  # Прапорець відображення опису
if "doc_mode" not in st.session_state:
    st.session_state.doc_mode = "view"  # Режим роботи з документами (перегляд або чат)
if "audio_mode" not in st.session_state:
    st.session_state.audio_mode = "view"  # Режим роботи з аудіо (перегляд або чат)
if "video_mode" not in st.session_state:
    st.session_state.video_mode = "view"  # Режим роботи з відео (перегляд або чат)
if "image_mode" not in st.session_state:
    st.session_state.image_mode = "view"  # Режим роботи з зображеннями (перегляд або чат)
if 'image_context_text' not in st.session_state:
    st.session_state.image_context_text = {'context': ''}  # Зберігає контекст для зображень
if 'video_context_text' not in st.session_state:
    st.session_state.video_context_text = {'context': ''}  # Зберігає контекст для відео
if 'document_context_text' not in st.session_state:
    st.session_state.document_context_text = {'context': ''}  # Зберігає контекст для документів

if 'audio_context_text' not in st.session_state:
    st.session_state.audio_context_text = {'context': ''}  # Зберігає контекст для аудіо
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 0
if 'chunk_overlap' not in st.session_state:
    st.session_state.chunk_overlap = 0
if 'ret_k_results' not in st.session_state:
    st.session_state.ret_k_results = 0
if 'video_summary' not in st.session_state:
    st.session_state.video_summary = False  # Прапорець відображення узагальненої інформації
if 'audio_summary' not in st.session_state:
    st.session_state.audio_summary = False  # Прапорець відображення узагальненої інформації
if 'image_summary' not in st.session_state:
    st.session_state.image_summary = False  # Прапорець відображення узагальненої інформації
if 'document_summary' not in st.session_state:
    st.session_state.document_summary = False  # Прапорець відображення узагальненої інформації
if 'youtube_video' not in st.session_state:
    st.session_state.youtube_video = False
if 'local_video' not in st.session_state:
    st.session_state.local_video = False


def select_database():
    """Функція для вибору бази даних та скидання інших станів.
    Скидає стан чату, вимикає опис та встановлює прапорець вибору бази даних,
    одночасно скидаючи інші прапорці вибору."""
    reset_chat()  # Скидання стану чату
    st.session_state.description = False  # Вимкнення опису
    st.session_state.database_selected = True  # Встановлення прапорця вибору бази даних
    st.session_state.document_selected = False  # Скидання прапорця вибору документа
    st.session_state.video_selected = False  # Скидання прапорця вибору відео
    st.session_state.image_selected = False  # Скидання прапорця вибору зображення
    st.session_state.start_chat = False  # Скидання прапорця початку чату


def select_document():
    """Функція для вибору режиму роботи з документами.
    Скидає стан чату, вимикає опис, встановлює поточний режим роботи
    та відповідні прапорці для роботи з документами."""
    reset_chat()  # Скидання стану чату
    st.session_state.description = False  # Вимкнення опису
    st.session_state.current_mode = "document_mode"  # Встановлення поточного режиму
    st.session_state.document_selected = True  # Встановлення прапорця вибору документа
    st.session_state.video_selected = False  # Скидання прапорця вибору відео
    st.session_state.image_selected = False  # Скидання прапорця вибору зображення
    st.session_state.start_chat = False  # Скидання прапорця початку чату
    st.session_state.database_selected = False  # Скидання прапорця вибору бази даних
    st.session_state.document_processed = False  # Скидання прапорця обробки документа
    st.session_state.audio_selected = False  # Скидання прапорця обробки аудіо
  # Скидання прапорця обробки зображення

def select_video():
    """Функція для вибору режиму роботи з відео.
    Скидає стан чату, вимикає опис, встановлює поточний режим роботи
    та відповідні прапорці для роботи з відео."""
    reset_chat()  # Скидання стану чату
    st.session_state.description = False  # Вимкнення опису
    st.session_state.current_mode = "video_mode"  # Встановлення поточного режиму
    st.session_state.document_selected = False  # Скидання прапорця вибору документа
    st.session_state.video_selected = True  # Встановлення прапорця вибору відео
    st.session_state.image_selected = False  # Скидання прапорця вибору зображення
    st.session_state.start_chat = False  # Скидання прапорця початку чату
    st.session_state.database_selected = False  # Скидання прапорця вибору бази даних
    st.session_state.video_processed = False 
    st.session_state.audio_selected = False  # Скидання прапорця обробки аудіо

def select_image():
    """Функція для вибору режиму роботи з зображеннями.
    Скидає стан чату, вимикає опис, встановлює поточний режим роботи
    та відповідні прапорці для роботи з зображеннями."""
    reset_chat()  # Скидання стану чату
    st.session_state.description = False  # Вимкнення опису
    st.session_state.current_mode = "image_mode"  # Встановлення поточного режиму
    st.session_state.document_selected = False  # Скидання прапорця вибору документа
    st.session_state.video_selected = False  # Скидання прапорця вибору відео
    st.session_state.image_selected = True  # Встановлення прапорця вибору зображення
    st.session_state.start_chat = False  # Скидання прапорця початку чату
    st.session_state.database_selected = False  # Скидання прапорця вибору бази даних
    st.session_state.image_processed = False
    st.session_state.audio_selected = False  # Скидання прапорця обробки зображення

def select_audio():
    """Функція для вибору режиму роботи з аудіо.
    Скидає стан чату, вимикає опис, встановлює поточний режим роботи
    та відповідні прапорці для роботи з аудіо."""
    reset_chat()  # Скидання стану чату
    st.session_state.description = False  # Вимкнення опису
    st.session_state.current_mode = "audio_mode"  # Встановлення поточного режиму
    st.session_state.document_selected = False  # Скидання прапорця вибору документа
    st.session_state.video_selected = False  # Скидання прапорця вибору відео
    st.session_state.image_selected = False  # Скидання прапорця вибору зображення
    st.session_state.audio_selected = True  # Встановлення прапорця вибору аудіо
    st.session_state.start_chat = False  # Скидання прапорця початку чату
    st.session_state.database_selected = False  # Скидання прапорця вибору бази даних
    st.session_state.image_processed = False  # Скидання прапорця обробки зображення


def start_chat():
    """Функція для запуску режиму чату.
    Скидає стан чату, вимикає опис, встановлює поточний режим роботи
    та відповідні прапорці для роботи в режимі чату."""
    reset_chat()  # Скидання стану чату
    st.session_state.description = False  # Вимкнення опису
    st.session_state.current_mode = "main_chat"  # Встановлення поточного режиму
    st.session_state.start_chat = True  # Встановлення прапорця початку чату
    st.session_state.document_selected = False  # Скидання прапорця вибору документа
    st.session_state.video_selected = False  # Скидання прапорця вибору відео
    st.session_state.image_selected = False  # Скидання прапорця вибору зображення
    st.session_state.database_selected = False  # Скидання прапорця вибору бази даних
    st.session_state.audio_selected = False  # Скидання прапорця вибору аудіо

def stop_chat():
    """Функція для зупинки режиму чату.
    Скидає прапорець початку чату, що призводить до виходу з режиму чату."""
    st.session_state.start_chat = False  # Скидання прапорця початку чату

def greet_once(text: str):
    """Функція для відображення привітання один раз.
    
    Аргументи:
        text (str): Текст привітання, який буде відображено
    
    Додає повідомлення привітання від асистента в історію чату,
    якщо користувач ще не був привітаний."""
    if not st.session_state.get("greeted"):  # Перевірка, чи було вже привітання
        st.session_state.messages.append({"role": "assistant", "content": text})  # Додавання привітання
        st.session_state.greeted = True  # Встановлення прапорця привітання

# Ініціалізація змінних для опису та заголовка додатку
current_description = APP_DESCRIPTION  # Опис додатку з конфігураційного файлу
current_title = "Агент для роботи з данними"  # Заголовок додатку
if st.session_state.mode == "menu":  # Якщо режим - меню
    st.session_state.title=current_title  # Встановлення заголовка
    st.session_state.description = current_description  # Встановлення опису
    st.title(st.session_state.title)  # Відображення заголовка
    st.markdown(st.session_state.description)  # Відображення опису
# --------------------------------------------------------------
# Кнопки переміщені в sidebar для кращої організації інтерфейсу
with st.sidebar:
    # Ініціалізація клієнта Milvus, якщо він ще не створений
    if "milvus_client" not in st.session_state:
        st.session_state.milvus_client = MilvusClient(uri="http://127.0.0.1:19530")  # Створення клієнта
        client = st.session_state.milvus_client  # Збереження посилання на клієнта
        st.session_state.collections = client.list_collections()  # Отримання списку колекцій
    
    # Вибір моделі мови для використання
    llm_option = st.selectbox(
        "Яку модель використовувати?",
        ("qwen3:14b", "Україномовну", "gemma3:27b", "gemma3:12b", "qwen3:30b", "qwen3:30b-a3b"),)  # Опції вибору моделі
    st.session_state.llm_option = llm_option  # Збереження вибраної моделі
    st.markdown(f'Ви обрали модель: {st.session_state.llm_option}')
    if st.session_state.llm_option == "Україномовну":
        # Якщо вибрана україномовна модель
        st.session_state.ukr_generator = create_ukr_llm()  # Створення україномовної моделі
        st.session_state.llm_option = llm_option  # Повторне збереження вибраної моделі


    choise_system_prompt = st.selectbox(
        "Який системний промт використовувати?",
        ("проповедник", "аналитик",))  # Опції вибору моделі
    if choise_system_prompt == "проповедник":
        st.session_state.system_prompt = RELIGIOUS_SYSTEM_PROMPT
    elif choise_system_prompt == "аналитик":
        st.session_state.system_prompt = REGULAR_SYSTEM_PROMPT
    st.markdown(f'Ви обрали режим роботи: {choise_system_prompt}')
   

    st.select_slider('Розмір фрагменту',  options=[256, 512, 1024, 2048, 4096], key="chunk_size")
    st.select_slider('Перекриття фрагментів',  options=[128, 256, 512, 1024, 2048], key="chunk_overlap")
    st.select_slider('Кількість результатів',  options=[1, 5, 10, 20, 30, 40, 50], key="ret_k_results")


    # Введення назви колекції для роботи
    collection_name = st.text_input("Введи назву колекції")
    st.write('Існують колекції: ')  # Заголовок для списку колекцій
    st.session_state.mode = "collections"  # Встановлення режиму перегляду колекцій
    for i in st.session_state.collections:  # Перебір усіх колекцій
        st.markdown(f'Колекція: {i}')  # Відображення назви колекції

    if collection_name:                                        # Якщо користувач ввів назву колекції
        st.session_state.collection_name = collection_name     # Збереження назви колекції
        dense_ef = create_bge_m3_embeddings()  # Створення функції вбудовування
        retr = HybridRetriever(  # Створення гібридного пошуковика
            client=st.session_state.milvus_client,  # Клієнт Milvus
            collection_name=collection_name,  # Назва колекції
            dense_embedding_function=dense_ef,  # Функція вбудовування
        )
        retr.build_collection()                      # Створення або завантаження колекції
        st.session_state.collections = (             # Оновлення списку колекцій у sidebar
            st.session_state.milvus_client.list_collections()
        )
        st.header("Меню")  # Заголовок меню
        st.button("Обробка документу",  on_click=select_document, key="btn_doc")  # Кнопка для обробки документа
        st.button("Обробка відео",     on_click=select_video,   key="btn_vid")  # Кнопка для обробки відео
        st.button("Обробка зображення",on_click=select_image,   key="btn_img")  # Кнопка для обробки зображення
        st.button("Обробка аудіо", on_click=select_audio,   key="btn_audio")  # Кнопка для обробки аудіо
        st.button("Почати діалог", on_click=start_chat,     key="btn_chat")  # Кнопка для початку діалогу
        st.divider()  # Розділювач
        st.write(f"Поточний режим: {st.session_state.current_mode}")  # Відображення поточного режиму

# Відображення вибраної колекції
st.markdown(f"Ви обрали колекцію {collection_name}")
print(torch.cuda.is_available(), 'cuda')
# Режим чату
if st.session_state.start_chat:
    collection_name = st.session_state.collection_name  # Отримання назви колекції
    st.session_state.mode = st.session_state.current_mode  # Встановлення режиму
    st.title("Пошук по колекції")  # Заголовок
    main_chat(collection_name)


# Режим роботи з документами
if st.session_state.document_selected:
    collection_name = st.session_state.collection_name  # Отримання назви колекції
    st.session_state.mode = st.session_state.current_mode  # Встановлення режиму

    if st.session_state.doc_mode == "view":  # Якщо режим перегляду документа
        st.title("Робота з документами")  # Заголовок
        st.button(
                    "Узагальнити інформацію",
                    on_click=on_summarise_document,
                    # Оновлення режиму
                    key="summary_document",  # Ключ кнопки
                )# Заголовок
        st.markdown(f"Узагальнити інформацію: {st.session_state.document_summary}")
        document_mode(collection_name, st.session_state.document_summary)  # Виклик функції обробки документа
        if st.session_state.document_processed:  # Якщо документ оброблений
            st.button(
                "Почати чат",  # Кнопка для початку чату
                on_click=lambda: st.session_state.update(doc_mode="chat"),  # Оновлення режиму
                key="start_doc_chat",  # Ключ кнопки
            )

    elif st.session_state.doc_mode == "chat":  # Якщо режим чату з документом
        chat_document_mode(collection_name, st.session_state.llm_option)  # Виклик функції чату з документом
        st.button("Новий документ", on_click=lambda: st.session_state.update(doc_mode="view"), key="new_doc",)  # Кнопка для нового документа
        

# Режим роботи з відео
if st.session_state.video_selected:
    collection_name = st.session_state.collection_name  # Отримання назви колекції
    st.session_state.mode = st.session_state.current_mode  # Встановлення режиму
    if st.session_state.video_mode == "view":  # Якщо режим перегляду відео
        st.title('Робота з відео')
        st.button(
                    "Узагальнити інформацію",
                    on_click=on_summarise_video,
                    # Оновлення режиму
                    key="summary_video",  # Ключ кнопки
                )# Заголовок
        st.markdown(f"Узагальнити інформацію: {st.session_state.video_summary}")
        video_mode(collection_name, summary = st.session_state.video_summary)  # Виклик функції обробки відео
        if st.session_state.video_processed:  # Якщо відео оброблене
            st.button(
                    "Почати чат",  # Кнопка для початку чату
                    on_click=lambda: st.session_state.update(video_mode="chat"),  # Оновлення режиму
                    key="start_video_chat",  # Ключ кнопки
                )

    elif st.session_state.video_mode == "chat":  # Якщо режим чату з відео
        chat_video_mode(collection_name, st.session_state.llm_option)  # Виклик функції чату з відео
        st.button("Нове відео", on_click=lambda: st.session_state.update(video_mode="view"), key="new_video",)  # Кнопка для нового відео


# Режим роботи з зображеннями
if st.session_state.image_selected:
    collection_name = st.session_state.collection_name  # Отримання назви колекції
    collection_name = st.session_state.collection_name
    st.session_state.mode = st.session_state.current_mode
    if st.session_state.image_mode == "view":
        st.title('Робота з зображеннями')
        st.button(
                    "Узагальнити інформацію",
                    on_click=on_summarise_image,
                    # Оновлення режиму
                    key="summary_image",  # Ключ кнопки
                )# Заголовок
        st.markdown(f"Узагальнити інформацію: {st.session_state.image_summary}")
        image_mode(collection_name, st.session_state.image_summary)
        if st.session_state.image_processed:
                st.button(
                    "Почати чат",
                    on_click=lambda: st.session_state.update(image_mode="chat"),
                    key="start_image_chat",
                )
    elif st.session_state.image_mode == "chat":
        chat_image_mode(collection_name, st.session_state.llm_option)
        st.button("Нове зображення", on_click=lambda: st.session_state.update(image_mode="view"), key="new_image",)
        


if st.session_state.audio_selected:
    collection_name = st.session_state.collection_name
    st.session_state.mode = st.session_state.current_mode
    if st.session_state.audio_mode == "view":
        st.title('Робота з аудіо')
        st.button(
                    "Узагальнити інформацію",
                    on_click=on_summarise_audio,
                    # Оновлення режиму
                    key="summary_audio",  # Ключ кнопки
                )# Заголовок
        st.markdown(f"Узагальнити інформацію: {st.session_state.audio_summary}")
        audio_mode(collection_name, st.session_state.audio_summary)
        if st.session_state.audio_processed:
                st.button(
                    "Почати чат",
                    on_click=lambda: st.session_state.update(audio_mode="chat"),
                    key="start_audio_chat",
                )
    elif st.session_state.audio_mode == "chat":
        chat_audio_mode(collection_name, st.session_state.llm_option)
        st.button("Нове аудіо", on_click=lambda: st.session_state.update(audio_mode="view"), key="new_audio",)


