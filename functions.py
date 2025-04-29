# Імпорт необхідних бібліотек для роботи з Milvus - векторною базою даних для пошуку схожих документів
from milvus_model.hybrid import BGEM3EmbeddingFunction
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# Імпорт бібліотек для роботи з LLM (великими мовними моделями)
from PIL import Image  # Для обробки зображень
from langchain_ollama import ChatOllama  # Для взаємодії з Ollama API
from langchain_core.prompts import ChatPromptTemplate  # Для створення шаблонів промптів
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Для розбиття тексту на частини
# Імпорт для ранжування результатів пошуку
from pymilvus.model.reranker import BGERerankFunction  # Функція для переранжування результатів пошуку
# Імпорт для обробки тексту та зображень
from transformers import pipeline as hf_pipeline  # Конвеєр для обробки тексту та зображень
# Імпорт для роботи з документами різних форматів
from PyPDF2 import PdfReader  # Для читання PDF-файлів
from docx import Document as DocxDocument  # Для роботи з Word-документами
# Імпорт стандартних бібліотек Python
from datetime import datetime as dt  # Для роботи з датою та часом
from pathlib import Path  # Для роботи з шляхами файлів
from typing import Any, Dict  # Для типізації
import re  # Для роботи з регулярними виразами
import logging  # Для логування
import streamlit as st  # Для створення веб-інтерфейсу
import ollama  # Для взаємодії з Ollama API
import json  # Для роботи з JSON
import torch  # Для роботи з тензорами та нейромережами
import yt_dlp  # Для завантаження відео з YouTube
import hashlib  # Для створення хешів
import whisper  # Для транскрибації аудіо
import os  # Для роботи з операційною системою
import uuid  # Для генерації унікальних ідентифікаторів
import pandas as pd  # Для роботи з табличними даними
import base64  # Для кодування даних у base64
import io  # Для роботи з потоками введення-виведення
import sqlite3  # Для роботи з SQLite базами даних
from tqdm import tqdm  # Для відображення прогресу
from datetime import datetime  # Для роботи з датою та часом
from config import *  # Імпорт всіх змінних з файлу конфігурації
from rag import *  # Імпорт всіх функцій з файлу rag.py
# Налаштування логування для відстеження роботи програми
logger = logging.getLogger(__name__)

# Ініціалізація моделі для створення ембедінгів (векторних представлень) тексту
EMBEDDER = BGEM3EmbeddingFunction(
    model_name="BAAI/bge-m3",  # Назва моделі для ембедінгів
    device="cpu",        # Використання CPU для обчислень (найбезпечніший варіант)
    use_fp16=False       # Відключення використання половинної точності (fp16)
)

def reset_chat():
    """Скидає історію чату в сесії Streamlit, видаляючи всі повідомлення та стан привітання"""
    st.session_state.messages      = []  # Очищення списку повідомлень
    st.session_state.chat_history  = []  # Очищення історії чату
    st.session_state.pop("greeted", None)  # Видалення стану привітання


def create_bge_m3_embeddings():
    """Створює функцію для генерації ембедінгів BGE-M3, повертаючи ініціалізований об'єкт EMBEDDER"""
    bge_m3_ef = EMBEDDER  # Використання глобального об'єкта EMBEDDER
    
    return bge_m3_ef

def create_ukr_llm():
    """Створює та повертає українську мовну модель на основі MamayLM-Gemma для генерації тексту"""
    return hf_pipeline(
        task="text-generation",  # Завдання - генерація тексту
        model="INSAIT-Institute/MamayLM-Gemma-2-9B-IT-v0.1",  # Українська модель
        torch_dtype=torch.bfloat16,  # Використання bfloat16 для оптимізації пам'яті
        device_map="auto",  # Автоматичне визначення пристрою (CPU/GPU)
    )



def create_llm(model_name):
    """Створює екземпляр моделі LLM з вказаною назвою, налаштовуючи температуру для детермінованих відповідей"""
    llm = ChatOllama(
        model=model_name,  # Назва моделі для використання
        temperature=0,  # Температура 0 для детермінованих відповідей
    )
    return llm



def rerank_search(documents, query, limit=10):
    """Перераховує результати пошуку за допомогою BGE-reranker для покращення релевантності"""
    bge_rf = BGERerankFunction(
        model_name="BAAI/bge-reranker-v2-m3",  # Модель для переранжування
        device="cpu",  # Використання CPU
        top_k=3  # Кількість найкращих результатів для повернення
    )
    reranked_results = bge_rf(query, documents)  # Переранжування результатів
    return reranked_results


def create_prompt(system_prompt):
    """Створює шаблон промпту для чат-моделі з системним промптом та місцями для контексту і питання"""
    return ChatPromptTemplate.from_messages([
        ("system",
         system_prompt),  # Системний промпт для налаштування поведінки моделі
        ("human",
         "Context:\n{context}\n\nQuestion: {question}")  # Шаблон для запиту користувача
    ])

def create_chain(llm, prompt):
    """Створює ланцюжок обробки з LLM та промпту для послідовної обробки запитів"""
    chain = prompt | llm  # Об'єднання промпту та моделі в ланцюжок
    return chain

def get_llm_context(chain, context):
    """Отримує відповідь від моделі на основі наданого контексту без конкретного питання"""
    response = chain.invoke(
        {
            "text": context,  # Передача контексту в ланцюжок
        }
    )
    return response


def get_llm_response(chain, question, context):
    """Отримує відповідь від моделі на основі питання та контексту"""
    response = chain.invoke(
    {
        "question": question,  # Питання користувача
        "context": context,  # Контекст для відповіді
    })
    return response


# Базове налаштування логування для відстеження роботи програми
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dir(dir_name):
    """Створює директорію з вказаною назвою, якщо вона не існує"""
    folder = Path(dir_name)  # Створення об'єкта Path
    folder.mkdir(parents=True, exist_ok=True)  # Створення директорії з батьківськими директоріями

def create_logger(logger_name = __name__):
    """Створює логер з вказаною назвою та рівнем логування INFO"""
    logger = logging.getLogger(logger_name)  # Створення логера
    logger.setLevel(logging.INFO)  # Встановлення рівня логування
    return logger

def extract_video_id(url: str):
    """Витягує ідентифікатор відео з URL YouTube за допомогою регулярного виразу"""
    youtube_regex = (
    r'(https?://)?(www\.)?'
    '(youtube|youtu|youtube-nocookie)\\.(com|be)/'
    '(watch\\?v=|embed/|v/|.+\\?v=)?([^&=%\\?]{11})')  # Регулярний вираз для пошуку ID відео
    match = re.match(youtube_regex, url)  # Пошук співпадіння
    if match:
        return match.group(6)  # Повернення ID відео
    return ""  # Повернення порожнього рядка, якщо ID не знайдено

def clean_filename(filename: str) -> str:
    """Очищення імені файлу від недопустимих символів для безпечного збереження"""
    # Видаляємо недопустимі символи для імен файлів
    return re.sub(r'[\\/*?:"<>|]', "", filename)  # Заміна недопустимих символів на порожній рядок

def query_ollama(prompt, image_base64, model):
    """Запитує Ollama з зображенням та промптом для отримання відповіді на основі зображення"""
    response = ollama.chat(
        model=model,  # Модель для використання
        messages=[{
            'role': 'user',  # Роль користувача
            'content': prompt,  # Текст промпту
            'images': [image_base64]  # Зображення у форматі base64
        }]
    )
    return response['message']['content']  # Повернення відповіді моделі

def image_to_base64(image):
    """Конвертує зображення у формат base64 для передачі в API"""
    if image.mode == "RGBA":
        image = image.convert("RGB")  # Прибираємо альфа-канал для сумісності
    img_byte_arr = io.BytesIO()  # Створення буфера для зображення
    image.save(img_byte_arr, format="JPEG")  # Збереження зображення у форматі JPEG
    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")  # Кодування в base64

def generate_caption(image_path: str):
    """Генерує опис для зображення за допомогою моделі Gemma, аналізуючи вміст зображення"""

    pipe = pipeline(
        "image-text-to-text",  # Тип завдання - генерація тексту на основі зображення
        model="google/gemma-3-4b-it",  # Модель для використання
        device="cpu",  # Використання CPU
        torch_dtype=torch.bfloat16  # Використання bfloat16 для оптимізації пам'яті
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]  # Системний промпт
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},  # Шлях до зображення
                {"type": "text", "text": "What animal is on the candy?"}  # Питання про зображення
            ]
        }
    ]


    output = pipe(text=messages, max_new_tokens=200)  # Генерація відповіді
    print(output[0]["generated_text"][-1]["content"])  # Виведення відповіді


@st.cache_data(show_spinner="Транскрибуємо відео…")
def process_video(url: str, video_dir: str, model_size: str):
    """Обробляє відео з YouTube: завантажує, транскрибує та розбиває на частини для подальшого аналізу"""
    whisper_model = get_whisper(model_size)  # Отримання моделі Whisper (кешується)
    logger = logging.getLogger(__name__)  # Створення логера
    
    video_chunks = []  # Список для зберігання частин відео
    video_id = extract_video_id(url)  # Витягування ID відео
    unique_video_id = f"{video_id}_{hashlib.md5(url.encode()).hexdigest()[:8]}"  # Створення унікального ID
    if not video_id:
        raise ValueError("Неправильний URL відео YouTube")  # Перевірка коректності URL
    logger.info(f"Обробка відео з URL: {url}")  # Логування початку обробки
    try:
        # Створюємо об'єкт Path для директорії
        video_dir_path = Path(video_dir)  # Створення об'єкта Path
        video_dir_path.mkdir(parents=True, exist_ok=True)  # Створення директорії

        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            logger.info("Отримання інформації про відео...")  # Логування
            info = ydl.extract_info(url, download=False)  # Отримання інформації про відео
            # Отримуємо та очищуємо назву
            title = info.get("title", "video")  # Отримання назви відео
            safe_title = clean_filename(title)  # Очищення назви
            unique_safe_title = f"{safe_title}_{hashlib.md5(url.encode()).hexdigest()[:8]}"  # Створення унікальної назви
            logger.info(f"Отримано назву відео: {title}, очищена назва: {safe_title}")  # Логування

            video_info = {
                    "title": title,  # Назва відео
                    "uploader": info.get("uploader", "Невідомий автор"),  # Автор відео
                    "duration": info.get("duration", 0),  # Тривалість відео
                    "upload_date": info.get("upload_date", ""),  # Дата завантаження
                    "view_count": info.get("view_count", 0),  # Кількість переглядів
                    "like_count": info.get("like_count", 0),  # Кількість лайків
                    "description": info.get("description", ""),  # Опис відео
                    "video_id": video_id}  # ID відео
            logger.info(f"Знайдено відео: {video_info['title']} ({video_info['duration']} сек)")  # Логування

            final_video_path = video_dir_path / f"{safe_title}.mp4"  # Шлях до відео
            final_audio_path = video_dir_path / f"{safe_title}.mp3"  # Шлях до аудіо
            transcript_path = video_dir_path / f"{safe_title}_transcript.txt"  # Шлях до транскрипції
            
            logger.info(f"Шляхи до файлів:")  # Логування
            logger.info(f"Відео: {final_video_path}")  # Логування
            logger.info(f"Аудіо: {final_audio_path}")  # Логування
            logger.info(f"Транскрипція: {transcript_path}")  # Логування

            ydl_opts = {
                'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',  # Формат відео
                'paths': {'home': str(video_dir_path)},  # Шлях для збереження
                'outtmpl': {'default': f'{safe_title}.%(ext)s'},  # Шаблон імені файлу
                'postprocessors': [
                    {'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4'},  # Конвертація у mp4
                    {'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}  # Витягування аудіо
                ],
                'merge_output_format': 'mp4',  # Формат виходу
                'keepvideo': True,  # Зберігання відео
                'quiet': True,  # Тихий режим
                'noplaylist': True  # Без плейлистів
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                if not final_video_path.exists() or not final_audio_path.exists():
                    logger.info("Починаємо завантаження відео...")  # Логування
                    ydl_download.download([url])  # Завантаження відео
                    logger.info("Завантаження завершено")  # Логування
                else:
                    logger.info("Файли вже існують, пропускаємо завантаження")  # Логування

            if not final_video_path.exists():
                logger.error(f"Відео файл не знайдено після завантаження: {final_video_path}")  # Логування помилки
                raise FileNotFoundError(f"Відео файл не знайдено: {final_video_path}")  # Виклик помилки

            logger.info("Починаємо транскрибацію...")  # Логування
            result = whisper_model.transcribe(str(final_video_path))  # Транскрибація відео
            transcript = result["text"]  # Отримання тексту транскрипції
            logger.info(f"Транскрибація завершена, отримано {len(transcript)} символів")  # Логування
            
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript)  # Запис транскрипції у файл
            logger.info(f"Транскрипцію збережено")  # Логування

            logger.info("Підготовка тексту...")  # Логування
            texts = prepare_text(transcript)  # Підготовка тексту
            logger.info(f"Текст підготовлено, довжина: {len(texts)}")  # Логування
            
            logger.info("Розбиття тексту на частини...")  # Логування
            texts = chunk_text(texts)  # Розбиття тексту на частини
            logger.info(f"Текст розбито на {len(texts)} частин")  # Логування

            for i, text_chunk in enumerate(texts):
                chunk = {
                    "pk": i,  # Первинний ключ
                    "title": video_info["title"],  # Назва відео
                    "uploader": video_info["uploader"],  # Автор відео
                    "duration": video_info["duration"],  # Тривалість відео
                    "upload_date": video_info["upload_date"],  # Дата завантаження
                    "view_count": video_info["view_count"],  # Кількість переглядів
                    "like_count": video_info["like_count"],  # Кількість лайків
                    "description": text_chunk,  # Опис (частина тексту)
                    "text": text_chunk,  # Текст частини
                    "metadata": {
                        "video_id": video_id,  # ID відео
                        "timestamp": dt.now().isoformat(),  # Часова мітка
                        "url": url,  # URL відео
                        "video_path": str(final_video_path),  # Шлях до відео
                        "audio_path": str(final_audio_path),  # Шлях до аудіо
                        "transcript_path": str(transcript_path)  # Шлях до транскрипції
                    }
                }
                video_chunks.append(chunk)  # Додавання частини до списку

            logger.info(f"Створено {len(video_chunks)} документів")  # Логування
            return video_chunks, transcript, final_video_path, safe_title, video_id, title, unique_video_id, info, unique_safe_title, url 
    

    except Exception as e:
        logger.error(f"Помилка при обробці відео: {str(e)}", exc_info=True)  # Логування помилки
        raise


# Функції для екстракції даних з різних типів файлів
def data_extraction(path):
    """Визначає тип файлу за розширенням та викликає відповідну функцію екстракції"""
    ext = path.rsplit(".", 1)[-1].lower()  # Отримання розширення файлу
    if ext == "pdf":
        return extract_from_pdf(path)  # Екстракція з PDF
    if ext in ("xlsx", "xls"):
        return extract_from_excel(path)  # Екстракція з Excel
    if ext in ("doc", "docx"):
        return extract_from_word(path)  # Екстракція з Word
    if ext == "md":
        return extract_from_markdown(path)  # Екстракція з Markdown
    if ext == "csv":
        return extract_from_csv(path)  # Екстракція з CSV
    raise ValueError(f"Неподдерживаемый формат: {ext}")  # Виклик помилки для непідтримуваних форматів

def extract_from_pdf(p):
    """Витягує текст з PDF-файлу, об'єднуючи текст з усіх сторінок"""
    txt = ""  # Порожній рядок для зберігання тексту
    with open(p, "rb") as f:
        for pg in PdfReader(f).pages:  # Перебір сторінок
            txt += pg.extract_text() or ""  # Додавання тексту сторінки
    return txt  # Повернення тексту

def extract_from_excel(p, **kw):
    """Витягує дані з Excel-файлу, перетворюючи їх на DataFrame"""
    return pd.read_excel(p, dtype=str, **kw).fillna("")  # Читання Excel-файлу та заповнення пропусків

def extract_from_word(p):
    """Витягує текст з Word-документа, об'єднуючи текст з усіх параграфів"""
    return "\n".join(par.text for par in DocxDocument(p).paragraphs)  # Об'єднання тексту параграфів

def extract_from_markdown(p):
    """Витягує текст з Markdown-файлу, читаючи його як текстовий файл"""
    return open(p, encoding="utf-8").read()  # Читання файлу

def extract_from_csv(p, **kw):
    """Витягує дані з CSV-файлу, перетворюючи їх на DataFrame"""
    return pd.read_csv(p, dtype=str, **kw).fillna("")  # Читання CSV-файлу та заповнення пропусків

def prepare_text(text: Any) -> str:
    """
    Підготовка тексту (DataFrame, dict, або інший тип) до обробки, перетворюючи його на чистий рядок
    """
    # 1) Якщо DataFrame — склеюємо всі комірки
    if isinstance(text, pd.DataFrame):
        # об'єднуємо всі значення в один рядок
        text = ' '.join(text.values.flatten().astype(str))  # Об'єднання всіх значень
    # 2) Якщо dict — склеюємо всі значення
    elif isinstance(text, dict):
        text = ' '.join(str(v) for v in text.values())  # Об'єднання всіх значень
    # 3) Якщо не рядок — приводимо до рядка
    elif not isinstance(text, str):
        text = str(text)  # Перетворення на рядок

    # 4) Очищаємо отриманий рядок
    # видаляємо переведення рядків, табуляції та зайві пробіли
    txt = text.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')  # Заміна спеціальних символів
    txt = ' '.join(txt.split())  # прибираємо подвійні/зайві пробіли

    return txt.strip()  # Повернення очищеного рядка

def chunk_text(text: str, size=1024, overlap=256):
    """Розбиває текст на частини вказаного розміру з перекриттям для кращого пошуку"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,  # Розмір частини
        chunk_overlap=overlap,  # Перекриття між частинами
        length_function=len,  # Функція для визначення довжини
        is_separator_regex=False  # Без використання регулярних виразів
    )
    return splitter.split_text(text)  # Розбиття тексту


def build_json(text: str, original_filename: str, path: str) -> dict:
    """Створює JSON-структуру для тексту з метаданими для зберігання в Milvus"""
    original_uuid = hashlib.sha256(text.encode()).hexdigest()  # Створення унікального ID
    doc_id = original_uuid[:16]  # Фіксований ідентифікатор (перші 16 символів)
    chunks = chunk_text(text)  # Розбиття тексту на частини
    upload_date = dt.now().isoformat()  # Поточна дата та час
    return {
        "doc_id": doc_id,  # ID документа
        "original_uuid": original_uuid,  # Повний унікальний ID
        "content": text,  # Повний текст
        "chunks": [
            {
                "chunk_id": f"{doc_id}_c{i}",  # ID частини (детермінований)
                "original_index": i,  # Індекс частини
                "content": c,  # Текст частини
                "file_path": path,  # Шлях до файлу
                "file_name": original_filename,  # Оригінальна назва файлу
                "upload_date": upload_date,  # Дата завантаження
            }
            for i, c in enumerate(chunks)  # Перебір частин
        ],
    }


def build_image_json(text: str, file_path: str, original_filename: str) -> dict:
    """Створює JSON-структуру для зображення з описом та метаданими для зберігання в Milvus"""
    txt = prepare_text(text)  # Підготовка тексту

    original_uuid = hashlib.sha256(txt.encode()).hexdigest()  # Створення унікального ID
    doc_id = original_uuid[:16]  # Фіксований ідентифікатор (перші 16 символів)
    chunks = chunk_text(txt)  # Розбиття тексту на частини
    upload_date = dt.now().isoformat()  # Поточна дата та час

    return {
        "doc_id": doc_id,  # ID документа
        "original_uuid": original_uuid,  # Повний унікальний ID
        "content": txt,  # Повний текст
        "chunks": [
            {
                "chunk_id": f"{doc_id}_c{i}",  # ID частини (детермінований)
                "content": c,  # Текст частини
                "file_path": file_path,  # Шлях до файлу
                "file_name": original_filename,  # Оригінальна назва файлу
                "upload_date": upload_date,  # Дата завантаження
                "original_index": i,  # Індекс частини
            }
            for i, c in enumerate(chunks)  # Перебір частин
        ],
    }



def build_video_json(
    txt: str,
    meta: Dict[str, Any],
    file_path: str,
    original_filename: str,
    chunk_size: int = 1024,
    overlap: int = 256,
) -> Dict[str, Any]:
    """
    Перетворює розшифровку відео в структуру JSON для зберігання в Milvus,
    включаючи метадані відео та розбиття тексту на частини
    """
    file_path = str(file_path)  # Перетворення шляху на рядок
    original_uuid = hashlib.sha256(txt.encode()).hexdigest()  # Створення унікального ID
    doc_id = f"vid_{meta['unique_video_id']}"  # Детермінований ID
    chunks = chunk_text(txt, size=chunk_size, overlap=overlap)  # Розбиття тексту на частини
    upload_date = dt.now().isoformat()  # Поточна дата та час

    return {
        "doc_id": doc_id,
        "original_uuid": original_uuid,
        "video_meta": {                               # всі метадані відео
            "video_id":     meta["video_id"],
            "unique_video_id":    meta["unique_video_id"],
            "title":        meta.get("title", ""),
            "uploader":     meta.get("uploader", ""),
            "duration":     meta.get("duration", 0),
            "upload_date":  meta.get("upload_date", ""),
            "view_count":   meta.get("view_count", 0),
            "like_count":   meta.get("like_count", 0),
            "url":          meta.get("url", ""),
        },
        "content": txt,
        "chunks": [
            {
                "chunk_id":       f"{doc_id}_chunk_{i}",
                "original_index": i,
                "content":        c,
                "file_path":      file_path,
                "file_name":      original_filename,
                "upload_date":    upload_date,
            }
            for i, c in enumerate(chunks)
        ],
    }


def init_db():
    """
    Ініціалізує базу даних, створюючи таблицю history, якщо вона не існує.
    Таблиця містить поля:
    - id: унікальний ідентифікатор запису
    - ts: часова мітка взаємодії
    - mode: режим взаємодії (документ, зображення, відео, чат)
    - query: запит користувача
    - answer: відповідь системи
    """
    conn = sqlite3.connect(DB_PATH)  # Підключення до бази даних
    cur = conn.cursor()  # Створення курсора для виконання SQL-запитів
    cur.execute(
        """CREATE TABLE IF NOT EXISTS history(
                id     INTEGER PRIMARY KEY AUTOINCREMENT,  
                ts     TEXT,  
                mode   TEXT,  
                query  TEXT,  
                answer TEXT   
        )"""
    )
    conn.commit(); conn.close()  # Збереження змін та закриття з'єднання

def get_conn():
    """Створює з'єднання з базою даних та ініціалізує таблицю, якщо потрібно"""
    conn = sqlite3.connect(DB_PATH)
    # створюємо таблицю один раз
    conn.execute(
        """CREATE TABLE IF NOT EXISTS history (
               id     INTEGER PRIMARY KEY AUTOINCREMENT,
               ts     TEXT,
               mode   TEXT,
               query  TEXT,
               answer TEXT
           )"""
    )
    return conn

def log_interaction(mode, query, answer):
    """Записує взаємодію користувача з системою в базу даних"""
    ts = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO history(ts,mode,query,answer) VALUES (?,?,?,?)",
            (ts, mode, query, answer),
        )
@st.cache_resource(show_spinner="Завантажуємо модель Whisper…")
def get_whisper(model_size: str = "base"):
    return whisper.load_model(model_size)


""" 
@st.cache_data(show_spinner="Транскрибуємо відео…")
def fetch_and_transcribe(uploaded_file: str, model_size: str):
    ydl_opts = {"quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(uploaded_file, download=False)
        title = info.get("title", "video")
        safe = clean_filename(title)
        unique_safe_title = f"{safe}_{hashlib.md5(uploaded_file.encode()).hexdigest()[:8]}"
        video_id = extract_video_id(uploaded_file)
        unique_video_id = f"{video_id}_{hashlib.md5(uploaded_file.encode()).hexdigest()[:8]}"
        file_path = os.path.join("video", f"{safe}_{unique_video_id}.mp4")
    txt = process_video(uploaded_file, "video", get_whisper(model_size), logging.getLogger(__name__))[1]
    return txt, file_path, safe, video_id,  title, unique_video_id, info, unique_safe_title, uploaded_file
"""

def build_audio_json(
    txt: str,
    meta: Dict[str, Any],
    file_path: str,
    original_filename: str,
    chunk_size: int = 1024,
    overlap: int = 256,
) -> Dict[str, Any]:
    """
    Формирует единый JSON для аудио:
    - разбивает prepared-text на чанки
    - собирает метаданные
    - возвращает структуру, готовую к upsert в Milvus
    """
    file_path = str(file_path)
    original_uuid = hashlib.sha256(txt.encode()).hexdigest()
    doc_id = f"aud_{meta['unique_audio_id']}"
    chunks = chunk_text(txt, size=chunk_size, overlap=overlap)

    return {
        "doc_id": doc_id,
        "original_uuid": original_uuid,
        "audio_meta": {
            "audio_id":         meta["audio_id"],
            "unique_audio_id":  meta["unique_audio_id"],
            "title":            meta["title"],
            "upload_date":      meta["upload_date"],
        },
        "content": txt,
        "chunks": [
            {
                "chunk_id":       f"{doc_id}_chunk_{i}",
                "original_index": i,
                "content":        c,
                "file_path":      file_path,
                "file_name":      original_filename,
                "upload_date":    meta["upload_date"],
            }
            for i, c in enumerate(chunks)
        ],
    }


@st.cache_data(show_spinner="Транскрибуємо аудіо…")
def process_audio(filepath: str, model_size: str, dir_name: str, title: str) -> str:
    """
    Транскрибує аудіо і повертає чистый текст.
    Ничего больше не делает — не разбивает на чанки и не формирует метаданные.
    """
    # Загрузили модель один раз (кешируется get_whisper)
    whisper_model = get_whisper(model_size)
    logger = logging.getLogger(__name__)
    logger.info(f"Транскрибація аудіо: {filepath}")

    result = whisper_model.transcribe(str(filepath))
    transcript = result["text"]

    # Сохраняем транскрипт рядом с файлом
    transcript_path = os.path.join(dir_name, f"{title}_transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    logger.info(f"Транскрипт збережено: {transcript_path}")

    return transcript


def audio_mode(collection_name: str, summary: bool):
    audio_dir = "audio"
    create_dir(audio_dir)

    uploaded_file = st.file_uploader(
        "Оберіть аудіофайл",
        type=["mp3", "wav", "flac", "aac", "m4a", "ogg"],
    )
    if uploaded_file is None:
        return

    # --- Сохранение файла ---
    full_name = uploaded_file.name
    title, ext = os.path.splitext(full_name)
    filepath = os.path.join(audio_dir, full_name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Метаданные
    audio_id         = Path(filepath).stem
    safe_title       = clean_filename(title)
    unique_audio_id = f"{audio_id}_{hashlib.md5(filepath.encode()).hexdigest()[:8]}"
    upload_date      = dt.now().isoformat()

    meta = {
        "title":             title,
        "audio_id":          audio_id,
        "unique_audio_id":   unique_audio_id,
        "upload_date":       upload_date,
    }

    # Выбор модели и кнопка запуска
    model_size = st.selectbox(
        "Модель Whisper:",
        ["tiny", "base", "small", "medium", "large"],
        index=1,
    )
    if not st.button("Обробити"):
        return

    # --- Транскрибация ---
    transcript = process_audio(filepath, model_size, audio_dir, title)
    st.markdown(transcript)

    # Подготовка текста к чанкингу
    cleaned = prepare_text(transcript)
    if summary and not st.session_state.audio_context_text.get("context"):
        llm = create_llm("gemma3:4b")
        summ = summarise_transcript(cleaned, llm)
        st.session_state.audio_context_text["context"] = summ

    # --- Построение единого JSON ---
    audio_json = build_audio_json(
        txt=cleaned,
        meta=meta,
        file_path=filepath,
        original_filename=full_name,
    )
    st.session_state["last_audio_json"] = audio_json
    st.session_state.audio_processed = True

    # --- Загрузка в Milvus ---
    dense_ef = create_bge_m3_embeddings()
    retriever = AudioHybridRetriever(
        client=st.session_state.milvus_client,
        collection_name=collection_name,
        dense_embedding_function=dense_ef,
    )
    if st.session_state.audio_processed:
        retriever.build_collection()
        for chunk in audio_json["chunks"]:
            metadata = {
                **audio_json["audio_meta"],
                "doc_id":        audio_json["doc_id"],
                "original_uuid": audio_json["original_uuid"],
                "chunk_id":      chunk["chunk_id"],
                "original_index": chunk["original_index"],
                "file_path":      chunk["file_path"],
                "file_name":      chunk["file_name"],
                "upload_date":    chunk["upload_date"],
            }
            retriever.upsert_data(chunk["content"], metadata)

    st.success("Аудіо оброблено та завантажено в колекцію.")
    st.session_state.audio_processed = True


def chat_audio_mode(collection_name, llm_option):
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "Привіт! Я готовий до розмови. Що вас цікавить?"})
         
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    # ввод користувача
    if q := st.chat_input("Ваш запит"):
        st.session_state.messages.append(dict(role="user", content=q))
        with st.chat_message("user"): st.markdown(q)
        with st.chat_message("assistant"):
            st.write("Шукаю відповідь…")
            dense_ef = create_bge_m3_embeddings()
            retr = AudioHybridRetriever(
            client=st.session_state.milvus_client,
            collection_name=collection_name,
            dense_embedding_function=dense_ef,) 
            retr.build_collection()          # ← додали
            chunks = retr.search(q, mode="hybrid", k=5)
            ctx = "\n---\n".join([c["content"] for c in chunks]) if chunks else ""
            user_query = q+"\n---\n"+ctx
            if llm_option == "Україномовну":
                chat = [
                    {"role": "system", "content": "You are useful assistant. Use the info to answer user query. Answer in Ukrainian"},
                    {"role": "user", "content": user_query}
                    ]
                response = st.session_state.ukr_generator(chat, max_new_tokens=512)
                answer = response[0]["generated_text"][-1]["content"]
            else:   
                llm = create_llm("gemma3:4b")
                chain = create_chain(llm, create_prompt(REGULAR_SYSTEM_PROMPT))
                response = get_llm_response(chain, q, ctx).content
                answer = response.content if hasattr(response, "content") else str(response)

            st.markdown(answer)
            with st.expander("Показати використаний контекст"):
                        st.markdown(
                            f"**Файл:** {chunks[0]['file_name']}  \n"
                            f"**Chunk ID:** `{chunks[0]['chunk_id']}`  \n"
                            f"**Дата завантаження:** {chunks[0]['upload_date']}  \n"
                            f"**Score:** {chunks[0]['score']:.4f}  \n\n"
                            f"> {chunks[0]['content']}",
                        )
            log_interaction(st.session_state.current_mode, q, answer)
            st.session_state.messages.append(dict(role="assistant", content=answer))




def video_mode(collection_name, summary):
    url = st.text_input("Введіть URL відео")
    model_size = st.selectbox("Модель Whisper:", ["tiny", "base", "small", "medium", "large"], index=1)
    if url and st.button("Обробити"):
        video_chunks, txt, file_path, safe, video_id, title, unique_video_id, info, unique_safe_title, uploaded_file = process_video(url, "video", model_size)
        #txt, file_path, safe, video_id, title, unique_video_id, info, unique_safe_title, uploaded_file = fetch_and_transcribe(url, model_size)
        video_meta = {
            "video_id": video_id,
            "unique_video_id": unique_video_id,
            "title": title,
            "uploader": info.get("uploader", ""),
            "duration": info.get("duration", 0),
            "upload_date": info.get("upload_date", ""),
            "view_count": info.get("view_count", 0),
            "like_count": info.get("like_count", 0),
            "url": uploaded_file,
            "file_path": file_path,
            "file_name": unique_safe_title,
            "upload_date": info.get("upload_date", ""),
        }
        st.markdown(txt)
        txt = prepare_text(txt)
        if summary and not st.session_state.video_context_text['context']:
            llm = create_llm("gemma3:4b")
            summary  = summarise_transcript(txt, llm)
            print(summary, 'summary')
            st.session_state.video_context_text['context'] = summary
        video_json = build_video_json(txt, video_meta, file_path, safe)
        st.session_state['last_video_json'] = video_json
        st.session_state.video_processed = True

    

    # Після рендеру блоку відео подальший код не потрібен
    dense_ef = create_bge_m3_embeddings()
    standard_retriever = VideoHybridRetriever(
    client=st.session_state.milvus_client,
    collection_name=collection_name,
    dense_embedding_function=dense_ef,)
    if st.session_state.get('video_processed'):
        video_json = st.session_state['last_video_json']
        standard_retriever.build_collection()
        for ch in video_json["chunks"]:
            metadata = {
                **video_json["video_meta"],
                "doc_id":        video_json["doc_id"],
                "original_uuid": video_json["original_uuid"],
                "chunk_id":      ch["chunk_id"],
                "original_index": ch["original_index"],
                "file_path":      ch["file_path"],
                "file_name":      ch["file_name"],
                "upload_date":    ch["upload_date"],
                }
            standard_retriever.upsert_data(ch["content"], metadata)
        st.session_state.video_processed = True

def chat_video_mode(collection_name, llm_option):
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "Привіт! Я готовий до розмови. Що вас цікавить?"})
         
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    # ввод користувача
    if q := st.chat_input("Ваш запит"):
        st.session_state.messages.append(dict(role="user", content=q))
        with st.chat_message("user"): st.markdown(q)
        with st.chat_message("assistant"):
            st.write("Шукаю відповідь…")
            dense_ef = create_bge_m3_embeddings()
            retr = VideoHybridRetriever(
            client=st.session_state.milvus_client,
            collection_name=collection_name,
            dense_embedding_function=dense_ef,) 
            retr.build_collection()          # ← додали
            chunks = retr.search(q, mode="hybrid", k=5)
            ctx = "\n---\n".join([c["content"] for c in chunks]) if chunks else ""
            user_query = q+"\n---\n"+ctx
            if llm_option == "Україномовну":
                chat = [
                    {"role": "system", "content": "You are useful assistant. Use the info to answer user query. Answer in Ukrainian"},
                    {"role": "user", "content": user_query}
                    ]
                response = st.session_state.ukr_generator(chat, max_new_tokens=512)
                answer = response[0]["generated_text"][-1]["content"]
            else:   
                llm = create_llm("gemma3:4b")
                chain = create_chain(llm, create_prompt(REGULAR_SYSTEM_PROMPT))
                response = get_llm_response(chain, q, ctx).content
                answer = response.content if hasattr(response, "content") else str(response)

            st.markdown(answer)
            with st.expander("Показати використаний контекст"):
                        st.markdown(
                            f"**Файл:** {chunks[0]['file_name']}  \n"
                            f"**Chunk ID:** `{chunks[0]['chunk_id']}`  \n"
                            f"**Дата завантаження:** {chunks[0]['upload_date']}  \n"
                            f"**Score:** {chunks[0]['score']:.4f}  \n\n"
                            f"> {chunks[0]['content']}",
                        )
            log_interaction(st.session_state.current_mode, q, answer)
            st.session_state.messages.append(dict(role="assistant", content=answer))


def image_mode(collection_name, summary = True):
    st.session_state.current_mode = "image"
    st.subheader("Завантаження зображень")
    uploaded_images = st.file_uploader(
    "Обери одне або кілька зображень",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
    key="image_uploader"
)
    if uploaded_images:
        image_dir = "images"
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
            st.info(f"Створено директорію {image_dir} для збереження зображень")
        
        for img_file in uploaded_images:
            # превью в интерфейсе
            st.image(img_file)
            pil_img = Image.open(img_file)
            img_b64 = image_to_base64(pil_img)
            prompt = 'Опиши детально, что ты видишь на этом изображении, как профессиональный OSINT-аналитик. Опиши всё на русском языке. Не используй списки в описании. Строго отвечай только на русском языке.'
            with st.spinner("Генерую опис…"):
                caption = query_ollama(prompt, img_b64, "gemma3:4b")
                llm      = create_llm("gemma3:4b")
                if summary and not st.session_state.image_context_text['context']:
                    summary  = summarise_transcript(caption, llm)
                    print(summary, 'summary')
                    st.session_state.image_context_text['context'] = summary
            st.write(caption)
            # ---------- сохранение ----------
            file_ext = os.path.splitext(img_file.name)[1]
            print(file_ext, 'file_ext')
            unique_name = f"{uuid.uuid4().hex}{file_ext}"

            print(unique_name, 'unique_name')
            
            # Зберігаємо в директорії data (як було раніше)
            data_path = os.path.join("data", unique_name)
            print(data_path, 'data_path')
            with open(data_path, "wb") as f:
                f.write(img_file.getbuffer())
            st.success(f"Файл {img_file.name} збережено як {data_path}")
            st.session_state.image_processed = True
                    
            # Зберігаємо копію в директорії images
            image_path = os.path.join(image_dir, unique_name)
            with open(image_path, "wb") as f:
                f.write(img_file.getbuffer())
            st.success(f"Копію файлу збережено в {image_path}")

            # ---------- запись в Milvus ----------
            raw = build_image_json(caption, data_path, img_file.name)
            if isinstance(raw, dict):
                docs = [raw]
            elif isinstance(raw, list):
                docs = raw
            else:
                raise ValueError("Неподдерживаемый формат из build_json")
            
            print(docs, 'docs')
            
            is_insert = True
            if is_insert:
                dense_ef = create_bge_m3_embeddings()
                standard_retriever = ImageHybridRetriever(
                    client=st.session_state.milvus_client,
                    collection_name=collection_name,
                    dense_embedding_function=dense_ef,)
                standard_retriever.build_collection()          # ← додали
                for doc in docs:
                #Added funct
                    existing = standard_retriever.client.query(
                    collection_name=collection_name,
                    filter=f'original_uuid == "{doc["original_uuid"]}"',
                    output_fields=["chunk_id"])
                    existing_ids = {row["chunk_id"] for row in existing}
                #End of adding

                    for chunk in doc["chunks"]:
                        #Added funct
                        if chunk["chunk_id"] in existing_ids:        # дубль – пропускаем
                            continue
                        #End of adding
                        metadata = {
                            "doc_id":          doc["doc_id"],
                            "original_uuid":  doc["original_uuid"],
                            "chunk_id":       chunk["chunk_id"],
                            "original_index": chunk["original_index"],
                            "content":        chunk["content"],
                            "file_path":      chunk["file_path"],
                            "file_name":      chunk["file_name"],
                            "upload_date":    chunk["upload_date"],
                        }
                        standard_retriever.upsert_data(chunk["content"], metadata)
            st.session_state.image_processed=True

def chat_image_mode(collection_name, llm_option):
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "Привіт! Я готовий до розмови. Що вас цікавить?"})
         
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Ввод пользователя
    if query := st.chat_input("Введіть ваше запитання...", key="document_chat_input"):
        # Добавляем сообщение пользователя в историю
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role":"user","content":query})



        with st.chat_message("user"):
            st.markdown(query)

        # Стартуем поиск по векторам
        with st.chat_message("assistant"):
            st.write("Шукаю відповідь...")
            # Создаём извлекатель векторов
            dense_ef = create_bge_m3_embeddings()
            retriever = ImageHybridRetriever(
                client=st.session_state.milvus_client,
                collection_name=collection_name,
                dense_embedding_function=dense_ef,
            )
            retriever.build_collection()          # ← додали
            # Ищем 5 наиболее похожих чанков
            results = retriever.search(query, mode="hybrid", k=5)
            if not results:
                answer = "На жаль, не знайшов релевантної інформації."
            else:
                #best = results[0]["content"] For the beast but only one result
                context = "\n---\n".join(r["content"] for r in results)
                print(context, 'context')
                prev_a = "\n".join(m["content"] for m in st.session_state.chat_history[-4:] if m["role"]=="assistant") 
                print(prev_a, 'prev_a')
                full_ctx = prev_a + "\n" + context if prev_a else context  # Формування повного контексту з історії та знайдених результатів
                print(full_ctx, 'full_ctx')

                # Формируем промпт и вызываем LLM
                prompt = create_prompt(REGULAR_SYSTEM_PROMPT)
                user_query = query+"\n---\n"+full_ctx
                if llm_option == "Україномовну":
                    chat = [
                        {"role": "system", "content": "You are useful assistant. Use the info to answer user query. Answer in Ukrainian"},
                        {"role": "user", "content": user_query}
                        ]
                    response = st.session_state.ukr_generator(chat, max_new_tokens=512)
                    answer = response[0]["generated_text"][-1]["content"]
                else:   
                    llm = create_llm("gemma3:4b")
                    chain = create_chain(llm, prompt)
                    response = get_llm_response(chain, query, full_ctx)
                    answer = response.content if hasattr(response, "content") else str(response)


                # Выводим ответ и сохраняем в историю
                st.markdown(answer)
                with st.expander("Показати використаний контекст"):
                            st.markdown(
                                f"**Файл:** {results[0]['file_name']}  \n"
                                f"**Chunk ID:** `{results[0]['chunk_id']}`  \n"
                                f"**Дата завантаження:** {results[0]['upload_date']}  \n"
                                f"**Score:** {results[0]['score']:.4f}  \n\n"
                                f"> {results[0]['content']}"
                            )
                st.session_state.messages.append({"role": "assistant", "content": answer})
                log_interaction(st.session_state.current_mode or "chat", query, answer)
        st.stop()

def document_mode(collection_name, summary = False):
    uploaded_file = st.file_uploader("Завантажте документ", type=['pdf', 'xlsx', 'xls', 'doc', 'docx', 'md'])
    if uploaded_file is not None:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext in [".xlsx", ".xls", ".csv"]:
            if ext == ".csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.dataframe(df.head(10), use_container_width=True)
            preview_text = " ".join(df.astype(str).head(10).values.flatten())
        elif ext == ".pdf":
            from PyPDF2 import PdfReader
            reader = PdfReader(uploaded_file)
            first_page_text = reader.pages[0].extract_text() or ""
            preview_text = first_page_text[:1000]
        elif ext in [".docx", ".doc"]:
            docx_obj = DocxDocument(uploaded_file)
            text = "\n".join(p.text for p in docx_obj.paragraphs)
            preview_text = text[:1000]
        else:  # docx, md, txt, etc.
            raw_bytes = uploaded_file.read()
            text = raw_bytes.decode("utf-8", errors="ignore")
            preview_text = text[:1000]
        st.markdown(preview_text)
        uploaded_file.seek(0)
        # Отримуємо розширення файлу з оригінальної назви
        file_extension = os.path.splitext(uploaded_file.name)[1]
        # Генеруємо унікальне ім'я файлу використовуючи UUID
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        # Формуємо повний шлях до файлу в папці data
        file_path = os.path.join("data", unique_filename)
        # Зберігаємо оригінальну назву файлу
        original_filename = uploaded_file.name
        
        # Відкриваємо файл для запису в бінарному режимі
        with open(file_path, "wb") as f:
            # Записуємо вміст завантаженого файлу
            f.write(uploaded_file.getbuffer())
        # Показуємо повідомлення про успішне збереження
        st.success(f"Файл {original_filename} успішно збережено як {file_path}")
        raw = data_extraction(file_path)
        raw_str = raw.to_csv(index=False) if isinstance(raw, pd.DataFrame) else (
        json.dumps(raw, ensure_ascii=False) if isinstance(raw, dict) else str(raw)
    )
        txt = prepare_text(raw_str)
        if summary and not st.session_state.document_context_text['context']:
            summary = summarise_transcript(txt, create_llm("gemma3:4b"))
            st.session_state.document_context_text['context'] = summary
        raw = build_json(txt, original_filename, file_path)

        if isinstance(raw, dict):
            docs = [raw]
        elif isinstance(raw, list):
            docs = raw
        else:
            raise ValueError("Неподдерживаемый формат из build_json")

        dense_ef = create_bge_m3_embeddings()
        standard_retriever = HybridRetriever(
        client=st.session_state.milvus_client,
        collection_name=collection_name,
        dense_embedding_function=dense_ef,
        )

        is_insert = True
        if is_insert:
            standard_retriever.build_collection()
            for doc in docs:
                existing = standard_retriever.client.query(
                collection_name=collection_name,
                filter=f'original_uuid == "{doc["original_uuid"]}"',
                output_fields=["chunk_id"])
                existing_ids = {row["chunk_id"] for row in existing}
                for chunk in doc["chunks"]:
                    if chunk["chunk_id"] in existing_ids:        # дубль – пропускаем
                        continue
                    metadata = {
                        "doc_id":          doc["doc_id"],
                        "original_uuid":  doc["original_uuid"],
                        "chunk_id":       chunk["chunk_id"],
                        "original_index": chunk["original_index"],
                        "content":        chunk["content"],
                        "file_path":      chunk["file_path"],
                        "file_name":      chunk["file_name"],
                        "upload_date":    chunk["upload_date"],
                    }
                    standard_retriever.upsert_data(chunk["content"], metadata)
        st.session_state.document_processed = True
        st.session_state.document_start = False

        
def chat_document_mode(collection_name, llm_option):
    st.session_state.doc_mode = "chat"
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "Привіт! Я готовий до розмови. Що вас цікавить?"})
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Ввод пользователя
    if query := st.chat_input("Введіть ваше запитання...", key="document_chat_input"):
    # Добавляем сообщение пользователя в историю
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Стартуем поиск по векторам
        with st.chat_message("assistant"):
            st.write("Шукаю відповідь...")
            # Создаём извлекатель векторов
            dense_ef = create_bge_m3_embeddings()
            retriever = HybridRetriever(
                client=st.session_state.milvus_client,
                collection_name=collection_name,
                dense_embedding_function=dense_ef,
            )
            retriever.build_collection()          # ← додали
            # Ищем 5 наиболее похожих чанков
            results = retriever.search(query, mode="hybrid", k=5)
            if not results:
                answer = "На жаль, не знайшов релевантної інформації."
            else:
                best = results[0]["content"]
                print(best, 'best content')

                # Формируем промпт и вызываем LLM
                prompt = create_prompt(REGULAR_SYSTEM_PROMPT)
                user_query = query+"\n---\n"+best
                if llm_option == "Україномовну":
                    chat = [
                        {"role": "system", "content": "You are useful assistant. Use the info to answer user query. Answer in Ukrainian"},
                        {"role": "user", "content": user_query}
                        ]
                    response = st.session_state.ukr_generator(chat, max_new_tokens=512)
                    answer = response[0]["generated_text"][-1]["content"]
                else:   
                    llm = create_llm("gemma3:4b")
                    chain = create_chain(llm, prompt)
                    response = get_llm_response(chain, query, best)
                    answer = response.content if hasattr(response, "content") else str(response)

            # Выводим ответ и сохраняем в историю
            st.markdown(answer)
            with st.expander("Показати використаний контекст"):
                        st.markdown(
                            f"**Файл:** {results[0]['file_name']}  \n"
                            f"**Chunk ID:** `{results[0]['chunk_id']}`  \n"
                            f"**Дата завантаження:** {results[0]['upload_date']}  \n"
                            f"**Score:** {results[0]['score']:.4f}  \n\n"
                            f"> {results[0]['content']}"
                        )
            st.session_state.messages.append({"role": "assistant", "content": answer})
            log_interaction(st.session_state.current_mode or "chat", query, answer)


def create_summary_prompt() -> ChatPromptTemplate:
    """Возвращает промпт для LLM-суммаризации."""
    return ChatPromptTemplate.from_messages([
        ("system", SUMMARY_SYSTEM_PROMPT),
        ("human", "{text}")               # <- единственная переменная
    ])


def summarise_transcript(transcript: str, llm) -> str:
    prompt = create_summary_prompt()
    chain  = create_chain(llm, prompt)    # create_chain = prompt | llm
    response = chain.invoke({"text": transcript})
    return response.content