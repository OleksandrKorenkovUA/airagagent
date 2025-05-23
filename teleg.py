# teleg.py
import os
import logging
import telegram
from telegram import Update, Bot, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import random
import asyncio
from rag import HybridRetriever
from threading import Thread
from functions import create_bge_m3_embeddings, create_stream_prompt, rerank_search, query_ollama
from pymilvus import MilvusClient # Потрібно для type hinting, якщо використовується
from config import RELIGIOUS_SYSTEM_PROMPT
# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' # Більш детальний формат
)
logger = logging.getLogger(__name__)

# Глобальні змінні для teleg.py
telegram_app: Application = None
_telegram_bot_thread: Thread = None # Додаємо type hint для ясності

# Глобальні налаштування за замовчуванням для бота
BOT_DEFAULT_COLLECTION = "не_встановлено"
BOT_DEFAULT_LLM = "не_встановлено"
BOT_DEFAULT_K_RESULTS = 10
BOT_DEFAULT_SYSTEM_PROMPT = "..." # Ваше значення за замовчуванням
BOT_DEFAULT_MILVUS_CLIENT: MilvusClient = None # Для зберігання переданого клієнта Milvus

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    logger.error("Не знайдено токен Telegram бота. Встановіть змінну середовища TELEGRAM_TOKEN")
    raise ValueError("Не знайдено токен Telegram бота. Встановіть змінну середовища TELEGRAM_TOKEN")

user_settings = {}

# ... (async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):)
# У функції start, переконайтеся, що використовуєте BOT_DEFAULT_MILVUS_CLIENT
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ...
    chat_id = update.effective_chat.id
    user_settings[chat_id] = {
        "collection": BOT_DEFAULT_COLLECTION,
        "llm": BOT_DEFAULT_LLM,
        # "mode": BOT_DEFAULT_MODE, # Ви закоментували це, можливо, так і треба
        "k_results": BOT_DEFAULT_K_RESULTS,
        "system_prompt": BOT_DEFAULT_SYSTEM_PROMPT,
        "client": BOT_DEFAULT_MILVUS_CLIENT # Використовуємо клієнт, переданий в main
    }
    # ...

# ... (async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):)
# У handle_message, client береться з user_settings, що добре.

def main(default_collection: str, default_llm: str, default_k_results: int, default_system_prompt: str, default_client: MilvusClient): # Змінено тип default_client
    global telegram_app, BOT_DEFAULT_COLLECTION, BOT_DEFAULT_LLM, BOT_DEFAULT_K_RESULTS, BOT_DEFAULT_SYSTEM_PROMPT, BOT_DEFAULT_MILVUS_CLIENT

    logger.info(f"Функція teleg.main викликана з: collection='{default_collection}', llm='{default_llm}', k='{default_k_results}', client_type='{type(default_client)}'")

    BOT_DEFAULT_COLLECTION = default_collection
    BOT_DEFAULT_LLM = default_llm
    BOT_DEFAULT_K_RESULTS = default_k_results
    BOT_DEFAULT_SYSTEM_PROMPT = default_system_prompt
    BOT_DEFAULT_MILVUS_CLIENT = default_client # Зберігаємо переданий об'єкт клієнта Milvus

    if telegram_app and hasattr(telegram_app, 'running') and telegram_app.running: # Перевірка для python-telegram-bot v20+
        logger.warning("Telegram bot is already running. Skipping initialization.")
        return

    logger.info("Ініціалізація Telegram Application...")
    try:
        telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()
        # ... (додавання обробників)
        telegram_app.add_handler(CommandHandler("start", start))
        telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        logger.info("Запуск Telegram бота в режимі polling...")
        telegram_app.run_polling(allowed_updates=Update.ALL_TYPES)
    except telegram.error.Conflict as e_conflict: # Обробка помилки конфлікту тут теж
            logger.error(f"teleg.main: Telegram Conflict Error: {e_conflict}. Ймовірно, інший екземпляр вже запущено.")
    except Exception as e:
        logger.error(f"Exception during bot polling setup or execution: {e}", exc_info=True)
    finally:
        logger.info("Telegram бот зупинено (run_polling завершено або виникла помилка при запуску).")
        # telegram_app = None # Не обнуляйте тут, якщо stop_telegram_bot має це робити
                           # Або, якщо обнуляєте, то stop_telegram_bot має це враховувати

def stop_telegram_bot():
    global telegram_app, _telegram_bot_thread
    logger.info("Attempting to stop the Telegram bot (teleg.stop_telegram_bot)...")

    current_loop = None
    if telegram_app and hasattr(telegram_app, '_event_loop'): # Для v20, _event_loop може бути недоступним після зупинки
        current_loop = telegram_app._event_loop

    if telegram_app: # Якщо є екземпляр програми
        logger.info(f"Telegram application instance found. Attempting to stop. Running: {hasattr(telegram_app, 'running') and telegram_app.running}")
        if hasattr(telegram_app, 'stop') and callable(telegram_app.stop):
            # Для python-telegram-bot v20+ stop() є асинхронним
            if current_loop and current_loop.is_running():
                logger.info(f"Event loop {id(current_loop)} is running. Scheduling telegram_app.stop().")
                future = asyncio.run_coroutine_threadsafe(telegram_app.stop(), current_loop)
                try:
                    future.result(timeout=10) # Збільшено таймаут
                    logger.info("Telegram bot stop() coroutine executed.")
                except TimeoutError:
                    logger.error("Timeout waiting for Telegram bot stop() to complete.")
                except Exception as e:
                    logger.error(f"Error executing Telegram bot stop(): {e}", exc_info=True)
            elif hasattr(telegram_app, 'running') and telegram_app.running:
                 logger.warning("Event loop not accessible or not running, but app claims to be running. Trying to call stop directly (might not work as intended from a different thread without a loop).")
                 # Це може не спрацювати належним чином без циклу подій
                 # asyncio.run(telegram_app.stop()) # НЕБЕЗПЕЧНО викликати так з іншого потоку
            else:
                logger.info("Telegram application is not running or event loop is not available/running.")
        else:
            logger.warning("telegram_app.stop() is not available or not callable.")
        telegram_app = None # Обнуляємо після спроби зупинки
    else:
        logger.info("Telegram application is already None (at the start of stop_telegram_bot).")

    if _telegram_bot_thread and _telegram_bot_thread.is_alive():
        logger.info(f"Joining Telegram bot thread (ID: {_telegram_bot_thread.native_id})...")
        _telegram_bot_thread.join(timeout=15) # Збільшено таймаут
        if _telegram_bot_thread.is_alive():
            logger.error("Telegram bot thread did not terminate after join.")
        else:
            logger.info("Telegram bot thread terminated successfully.")
    else:
        logger.info("Telegram bot thread is not active or already None (at the start of stop_telegram_bot or after join).")
    
    _telegram_bot_thread = None # Обнуляємо посилання на потік
    logger.info("stop_telegram_bot function finished.")

# Ваша функція handle_message з asyncio.to_thread (яку ви вже, ймовірно, маєте)
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    logger.info(f"ChatID {chat_id}: Received message: '{text}'")

    if chat_id not in user_settings:
        logger.info(f"ChatID {chat_id}: User not in settings, calling start.")
        await start(update, context) # user_settings буде заповнено тут
        # Важливо: після start, user_settings[chat_id]["client"] буде BOT_DEFAULT_MILVUS_CLIENT
    
    settings = user_settings[chat_id]
    collection_name = settings["collection"]
    llm_name = settings.get("llm")
    k_results = settings.get("k_results")
    system_prompt = settings.get("system_prompt")
    # Клієнт Milvus береться з налаштувань користувача, які були встановлені в start()
    milvus_client_obj = settings.get("client") 

    if not milvus_client_obj:
        logger.error(f"ChatID {chat_id}: Milvus client is not available in user settings.")
        await update.message.reply_text("Помилка: Клієнт бази даних не налаштований.")
        return

    logger.info(f"ChatID {chat_id}: Processing with collection='{collection_name}', llm='{llm_name}', k='{k_results}', client_type='{type(milvus_client_obj)}'")

    # Функції-обгортки для блокуючих операцій
    def _blocking_search_and_rerank():
        # Ініціалізація dense_ef тут, якщо вона не важка, або передавати її
        dense_ef = create_bge_m3_embeddings() 
        retriever = HybridRetriever(client=milvus_client_obj, collection_name=collection_name, dense_embedding_function=dense_ef)
        # retriever.build_collection() # Зазвичай build_collection викликається один раз при створенні/завантаженні колекції, а не при кожному пошуку
                                      # Якщо колекція вже існує, цей виклик може бути непотрібним або навіть шкідливим при кожному запиті.
                                      # Переконайтеся, що він потрібен тут. Якщо ні - приберіть.

        raw_results = retriever.search(text, mode="hybrid", k=k_results)
        logger.info(f"ChatID {chat_id}: Raw Milvus results count: {len(raw_results)}")
        docs_for_ranker = [d["content"] for d in raw_results]
        reranked_results = rerank_search(docs_for_ranker, text, k_results) # Переконайтесь, що k_results тут доречне
        logger.info(f"ChatID {chat_id}: Reranked results count: {len(reranked_results)}")
        return reranked_results

    def _blocking_query_ollama(current_prompt_text):
        return query_ollama(system_prompt, current_prompt_text, llm_name, stream=False)

    try:
        reranked_search_results = await asyncio.to_thread(_blocking_search_and_rerank)
        
        if not reranked_search_results:
            logger.warning(f"ChatID {chat_id}: No results after search and rerank for query: '{text}'")
            await update.message.reply_text("На жаль, не знайдено релевантної інформації за вашим запитом.")
            return

        ctx = "\n---\n".join([c.text for c in reranked_search_results])
        current_prompt = create_stream_prompt(text, ctx) # Переконайтесь, що ця функція не блокуюча
        
        logger.info(f"ChatID {chat_id}: Sending prompt to Ollama (LLM: {llm_name}).")
        ollama_result = await asyncio.to_thread(_blocking_query_ollama, current_prompt)
        
        answer = ollama_result["message"]["content"]
        logger.info(f"ChatID {chat_id}: Received answer from Ollama.")
        await update.message.reply_text(answer)

    except Exception as e:
        logger.error(f"ChatID {chat_id}: Error in handle_message: {e}", exc_info=True)
        await update.message.reply_text("Вибачте, під час обробки вашого запиту сталася внутрішня помилка.")


if __name__ == '__main__':
    logger.info("Запуск teleg.py напряму для тестування...")
    # Для тестування потрібен реальний MilvusClient
    # test_milvus_client = MilvusClient(uri="http://127.0.0.1:19530")
    main(default_collection="religion", default_llm="gemma3:12b", default_k_results=10, default_system_prompt=RELIGIOUS_SYSTEM_PROMPT, default_client= MilvusClient(uri="http://127.0.0.1:19530"))
    logger.warning("Для прямого запуску teleg.py розкоментуйте та налаштуйте тестовий клієнт Milvus.")