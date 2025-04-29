import json  # Імпорт модуля для роботи з JSON-даними


''' Приклад взято звідси - https://milvus.io/docs/full_text_search_with_milvus.md '''

# Імпорт необхідних компонентів з бібліотеки pymilvus для роботи з векторною базою даних
from pymilvus import (
    MilvusClient,  # Клієнт для взаємодії з Milvus сервером
    DataType,      # Типи даних для полів у колекціях
    Function,      # Клас для визначення функцій обробки даних
    FunctionType,  # Типи функцій (BM25, тощо)
    AnnSearchRequest,  # Клас для формування запитів пошуку
    RRFRanker,     # Ранжувальник для гібридного пошуку
)

# Імпорт функції вбудовування для створення щільних векторів з тексту
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


class HybridRetriever:
    """Клас для гібридного пошуку документів з використанням щільних та розріджених векторів"""
    def __init__(self, client, collection_name="hybrid", dense_embedding_function=None):
        """Ініціалізація об'єкта HybridRetriever
        
        Параметри:
        client - клієнт Milvus для взаємодії з базою даних
        collection_name - назва колекції для зберігання документів
        dense_embedding_function - функція для створення щільних векторів з тексту
        """
        self.collection_name = collection_name  # Зберігаємо назву колекції
        self.embedding_function = dense_embedding_function  # Функція для створення векторних ембедінгів
        self.use_reranker = True  # Прапорець для використання ранжувальника
        self.use_sparse = True  # Прапорець для використання розріджених векторів
        self.client = client  # Зберігаємо посилання на клієнт Milvus

    def build_collection(self):
        """Створення або завантаження колекції Milvus для зберігання документів"""
        if hasattr(self.client, "has_collection"):
            exists = self.client.has_collection(self.collection_name)  # Перевірка наявності колекції новим методом
        else:  # старі версії pymilvus
            exists = self.collection_name in self.client.list_collections()  # Перевірка наявності колекції старим методом

        if exists:
            # ───────────── варіант «перевикористати» ─────────────
            self.client.load_collection(self.collection_name)  # Завантажуємо існуючу колекцію в пам'ять
            return 
        
        # Визначення розмірності щільного вектора залежно від типу функції ембедінгу
        if isinstance(self.embedding_function.dim, dict):
            dense_dim = self.embedding_function.dim["dense"]  # Отримуємо розмірність з словника
        else:
            dense_dim = self.embedding_function.dim  # Отримуємо розмірність напряму

        # Налаштування токенізатора для обробки тексту при створенні розріджених векторів
        tokenizer_params = {
            "tokenizer": "standard",  # Стандартний токенізатор для розбиття тексту
            "filter": [
                "lowercase",  # Перетворення всіх символів на нижній регістр
                {
                    "type": "length",  # Фільтр за довжиною токенів
                    "max": 200,  # Максимальна довжина токена
                },
                {"type": "stemmer", "language": "russian"},  # Стемінг для російської мови
                {
                    "type": "stop",  # Видалення стоп-слів
                    "stop_words": ["_russian_"  # Використання стандартного набору російських стоп-слів
                    ],
                },
            ],
        }

        # Створення схеми колекції для зберігання документів
        schema = MilvusClient.create_schema(enable_dynamic_field=True)  # Дозволяємо динамічні поля

        # Додавання поля для тексту з налаштуваннями аналізатора
        schema.add_field(
            field_name="content",  # Назва поля для зберігання тексту
            datatype=DataType.VARCHAR,  # Тип даних - текстовий рядок
            max_length=65535,  # Максимальна довжина тексту
            analyzer_params=tokenizer_params,  # Параметри аналізатора тексту
            enable_match=True,  # Дозволяємо точне співпадіння
            enable_analyzer=True,  # Включаємо аналізатор тексту
        )
        # Додавання поля для розрідженого вектора (BM25)
        schema.add_field(
            field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR  # Поле для розрідженого вектора
        )
        # Додавання поля для щільного вектора
        schema.add_field(
            field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dense_dim  # Поле для щільного вектора з вказаною розмірністю
        )
        # Додавання службових полів для ідентифікації та метаданих
        schema.add_field(
            field_name="original_uuid", datatype=DataType.VARCHAR, max_length=128  # Унікальний ідентифікатор оригінального документа
        )
        schema.add_field(
            field_name="doc_id", datatype=DataType.VARCHAR, max_length=64  # Ідентифікатор документа
        )
        schema.add_field(
             field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64,  # Первинний ключ - ідентифікатор фрагмента
        ),
        schema.add_field(field_name="original_index", datatype=DataType.INT32, nullable=True)  # Індекс фрагмента в оригінальному документі
        schema.add_field(field_name="file_path", datatype=DataType.VARCHAR, max_length=1024)  # Шлях до файлу
        schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=512)  # Назва файлу
        schema.add_field(field_name="upload_date", datatype=DataType.VARCHAR, max_length=1024)  # Дата завантаження

        # Налаштування функції BM25 для створення розріджених векторів з тексту
        functions = Function(
            name="bm25",  # Назва функції
            function_type=FunctionType.BM25,  # Тип функції - BM25
            input_field_names=["content"],  # Вхідне поле - текстовий контент
            output_field_names="sparse_vector",  # Вихідне поле - розріджений вектор
        )

        schema.add_function(functions)  # Додаємо функцію до схеми

        # Налаштування індексів для векторних полів для пришвидшення пошуку
        index_params = MilvusClient.prepare_index_params()  # Створюємо параметри індексів
        index_params.add_index(
            field_name="sparse_vector",  # Поле для індексації
            index_type="SPARSE_INVERTED_INDEX",  # Тип індексу для розріджених векторів
            metric_type="BM25",  # Метрика для порівняння - BM25
        )
        index_params.add_index(
            field_name="dense_vector", index_type="FLAT", metric_type="IP"  # Індекс для щільних векторів з метрикою внутрішнього добутку
        )

        # Створення колекції з визначеною схемою та індексами
        self.client.create_collection(
            collection_name=self.collection_name,  # Назва колекції
            schema=schema,  # Схема колекції
            index_params=index_params,  # Параметри індексів
        )

    def insert_data(self, chunk, metadata):
        """Вставка даних у колекцію
        
        Параметри:
        chunk - текстовий фрагмент для вбудовування у вектор
        metadata - метадані документа (ідентифікатори, шляхи до файлів тощо)
        """
        embedding = self.embedding_function([chunk])  # Створюємо векторне представлення тексту
        if isinstance(embedding, dict) and "dense" in embedding:
            dense_vec = embedding["dense"][0]  # Отримуємо щільний вектор з словника
        else:
            dense_vec = embedding[0]  # Отримуємо щільний вектор напряму
        self.client.insert(
            self.collection_name, {"dense_vector": dense_vec, **metadata}  # Вставляємо вектор та метадані в колекцію
        )

    def upsert_data(self, chunk, metadata):
        """Оновлення або вставка даних у колекцію
        
        Параметри:
        chunk - текстовий фрагмент для вбудовування
        metadata - метадані документа
        """
        embedding = self.embedding_function([chunk])  # Створюємо векторне представлення тексту
        dense_vec = embedding["dense"][0] if isinstance(embedding, dict) else embedding[0]  # Отримуємо щільний вектор

        # Примусове перегенерування розрідженого вектора шляхом передачі тексту знову
        self.client.upsert(
            self.collection_name,
            {
                "content": chunk,          # <-- критично важливо для генерації розрідженого вектора
                "dense_vector": dense_vec,  # Щільний вектор
                **metadata  # Метадані документа
            }
        )


    def search(self, query: str, k: int = 20, mode="hybrid"):
        """Пошук документів за запитом
        
        Параметри:
        query - текстовий запит користувача
        k - кількість результатів для повернення
        mode - режим пошуку: "sparse" (розріджений), "dense" (щільний) або "hybrid" (гібридний)
        
        Повертає:
        Список знайдених документів з метаданими та оцінками релевантності
        """

        # Поля, які будуть повернуті в результатах пошуку
        output_fields = [
            "content",  # Текстовий вміст документа
            "original_uuid",  # Унікальний ідентифікатор оригінального документа
            "doc_id",  # Ідентифікатор документа
            "chunk_id",  # Ідентифікатор фрагмента
            "original_index",  # Індекс фрагмента в оригінальному документі
            "file_path",  # Шлях до файлу
            "file_name",  # Назва файлу
            "upload_date",  # Дата завантаження
        ]
        # Створення вбудовування для запиту, якщо потрібно для щільного або гібридного пошуку
        if mode in ["dense", "hybrid"]:
            embedding = self.embedding_function([query])  # Створюємо векторне представлення запиту
            if isinstance(embedding, dict) and "dense" in embedding:
                dense_vec = embedding["dense"][0]  # Отримуємо щільний вектор з словника
            else:
                dense_vec = embedding[0]  # Отримуємо щільний вектор напряму

        # Пошук за розрідженим вектором (BM25)
        if mode == "sparse":
            results = self.client.search(
                collection_name=self.collection_name,  # Назва колекції
                data=[query],  # Текстовий запит
                anns_field="sparse_vector",  # Поле для пошуку - розріджений вектор
                limit=k,  # Кількість результатів
                output_fields=output_fields,  # Поля для повернення
            )
        # Пошук за щільним вектором
        elif mode == "dense":
            results = self.client.search(
                collection_name=self.collection_name,  # Назва колекції
                data=[dense_vec],  # Векторне представлення запиту
                anns_field="dense_vector",  # Поле для пошуку - щільний вектор
                limit=k,  # Кількість результатів
                output_fields=output_fields,  # Поля для повернення
            )
        # Гібридний пошук (комбінація розрідженого та щільного)
        elif mode == "hybrid":
            full_text_search_params = {"metric_type": "BM25"}  # Параметри для пошуку за розрідженим вектором
            full_text_search_req = AnnSearchRequest(
                [query], "sparse_vector", full_text_search_params, limit=k  # Запит для розрідженого пошуку
            )

            dense_search_params = {"metric_type": "IP"}  # Параметри для пошуку за щільним вектором
            dense_req = AnnSearchRequest(
                [dense_vec], "dense_vector", dense_search_params, limit=k  # Запит для щільного пошуку
            )

            results = self.client.hybrid_search(
                self.collection_name,  # Назва колекції
                [full_text_search_req, dense_req],  # Запити для гібридного пошуку
                ranker=RRFRanker(),  # Ранжувальник для комбінування результатів
                limit=k,  # Кількість результатів
                output_fields=output_fields,  # Поля для повернення
            )
        else:
            raise ValueError("Invalid mode")  # Помилка при неправильному режимі пошуку
        # Форматування результатів пошуку у зручний формат
        return [
            {
                "doc_id": doc["entity"]["doc_id"],  # Ідентифікатор документа
                "chunk_id": doc["entity"]["chunk_id"],  # Ідентифікатор фрагмента
                "content": doc["entity"]["content"],  # Текстовий вміст
                "score": doc["distance"],  # Оцінка релевантності
                "file_path": doc["entity"]["file_path"],  # Шлях до файлу
                "file_name": doc["entity"]["file_name"],  # Назва файлу
                "upload_date": doc["entity"]["upload_date"],  # Дата завантаження
            }
            for doc in results[0]  # Обробка кожного результату пошуку
        ]
    


class VideoHybridRetriever:
    """Клас для гібридного пошуку відеоматеріалів з використанням щільних та розріджених векторів"""
    def __init__(self, client, collection_name, dense_embedding_function=None):
            """Ініціалізація об'єкта VideoHybridRetriever
            
            Параметри:
            client - клієнт Milvus для взаємодії з базою даних
            collection_name - назва колекції для зберігання відеоматеріалів
            dense_embedding_function - функція для створення щільних векторів з тексту
            """
            self.collection_name = collection_name  # Зберігаємо назву колекції
            self.embedding_function = dense_embedding_function  # Функція для створення векторних ембедінгів
            self.use_reranker = True  # Прапорець для використання ранжувальника
            self.use_sparse = True  # Прапорець для використання розріджених векторів
            self.client = client  # Зберігаємо посилання на клієнт Milvus

    # ---------- створення колекції ----------
    def build_collection(self):
        """Створення або завантаження колекції для відеоматеріалів з відповідною схемою та індексами"""
        if self.client.has_collection(self.collection_name):
            self.client.load_collection(self.collection_name)  # Завантажуємо існуючу колекцію в пам'ять
            return

        # Визначення розмірності щільного вектора залежно від типу функції ембедінгу
        dense_dim = (
            self.embedding_function.dim["dense"]
            if isinstance(self.embedding_function.dim, dict)
            else self.embedding_function.dim
        )

        # Створення схеми колекції для зберігання відеоматеріалів
        schema = MilvusClient.create_schema(enable_dynamic_field=True)  # Дозволяємо динамічні поля

        # Основний текст + вектори
        schema.add_field("content", DataType.VARCHAR, max_length=65535,  # Поле для зберігання транскрипції відео
                         analyzer_params={  # Параметри аналізатора тексту
                             "tokenizer": "standard",  # Стандартний токенізатор
                             "filter": [
                                 "lowercase",  # Перетворення на нижній регістр
                                 {"type": "length",  "max": 200},  # Обмеження довжини токенів
                                 {"type": "stemmer", "language": "russian"},  # Стемінг для російської мови
                                 {"type": "stop",    "stop_words": ["_russian_"]},  # Видалення стоп-слів
                             ],
                         },
                         enable_match=True, enable_analyzer=True)  # Включення точного співпадіння та аналізатора
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)  # Поле для розрідженого вектора
        schema.add_field("dense_vector",  DataType.FLOAT_VECTOR, dim=dense_dim)  # Поле для щільного вектора

        # Службові поля документа для ідентифікації та метаданих
        schema.add_field("original_uuid",  DataType.VARCHAR, max_length=128)  # Унікальний ідентифікатор оригінального документа
        schema.add_field("doc_id",         DataType.VARCHAR, max_length=64)  # Ідентифікатор документа
        schema.add_field("chunk_id",       DataType.VARCHAR, is_primary=True,  # Первинний ключ - ідентифікатор фрагмента
                         auto_id=False,    max_length=64)
        schema.add_field("original_index", DataType.INT32, nullable=True)  # Індекс фрагмента в оригінальному документі

        # Метадані відео (всі — VARCHAR/INT для простоти фільтрації)
        schema.add_field("video_id",      DataType.VARCHAR, max_length=32, nullable=True)  # Ідентифікатор відео (наприклад, YouTube ID)
        schema.add_field("unique_video_id",     DataType.VARCHAR, max_length=40, nullable=True)  # Унікальний ідентифікатор відео
        schema.add_field("title",         DataType.VARCHAR, max_length=512, nullable=True)  # Назва відео
        schema.add_field("uploader",      DataType.VARCHAR, max_length=256, nullable=True)  # Автор відео
        schema.add_field("duration",      DataType.INT64, nullable=True)  # Тривалість відео в секундах
        schema.add_field("upload_date",   DataType.VARCHAR, max_length=1024)  # Дата завантаження відео
        schema.add_field("view_count",    DataType.INT64, nullable=True)  # Кількість переглядів
        schema.add_field("like_count",    DataType.INT64, nullable=True)  # Кількість вподобань
        schema.add_field("url",           DataType.VARCHAR, max_length=1024, nullable=True)  # URL відео
        schema.add_field("file_path",     DataType.VARCHAR, max_length=1024)  # Шлях до файлу
        schema.add_field("file_name",     DataType.VARCHAR, max_length=512)  # Назва файлу

        # BM25 для розріджених векторів з тексту транскрипції
        schema.add_function(Function(
            name="bm25",  # Назва функції
            function_type=FunctionType.BM25,  # Тип функції - BM25
            input_field_names=["content"],  # Вхідне поле - текстовий контент
            output_field_names="sparse_vector",  # Вихідне поле - розріджений вектор
        ))

        # Налаштування індексів для пришвидшення пошуку
        index_params = MilvusClient.prepare_index_params()  # Створюємо параметри індексів
        index_params.add_index(
            field_name="sparse_vector",  # Поле для індексації
            index_type="SPARSE_INVERTED_INDEX",  # Тип індексу для розріджених векторів
            metric_type="BM25",  # Метрика для порівняння - BM25
        )
        index_params.add_index(
            field_name="dense_vector", index_type="FLAT", metric_type="IP"  # Індекс для щільних векторів з метрикою внутрішнього добутку
        )

        # Створення колекції з визначеною схемою та індексами
        self.client.create_collection(
            collection_name=self.collection_name,  # Назва колекції
            schema=schema,  # Схема колекції
            index_params=index_params,  # Параметри індексів
        )


    def insert_data(self, chunk, metadata):
        """Вставка даних у колекцію відео
        
        Параметри:
        chunk - текстовий фрагмент транскрипції для вбудовування у вектор
        metadata - метадані відео (ідентифікатори, назва, автор, тощо)
        """
        embedding = self.embedding_function([chunk])  # Створюємо векторне представлення тексту
        if isinstance(embedding, dict) and "dense" in embedding:
            dense_vec = embedding["dense"][0]  # Отримуємо щільний вектор з словника
        else:
            dense_vec = embedding[0]  # Отримуємо щільний вектор напряму
        self.client.insert(
            self.collection_name, {"dense_vector": dense_vec, **metadata}  # Вставляємо вектор та метадані в колекцію
        )

    def upsert_data(self, chunk, metadata):
        """Оновлення або вставка даних у колекцію відео
        
        Параметри:
        chunk - текстовий фрагмент транскрипції для вбудовування
        metadata - метадані відео
        """
        embedding = self.embedding_function([chunk])  # Створюємо векторне представлення тексту
        dense_vec = embedding["dense"][0] if isinstance(embedding, dict) else embedding[0]  # Отримуємо щільний вектор

        # Примусове перегенерування розрідженого вектора
        self.client.upsert(
            self.collection_name,
            {
                "content": chunk,          # <-- критично важливо для генерації розрідженого вектора
                "dense_vector": dense_vec,  # Щільний вектор
                **metadata  # Метадані відео
            }
        )

    def search(self, query: str, k: int = 20, mode="hybrid"):
        """Пошук відеоматеріалів за запитом
        
        Параметри:
        query - текстовий запит користувача
        k - кількість результатів для повернення
        mode - режим пошуку: "sparse" (розріджений), "dense" (щільний) або "hybrid" (гібридний)
        
        Повертає:
        Список знайдених відеоматеріалів з метаданими та оцінками релевантності
        """

        # Поля, які будуть повернуті в результатах пошуку
        output_fields = [
                "content",  # Текстовий вміст (транскрипція)
                "original_uuid",  # Унікальний ідентифікатор оригінального документа
                "doc_id",  # Ідентифікатор документа
                "chunk_id",  # Ідентифікатор фрагмента
                "original_index",  # Індекс фрагмента в оригінальному документі
                "video_id",  # Ідентифікатор відео
                "unique_video_id",  # Унікальний ідентифікатор відео
                "title",  # Назва відео
                "uploader",  # Автор відео
                "duration",  # Тривалість відео
                "upload_date",  # Дата завантаження
                "view_count",  # Кількість переглядів
                "like_count",  # Кількість вподобань
                "url",  # URL відео
                "file_path",  # Шлях до файлу
                "file_name",  # Назва файлу
            ]
        # Створення вбудовування для запиту, якщо потрібно для щільного або гібридного пошуку
        if mode in ["dense", "hybrid"]:
            embedding = self.embedding_function([query])  # Створюємо векторне представлення запиту
            if isinstance(embedding, dict) and "dense" in embedding:
                dense_vec = embedding["dense"][0]  # Отримуємо щільний вектор з словника
            else:
                dense_vec = embedding[0]  # Отримуємо щільний вектор напряму

        # Пошук за розрідженим вектором (BM25)
        if mode == "sparse":
            results = self.client.search(
                collection_name=self.collection_name,  # Назва колекції
                data=[query],  # Текстовий запит
                anns_field="sparse_vector",  # Поле для пошуку - розріджений вектор
                limit=k,  # Кількість результатів
                output_fields=output_fields,  # Поля для повернення
            )
        # Пошук за щільним вектором
        elif mode == "dense":
            results = self.client.search(
                collection_name=self.collection_name,  # Назва колекції
                data=[dense_vec],  # Векторне представлення запиту
                anns_field="dense_vector",  # Поле для пошуку - щільний вектор
                limit=k,  # Кількість результатів
                output_fields=output_fields,  # Поля для повернення
            )
        # Гібридний пошук
        elif mode == "hybrid":
            full_text_search_params = {"metric_type": "BM25"}  # Параметри для пошуку за розрідженим вектором
            full_text_search_req = AnnSearchRequest(
                [query], "sparse_vector", full_text_search_params, limit=k  # Запит для розрідженого пошуку
            )

            dense_search_params = {"metric_type": "IP"}  # Параметри для пошуку за щільним вектором
            dense_req = AnnSearchRequest(
                [dense_vec], "dense_vector", dense_search_params, limit=k  # Запит для щільного пошуку
            )

            results = self.client.hybrid_search(
                self.collection_name,  # Назва колекції
                [full_text_search_req, dense_req],  # Запити для гібридного пошуку
                ranker=RRFRanker(),  # Ранжувальник для комбінування результатів
                limit=k,  # Кількість результатів
                output_fields=output_fields,  # Поля для повернення
            )
        else:
            raise ValueError("Invalid mode")  # Помилка при неправильному режимі пошуку
        # Форматування результатів пошуку з безпечним отриманням значень
        return [
        {
            "doc_id":      doc["entity"].get("doc_id"),          # ← .get() для безпечного отримання значення
            "doc_id":      doc["entity"].get("doc_id"),          # ← .get()
            "chunk_id":    doc["entity"].get("chunk_id"),
            "content":     doc["entity"].get("content"),
            "score":       doc["distance"],
            "video_id":    doc["entity"].get("video_id"),        # ← .get()
            "unique_video_id":   doc["entity"].get("unique_video_id"),
            "title":       doc["entity"].get("title"),
            "uploader":    doc["entity"].get("uploader"),
            "duration":    doc["entity"].get("duration"),
            "upload_date": doc["entity"].get("upload_date"),
            "view_count":  doc["entity"].get("view_count"),
            "like_count":  doc["entity"].get("like_count"),
            "url":         doc["entity"].get("url"),
            "file_path":   doc["entity"].get("file_path"),
            "file_name":   doc["entity"].get("file_name"),
        }
            for doc in results[0]
        ]
    


class ImageHybridRetriever:
    """Клас для гібридного пошуку зображень"""
    def __init__(self, client,  collection_name="hybrid", dense_embedding_function=None):
            """Ініціалізація об'єкта ImageHybridRetriever
            
            Параметри:
            client - клієнт Milvus
            collection_name - назва колекції
            dense_embedding_function - функція для створення щільних векторів
            """
            self.collection_name = collection_name
            self.embedding_function = dense_embedding_function
            self.use_reranker = True
            self.use_sparse = True
            self.client = client

    # ---------- створення колекції ----------
    def build_collection(self):
        """Створення або завантаження колекції для зображень"""
        if self.client.has_collection(self.collection_name):
            self.client.load_collection(self.collection_name)
            return

        # Визначення розмірності щільного вектора
        dense_dim = (
            self.embedding_function.dim["dense"]
            if isinstance(self.embedding_function.dim, dict)
            else self.embedding_function.dim
        )

        # Створення схеми колекції
        schema = MilvusClient.create_schema(enable_dynamic_field=True)

        # Основний текст + вектори
        schema.add_field("content", DataType.VARCHAR, max_length=65535,
                         analyzer_params={
                             "tokenizer": "standard",
                             "filter": [
                                 "lowercase",
                                 {"type": "length",  "max": 200},
                                 {"type": "stemmer", "language": "russian"},
                                 {"type": "stop",    "stop_words": ["_russian_"]},
                             ],
                         },
                         enable_match=True, enable_analyzer=True)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("dense_vector",  DataType.FLOAT_VECTOR, dim=dense_dim)

        # Службові поля документа
        schema.add_field("original_uuid",  DataType.VARCHAR, max_length=128)
        schema.add_field("doc_id",         DataType.VARCHAR, max_length=64)
        schema.add_field("chunk_id",       DataType.VARCHAR, is_primary=True, auto_id=False,    max_length=64)
        schema.add_field("file_path",     DataType.VARCHAR, max_length=1024)
        schema.add_field("file_name",     DataType.VARCHAR, max_length=512)
        schema.add_field("upload_date",   DataType.VARCHAR, max_length=1024)
        schema.add_field("original_index", DataType.INT32)

        # BM25 для розріджених векторів
        schema.add_function(Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names="sparse_vector",
        ))

        # Налаштування індексів
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        index_params.add_index(
            field_name="dense_vector", index_type="FLAT", metric_type="IP"
        )

        # Створення колекції
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )


    def insert_data(self, chunk, metadata):
        """Вставка даних у колекцію зображень
        
        Параметри:
        chunk - текстовий опис зображення для вбудовування
        metadata - метадані зображення
        """
        embedding = self.embedding_function([chunk])
        if isinstance(embedding, dict) and "dense" in embedding:
            dense_vec = embedding["dense"][0]
        else:
            dense_vec = embedding[0]
        self.client.insert(
            self.collection_name, {"dense_vector": dense_vec, **metadata}
        )

    def upsert_data(self, chunk, metadata):
        """Оновлення або вставка даних у колекцію зображень
        
        Параметри:
        chunk - текстовий опис зображення для вбудовування
        metadata - метадані зображення
        """
        embedding = self.embedding_function([chunk])
        dense_vec = embedding["dense"][0] if isinstance(embedding, dict) else embedding[0]

        # Примусове перегенерування розрідженого вектора
        self.client.upsert(
            self.collection_name,
            {
                "content": chunk,          # <-- критично важливо
                "dense_vector": dense_vec,
                **metadata
            }
        )

    def search(self, query: str, k: int = 20, mode="hybrid"):
        """Пошук зображень за запитом
        
        Параметри:
        query - текстовий запит
        k - кількість результатів
        mode - режим пошуку: "sparse", "dense" або "hybrid"
        
        Повертає:
        Список знайдених зображень з метаданими та оцінками релевантності
        """

        # Поля, які будуть повернуті в результатах пошуку
        output_fields = [
                "content",
                "file_path",
                "file_name",
                "upload_date",  
                "original_uuid",
                "doc_id",
                "chunk_id",
            ]
        # Створення вбудовування для запиту, якщо потрібно
        if mode in ["dense", "hybrid"]:
            embedding = self.embedding_function([query])
            if isinstance(embedding, dict) and "dense" in embedding:
                dense_vec = embedding["dense"][0]
            else:
                dense_vec = embedding[0]

        # Пошук за розрідженим вектором (BM25)
        if mode == "sparse":
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query],
                anns_field="sparse_vector",
                limit=k,
                output_fields=output_fields,
            )
        # Пошук за щільним вектором
        elif mode == "dense":
            results = self.client.search(
                collection_name=self.collection_name,
                data=[dense_vec],
                anns_field="dense_vector",
                limit=k,
                output_fields=output_fields,
            )
        # Гібридний пошук
        elif mode == "hybrid":
            full_text_search_params = {"metric_type": "BM25"}
            full_text_search_req = AnnSearchRequest(
                [query], "sparse_vector", full_text_search_params, limit=k
            )

            dense_search_params = {"metric_type": "IP"}
            dense_req = AnnSearchRequest(
                [dense_vec], "dense_vector", dense_search_params, limit=k
            )

            results = self.client.hybrid_search(
                self.collection_name,
                [full_text_search_req, dense_req],
                ranker=RRFRanker(),
                limit=k,
                output_fields=output_fields,
            )
        else:
            raise ValueError("Invalid mode")
        # Форматування результатів пошуку
        return [
            {
                "doc_id": doc["entity"]["doc_id"],
                "chunk_id": doc["entity"]["chunk_id"],
                "content": doc["entity"]["content"],
                "file_path": doc["entity"]["file_path"],
                "file_name": doc["entity"]["file_name"],
                "upload_date": doc["entity"]["upload_date"],
                "original_uuid": doc["entity"]["original_uuid"],    
                "score": doc["distance"],
            }
            for doc in results[0]
        ]





class AudioHybridRetriever:
    """Клас для гібридного пошуку audio"""
    def __init__(self, client, collection_name, dense_embedding_function=None):
            self.collection_name = collection_name
            self.embedding_function = dense_embedding_function
            self.use_reranker = True
            self.use_sparse = True
            self.client = client

    # ---------- створення колекції ----------
    def build_collection(self):
        """Створення або завантаження колекції для відеоматеріалів"""
        if self.client.has_collection(self.collection_name):
            self.client.load_collection(self.collection_name)
            return

        # Визначення розмірності щільного вектора
        dense_dim = (
            self.embedding_function.dim["dense"]
            if isinstance(self.embedding_function.dim, dict)
            else self.embedding_function.dim
        )

        # Створення схеми колекції
        schema = MilvusClient.create_schema(enable_dynamic_field=True)

        # Основний текст + вектори
        schema.add_field("content", DataType.VARCHAR, max_length=65535,
                         analyzer_params={
                             "tokenizer": "standard",
                             "filter": [
                                 "lowercase",
                                 {"type": "length",  "max": 200},
                                 {"type": "stemmer", "language": "russian"},
                                 {"type": "stop",    "stop_words": ["_russian_"]},
                             ],
                         },
                         enable_match=True, enable_analyzer=True)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("dense_vector",  DataType.FLOAT_VECTOR, dim=dense_dim)

        # Службові поля документа
        schema.add_field("original_uuid",  DataType.VARCHAR, max_length=128)
        schema.add_field("doc_id",         DataType.VARCHAR, max_length=64)
        schema.add_field("chunk_id",       DataType.VARCHAR, is_primary=True,
                         auto_id=False,    max_length=64)
        schema.add_field("audio_id",      DataType.VARCHAR, max_length=32, nullable=True)
        schema.add_field("title",         DataType.VARCHAR, max_length=512, nullable=True)
        schema.add_field("file_path",     DataType.VARCHAR, max_length=1024)
        schema.add_field("file_name",     DataType.VARCHAR, max_length=512)

        # BM25 для розріджених векторів
        schema.add_function(Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names="sparse_vector",
        ))

        # Налаштування індексів
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        index_params.add_index(
            field_name="dense_vector", index_type="FLAT", metric_type="IP"
        )

        # Створення колекції
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )


    def insert_data(self, chunk, metadata):
        """Вставка даних у колекцію відео
        
        Параметри:
        chunk - текстовий фрагмент для вбудовування
        metadata - метадані відео
        """
        embedding = self.embedding_function([chunk])
        if isinstance(embedding, dict) and "dense" in embedding:
            dense_vec = embedding["dense"][0]
        else:
            dense_vec = embedding[0]
        self.client.insert(
            self.collection_name, {"dense_vector": dense_vec, **metadata}
        )

    def upsert_data(self, chunk, metadata):
        """Оновлення або вставка даних у колекцію відео
        
        Параметри:
        chunk - текстовий фрагмент для вбудовування
        metadata - метадані відео
        """
        embedding = self.embedding_function([chunk])
        dense_vec = embedding["dense"][0] if isinstance(embedding, dict) else embedding[0]

        # Примусове перегенерування розрідженого вектора
        self.client.upsert(
            self.collection_name,
            {
                "content": chunk,          # <-- критично важливо
                "dense_vector": dense_vec,
                **metadata
            }
        )

    def search(self, query: str, k: int = 20, mode="hybrid"):
        # Поля, які будуть повернуті в результатах пошуку
        output_fields = [
                "content",
                "original_uuid",
                "doc_id",
                "chunk_id",
                "audio_id",
                "title",
                "file_path",
                "file_name",
            ]

        # Створення вбудовування для запиту, якщо потрібно
        if mode in ["dense", "hybrid"]:
            embedding = self.embedding_function([query])
            if isinstance(embedding, dict) and "dense" in embedding:
                dense_vec = embedding["dense"][0]
            else:
                dense_vec = embedding[0]

        # Пошук за розрідженим вектором (BM25)
        if mode == "sparse":
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query],
                anns_field="sparse_vector",
                limit=k,
                output_fields=output_fields,
            )
        # Пошук за щільним вектором
        elif mode == "dense":
            results = self.client.search(
                collection_name=self.collection_name,
                data=[dense_vec],
                anns_field="dense_vector",
                limit=k,
                output_fields=output_fields,
            )
        # Гібридний пошук
        elif mode == "hybrid":
            full_text_search_params = {"metric_type": "BM25"}
            full_text_search_req = AnnSearchRequest(
                [query], "sparse_vector", full_text_search_params, limit=k
            )

            dense_search_params = {"metric_type": "IP"}
            dense_req = AnnSearchRequest(
                [dense_vec], "dense_vector", dense_search_params, limit=k
            )

            results = self.client.hybrid_search(
                self.collection_name,
                [full_text_search_req, dense_req],
                ranker=RRFRanker(),
                limit=k,
                output_fields=output_fields,
            )
        else:
            raise ValueError("Invalid mode")
        # Форматування результатів пошуку з безпечним отриманням значень
        return [
        {
            "doc_id":      doc["entity"].get("doc_id"),          # ← .get()
            "chunk_id":    doc["entity"].get("chunk_id"),
            "content":     doc["entity"].get("content"),
            "score":       doc["distance"],
            "audio_id":    doc["entity"].get("audio_id"),        # ← .get()
            "title":       doc["entity"].get("title"),
            "upload_date": doc["entity"].get("upload_date"),
            "file_path":   doc["entity"].get("file_path"),
            "file_name":   doc["entity"].get("file_name"),
        }
            for doc in results[0]
        ]
    