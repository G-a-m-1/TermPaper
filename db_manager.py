import os
import ollama_manager
from queue import Queue
from threading import Thread
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Налаштування
DB_DIR = "./db"
DOCS_DIR = os.path.join("Data", "D_pdfs")
EMBEDDINGS = OllamaEmbeddings(model="qwen3-embedding:0.6b") #"nomic-embed-text"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", "!", "?", " "] # Пріоритет розрізу для збереження сенсу
)
BATCH_SIZE = 10


def process_single_pdf(file_path):
    """Функція для обробки одного файлу"""
    try:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        reader = PdfReader(file_path)
        source_url = (reader.metadata or {}).get("/Source", "")

        # Додаю джерело у metadata кожного документа
        if source_url:
            for doc in docs:
                doc.metadata["source_url"] = source_url

        docs = TEXT_SPLITTER.split_documents(docs)

        def clean_chunks(chunks:list)->list:
            """ Підфункція для очищення чанків від непотрібних символів"""
            cleaned = []
            for d in chunks:
                # Перебирає кожен символ тексту і залишає його якщо символ є ASCII або символ знаходиться в діапазоні кирилиці
                d.page_content = ''.join(c for c in d.page_content if (c.isascii() and c >= ' ') or '\u0400' <= c <= '\u04FF')
                # Перевірка на мінімальну довжину
                if len(d.page_content.strip()) < 30:
                    continue            
                cleaned.append(d)
            return cleaned

        return clean_chunks(docs)
    except Exception as e:
        print(f"Помилка при читанні {file_path}: {e}")
        return []

def db_add_clean_chunks(vectorstore:Chroma, processed_docs:list) -> None:
    """Завантажує у базу даних передані документи"""
    print(f"    Створено {len(processed_docs)} фрагментів...")
    for i in range(0, len(processed_docs), BATCH_SIZE):
        batch = processed_docs[i : i + BATCH_SIZE]
        try:
            vectorstore.add_documents(batch)
            print(f"    {i+1} - {min(i + BATCH_SIZE, len(processed_docs))} фрагменти успішно додані у базу.")
        except Exception as e:
            print(f"    Помилка: {e}")

def update_db(vectorstore, docs_dir = DOCS_DIR):
    """Функція для оновлення бази даних з файлів заданої директорії"""
    
    # Пошук файлів для обробки
    files_to_process = []
    for f in os.listdir(docs_dir):
        if f.lower().endswith(".pdf"):
            full_path = os.path.join(docs_dir, f)
            result = vectorstore.get(where={"source": full_path}, limit=1) # Перевірка чи вже є у базі
            if not result['ids']:
                files_to_process.append(full_path) # Якщо файлу нема - додаєм

    if not files_to_process:
        print("Нових документів не знайдено.")
        return

    print(f"Знайдено {len(files_to_process)} нових файлів. Обробка...")
    
    
    def producer(files:list, queue:Queue) -> None:
        """Функція для обробки багатьох файлів у потоці і передачі оброблених документів у чергу"""
        for i in range(len(files)):
            chunks = process_single_pdf(files[i])
            queue.put((chunks,i))
        queue.put(None)  # сигнал що все оброблено

    
    def consumer(queue:Queue, files:list) -> None:
        """Функція для завантаження у базу документів у потоці з черги"""
        while True:
            item = queue.get()
            if item is None:
                break
            chunks, index = item

            print(f"[{index+1}/{len(files)}] Обробка {os.path.basename(files[index])}...")
            db_add_clean_chunks(vectorstore, chunks)
            print(f"    Успішно оброблено!")


    queue = Queue(maxsize=3)  # буфер на 3 файли
    t_producer = Thread(target=producer, args=(files_to_process, queue))
    t_consumer = Thread(target=consumer, args=(queue, files_to_process))
    t_producer.start()
    t_consumer.start()
    t_producer.join()
    t_consumer.join()


def update_db_one_file(vectorstore, full_path:str) -> None:
    """Функція для оновлення бази даних з одного файлу"""
    chunks = process_single_pdf(full_path)
    db_add_clean_chunks(vectorstore, chunks)

def delete_db(vectorstore):
    """Функція для видалення бази даних"""
    all_ids = vectorstore.get()['ids']
    if not all_ids:
        print("[ПОПЕРЕДЖЕННЯ] База даних вже порожня.")
        return
    vectorstore.delete(ids=all_ids)
    print(f"Видалено {len(all_ids)} записів з бази даних.")


def get_vectorstore(db_dir:str = DB_DIR, embeddings:OllamaEmbeddings = EMBEDDINGS) -> Chroma:
    return Chroma(persist_directory=db_dir, embedding_function=embeddings)


if __name__ == "__main__":
    ollama_manager.start_ollama(False)
    if not os.path.exists(DOCS_DIR):
        print(f"[ПОПЕРЕДЖЕННЯ] Вхідна папка '{DOCS_DIR}' не існує!")
        os.makedirs(DOCS_DIR)
        
    vectorstore = get_vectorstore()
    while True:
        print("\n\n\nРежими роботи:\n1. Завантажити файли до бази\n2. Очистити всю базу\n0. Вийти з програми")
        while True:
            mode = input("\nВаш вибір: ")
            if mode.isdigit() and int(mode) in [0, 1, 2]:
                mode = int(mode)
                break
            print("Некоректний вибір. Спробуйте ще раз.")
        
        if mode == 0:
            ollama_manager.stop_ollama()
            input("\nРоботу завершено. Натисніть Enter для виходу...")
            exit()
        elif mode == 1:
            update_db(vectorstore)
            print("\n"*10)
        elif mode == 2:
            delete_db(vectorstore)
            print("\n"*10)