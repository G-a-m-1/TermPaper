import os
import re
import Ollama_manager
from concurrent.futures import ThreadPoolExecutor
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Налаштування
DB_DIR = "./db"
DOCS_DIR = os.path.join("Data", "D_pdfs")
EMBEDDINGS = OllamaEmbeddings(model="qwen3-embedding:0.6b", num_ctx = 2048) #"nomic-embed-text"
MAX_WORKERS = 12
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
        return TEXT_SPLITTER.split_documents(docs)
    except Exception as e:
        print(f"Помилка при читанні {file_path}: {e}")
        return []

def clean_chunks(chunks):
    cleaned = []
    for d in chunks:
        # Видаляю все, що не є текстом, цифрами, пунктуацією чи кирилицею
        text = re.sub(r'[^\x20-\x7E\u0400-\u04FF\s]', '', text) 
        # Нормалізація пробілів та переносів (текст одним рядком)
        text = " ".join(text.split())
        # Перевірка на мінімальну довжину
        if len(text.strip()) < 30:
            continue            
        d.page_content = text
        cleaned.append(d)
    return cleaned

def update_database():
    """Функція для оновлення бази даних"""
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=EMBEDDINGS)
    
    # Отримання метаданих
    existing_docs = vectorstore.get(include=['metadatas'])
    existing_sources = {m['source'] for m in existing_docs['metadatas']} if existing_docs['metadatas'] else set()

    files_to_process = [
        os.path.join(DOCS_DIR, f) for f in os.listdir(DOCS_DIR) 
        if f.lower().endswith(".pdf") and os.path.join(DOCS_DIR, f) not in existing_sources
    ]

    if not files_to_process:
        print("Нових документів не знайдено.")
        return

    print(f"Знайдено {len(files_to_process)} нових файлів. Обробка...")

    # Паралельне завантаження та розбиття на чанки
    all_raw_chunks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_single_pdf, files_to_process))
        for chunks in results:
            all_raw_chunks.extend(chunks)

    all_new_chunks = clean_chunks(all_raw_chunks)
    # Додавання в базу
    print(f"Створено {len(all_new_chunks)} фрагментів.")
    for i in range(0, len(all_new_chunks), BATCH_SIZE):
        print(f"Генерація ембеддінгів від {i} до {i + BATCH_SIZE} фрагменту...\t\t",end="")
        batch = all_new_chunks[i : i + BATCH_SIZE]
        try:
            vectorstore.add_documents(batch)
            print("Успішно!")
        except Exception as e:
            print(f"Пропущено фрагменти через помилку: {e}")

if __name__ == "__main__":
    Ollama_manager.start_ollama()
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
    update_database()
    Ollama_manager.stop_ollama()
    print("Завершено.")
    input()