import os
import ollama_manager
import db_manager
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Налаштування
DB_DIR = "./db"
DEBUG_INFORMATION = True
MODEL = "qwen3.5:2b"
NUM_CTX = 30000
K = 22
THINK = False

def initialize():
    """Ініціалізація Ollama, векторної бази та RAG-ланцюга."""
    ollama_manager.start_ollama(False)
 
    if not os.path.exists(DB_DIR):
        print(f"Помилка: Папка {DB_DIR} не знайдена!")
        exit()
 
    # Моделі
    llm = ChatOllama(model=MODEL, temperature=0.3, num_ctx=NUM_CTX, reasoning=THINK)
 
    # Ініціалізація векторної бази
    vectorstore = db_manager.get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": K})
 
    # Шаблон промпту
    template = """Ви помічник університету. Відповідайте повно на питання, використовуючи наданий контекст. Якщо відповіді немає в контексті, скажіть: "На жаль, я не знайшов цієї інформації в базі даних університету".
Відповідай виключно українською мовою. Не використовуй Markdown.

Контекст (кожен фрагмент містить посилання на джерело у форматі [Джерело: ..., сторінка ...]):
{context}


Питання користувача: {question}

Після своєї відповіді обов'язково вкажи список, який міститиме до трьох джерел найбільш підходящих до питання користувача і які ти найбільше використав для надання відповіді у форматі:
    Джерела інформації:
- Посилання, сторінк(а)/(и) X

Відповідь:
"""
    prompt = ChatPromptTemplate.from_template(template)
 
    rag_chain = prompt | llm | StrOutputParser()
 
    return vectorstore, retriever, rag_chain
 
 
def format_docs(docs):
    """Об'єднання документів у єдиний рядок контексту з метаданими."""
    parts = []
    for doc in docs:
        source_url = doc.metadata.get("source_url", "")
        pages = doc.metadata.get("pages", "")
        
        header = ""
        if source_url:
            header += f"[Джерело: {source_url}"
            if pages:
                header += f", сторінки {pages}"
            header += "]"
        
        parts.append(f"{header}\n{doc.page_content}" if header else doc.page_content)
    
    return "\n\n".join(parts)
 
def debug_docs(user_query: str, docs: list):
    """Виведення діагностичної інформації про знайдені документи."""
    if not DEBUG_INFORMATION:
        return
    print(f"{'='*115}\n")
    print(f"Питання користувача: {user_query}")
    print(f"Знайдено шматків: {len(docs)}")
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Невідоме джерело')
        source_url = doc.metadata.get('source_url', 'Невідоме джерело')
        page = doc.metadata.get("pages", 'Невідома к-сть сторінок')
        preview = doc.page_content[:400].replace('\n', ' ')
        print(f"\n[{i}]\tст.{page}\tсим.{len(doc.page_content)}\nФайл:{source} ({source_url})\nКонтент:{preview}...")
    print(f"{'-'*115}\n")
 
def debug_response(response: str):
    """Виведення відповіді моделі в консоль."""
    if not DEBUG_INFORMATION:
        return
    print(f"Відповідь моделі:\n{response}\n")
    print(f"{'='*115}\n")