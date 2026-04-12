import os
import ollama_manager
import db_manager
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Налаштування
DB_DIR = "./db"
DEBUG_INFORMATION = True


def initialize():
    """Ініціалізація Ollama, векторної бази та RAG-ланцюга."""
    ollama_manager.start_ollama(False)
 
    if not os.path.exists(DB_DIR):
        print(f"Помилка: Папка {DB_DIR} не знайдена!")
        exit()
 
    # Моделі
    llm = ChatOllama(model="qwen3.5:4b", temperature=0.3, num_ctx=13000)
 
    # Ініціалізація векторної бази
    vectorstore = db_manager.get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 13})
 
    # Шаблон промпту
    template = """Ви помічник університету. Відповідайте повно на питання, використовуючи лише наданий контекст. 
Якщо відповіді немає в контексті, скажіть: "На жаль, я не знайшов цієї інформації в базі даних університету".
Відповідай виключно українською мовою.

Контекст (кожен фрагмент містить посилання на джерело у форматі [Джерело: ..., сторінка ...]):
{context}


Питання користувача: {question}

Після своєї відповіді обов'язково вкажи список лише використаних тобою джерел для надання відповіді (не більше трьох джерел) у форматі:
    Джерела інформації:
- Посилання, сторінка X

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
        page = doc.metadata.get("page", "")
        
        header = ""
        if source_url:
            header += f"[Джерело: {source_url}"
            if page != "":
                header += f", сторінка {page + 1}"  # PyMuPDF рахує з 0
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
        preview = doc.page_content[:400].replace('\n', ' ')
        print(f"\n[{i}] Файл:{source}\nКонтент:{preview}...")
    print(f"{'-'*115}\n")
 
def debug_response(response: str):
    """Виведення відповіді моделі в консоль."""
    if not DEBUG_INFORMATION:
        return
    print(f"Відповідь моделі:\n{response}\n")
    print(f"{'='*115}\n")