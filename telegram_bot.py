import asyncio
import os
import ollama_manager
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.enums import ParseMode
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

ollama_manager.start_ollama()
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
if BOT_TOKEN == None:
    print(f"Помилка: BOT_TOKEN не знайдено!")
    exit()

# Налаштування
DB_DIR = "./db"
DEBUG_INFORMATION = True
# Моделі
EMBEDDINGS = OllamaEmbeddings(model="qwen3-embedding:0.6b", num_ctx = 2048) #"nomic-embed-text"
LLM = ChatOllama(model="qwen3:4b", temperature=0.3, num_ctx = 15000)

MAX_MESSAGE_LENGTH = 1000  # Обмеження довжини запиту

if not os.path.exists(DB_DIR):
    print(f"Помилка: Папка {DB_DIR} не знайдена!")
    exit()

# Ініціалізація векторної бази
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=EMBEDDINGS)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# Створюю шаблон промпту
template = """Ви помічник університету. Відповідайте повно на питання, використовуючи лише наданий контекст. 
Якщо відповіді немає в контексті, скажіть: "На жаль, я не знайшов цієї інформації в базі даних університету".
Відповідай виключно українською мовою.

Контекст:
{context}

Питання: {question}
Відповідь:"""

prompt = ChatPromptTemplate.from_template(template)

# Функція для об'єднання документів
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
      prompt
    | LLM
    | StrOutputParser()
)

# Налаштування бота
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    """Обробка команди /start"""
    await message.answer("Привіт! Я бот-помічник університету. Що ви хочете дізнатися?")

@dp.message(Command("help"))
async def help_cmd(message: types.Message):
    """Обробка команди /help"""
    await message.answer(
        "Як користуватися:\n"
        "Просто напишіть своє питання українською мовою.\n"
        "Наприклад: 'Налаштування eduroam на Android'"
    )

@dp.message()
async def handle_question(message: types.Message):
    """Обробка питань користувача"""
    # Перевіряю, чи є текст у повідомленні
    user_query = message.text
    if user_query is None:
        return

        # Обмеження довжини запиту
    if len(user_query) > MAX_MESSAGE_LENGTH:
        await message.answer("Питання занадто довге. Спробуйте сформулювати коротше.")
        return

    status_msg = await message.answer("Пошук відповіді...")
    
    try:
        # Отримання документів з бази
        docs = await asyncio.to_thread(retriever.invoke, user_query)
        
        # Вивід у консоль для діагностики 
        if DEBUG_INFORMATION == True:
            print(f"{'='*115}\n")
            print(f"Питання користувача: {user_query}")
            print(f"Знайдено шматків: {len(docs)}")
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'Невідоме джерело')
                preview = doc.page_content[:400].replace('\n', ' ')
                print(f"\n[{i}] Джерело:{source}\nКонтент:{preview}...")
            print(f"{'-'*115}\n")

        # Перевірка чи знайдено контекст
        if not docs:
            await status_msg.edit_text(
                "На жаль, не знайдено інформації по цьому запитанні."
            )
            return

        context_text = format_docs(docs)
        # Генерація відповіді через модель
        response = await asyncio.to_thread(rag_chain.invoke, {"context": context_text, "question": user_query})
        if DEBUG_INFORMATION == True:
            print(f"Відповідь моделі:\n{response}\n")
            print(f"{'='*115}\n")
        await status_msg.edit_text(response[:4096]) # 4096 - ліміт Telegram
        
    except asyncio.TimeoutError:
        await status_msg.edit_text("Перевищено час очікування. Спробуйте ще раз.")
    except Exception as e:
        print(f"Помилка: {e}")
        await status_msg.edit_text(f"Виникла технічна помилка. Спробуйте пізніше або перефразуйте питання.")

async def main():
    print(f"База даних: {DB_DIR}")
    print(f"Документів у базі: {vectorstore._collection.count()}")
    print("Бот запущений.")
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    except asyncio.CancelledError:
        pass
    finally:
        print("Закриття з'єднання з Telegram...")  
        await bot.session.close()
        print("З'єднання закрите.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот зупинений.")
    except Exception as e:
        print(f"Критична помилка: {e}")
    finally:
        print("Завершення роботи Ollama...")
        ollama_manager.stop_ollama()
        print("Роботу завершено.")
        input()