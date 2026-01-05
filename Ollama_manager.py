import subprocess
import time
import httpx  
import os

def start_ollama():
    """Перевіряє, чи працює Ollama, і запускає її, якщо ні."""
    url = "http://localhost:11434"
    
    try:
        # Перевіряю, чи сервер уже працює
        with httpx.Client(timeout=2) as client:
            client.get(url)
        print("Ollama вже запущена.")
    except (httpx.ConnectError, httpx.HTTPError):
        print("Запуск Ollama...")
        subprocess.Popen(
            ["ollama", "serve"],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        
        # Чекаю, поки сервер почне відповідати (макс 30 секунд)
        for i in range(30):
            try:
                with httpx.Client() as client:
                    client.get(url)
                print("Сервер Ollama готовий до роботи!")
                return
            except:
                time.sleep(1)
        print("Не вдалося запустити Ollama.")


def stop_ollama():
    """Завершує роботу всіх процесів Ollama."""
    print("Вимкнення Ollama...")
    try:
        if os.name == 'nt':  # Для Windows
            # /F — примусово, /IM — за іменем образу
            subprocess.run(
                ["taskkill", "/F", "/IM", "ollama.exe"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
        else:  # Для Linux / Mac
            subprocess.run(
                ["pkill", "ollama"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL)
        
        print("Ollama успішно вимкнена.")
    except Exception as e:
        print(f"Не вдалося вимкнути Ollama: {e}")