import subprocess
import time
import httpx  
import os
from dotenv import load_dotenv


def start_ollama(hidden = False):
    """Перевіряє, чи працює Ollama, і запускає її, якщо ні."""
    url = "http://localhost:11434"
    
    try:
        # Перевіряю, чи сервер уже працює
        with httpx.Client(timeout=2) as client:
            client.get(url)
        print("Ollama вже запущена.")
        return
    except (httpx.ConnectError, httpx.HTTPError):
        pass
    except Exception as e:
        print("[ПОМИЛКА] Невідома помилка при провірці стану Ollama:\n",e)
        return


    print("Запуск Ollama...")
    kwargs = {}

    load_dotenv()
    env = os.environ.copy()
    models_path = os.getenv("OLLAMA_MODELS")
    if models_path:
        env["OLLAMA_MODELS"] = models_path

    if hidden:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL

        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            kwargs["preexec_fn"] = os.setpgrp
    else:
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE

    subprocess.Popen(["ollama", "serve"], env=env, **kwargs)
        
    # Чекаю, поки сервер почне відповідати (макс 30 секунд)
    for _ in range(30):
        try:
            with httpx.Client() as client:
                if client.get(url).status_code == 200:
                    print("Сервер Ollama готовий до роботи!")
                    return
        except Exception as e:
            time.sleep(1)

    print("[ПОМИЛКА] Не вдалося запустити Ollama.")


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
        print(f"[ПОМИЛКА] Не вдалося вимкнути Ollama:\n{e}")


if __name__ == "__main__":
    while True:
        print("Режими роботи:\n1. Запустити ollama\n2. Вимкнути ollama\n0. Вийти з програми")
        while True:
            mode = input("\nВаш вибір: ")
            if mode.isdigit() and int(mode) in [0, 1, 2]:
                mode = int(mode)
                break
            print("Некоректний вибір. Спробуйте ще раз.")
        
        if mode == 0:
            input("\nРоботу завершено. Натисніть Enter для виходу...")
            exit()
        elif mode == 1:
            start_ollama()
            print("\n"*10)
        elif mode == 2:
            stop_ollama()
            print("\n"*10)