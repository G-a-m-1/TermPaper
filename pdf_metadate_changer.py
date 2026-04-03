import os
from pypdf import PdfReader, PdfWriter


def file_selection()->str:
    file_path = ""
    while True:
        file_path = input("Перетягніть PDF-файл у це вікно і натисніть Enter:\n> ")
        file_path = file_path.strip().replace('"', '').replace("'", "") # Очищення шляху
        if not os.path.exists(file_path): # Перевірка наявності файлу
            print(f"\nФайл не знайдено за шляхом: {file_path}. Спробуйте знову.\n")
        elif not file_path.lower().endswith(".pdf"): # Перевірка розширення
            print(f"\nФайл '{os.path.basename(file_path)}' не є PDF-документом.\n")
        else:
            break
    return file_path

def check_pdf_metadata() -> None:  
    file_path = file_selection()
    while True:
        print("\n\n\nРежими роботи:\n1. Вибрати інший файл\n2. Вивести всі метадані файлу\n3. Змінити метадані '/Source' файлу\n0. Вийти з програми")
        while True:
            mode = input("\nВаш вибір: ")
            if mode.isdigit() and int(mode) in [0, 1, 2, 3]:
                mode = int(mode)
                break
            print("Некоректний вибір. Спробуйте ще раз.")
       
        if mode == 0:
            exit()
        elif mode == 1:
            file_path = file_selection()
            print("\n"*10)
        elif mode == 2:
            mode_1(file_path)
            print("\n"*10)
        elif mode == 3:
            mode_2(file_path)
            print("\n"*10)

def mode_1(file_path:str)-> None: 
    print("Режими роботи 1. Виведення всіх метаданих файлу...")
    try:    
        reader = PdfReader(file_path)
        meta = reader.metadata

        if not meta:
            print("\nМетадані файла порожні або відсутні.")
        else:
            for key, value in meta.items(): # Виводжу всі ключі
                print(f"{key:20}: {value}")
        print(f"\nСторінок: {len(reader.pages)}")
        input("\nНатисніть Enter для повернення...")

    except Exception as e:
        print(f"\n[ПОМИЛКА]: {e}")
        input("\nНатисніть Enter для повернення...")
    

def mode_2(file_path:str)-> None: 
    print("Режими роботи 2. Зміна метаданих '/Source' файлу...")
    try:
        source = input("\nВставте посилання на джерело файлу і натисніть Enter:\n> ")
        reader = PdfReader(file_path, strict=False)
        writer = PdfWriter()
        writer.append(reader)
        writer.add_metadata({"/Source": source})
        tmp_path = file_path + ".tmp"
        with open(tmp_path, 'wb') as f: # записую в тичасовий файл, щоб в разі помилки не втратити pdf
            writer.write(f)
        os.replace(tmp_path, file_path) # замінюю оригінал на змінений файл

        print(f"\nМетадані успішно збережено.")
        input("\nНатисніть Enter для повернення...")
    except Exception as e:
        print(f"[ПОМИЛКА] Не вдалося записати метадані: {e}")
        input("\nНатисніть Enter для повернення...")

if __name__ == "__main__":
    check_pdf_metadata()