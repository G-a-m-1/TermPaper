import os
import pytesseract # (tesseract)
import shutil
from pypdf import PdfReader, PdfWriter
from pdf2image import convert_from_path # (poppler)
from fpdf import FPDF
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from PIL import ImageOps

MAX_WORKERS = os.cpu_count() or 12  
SOURCE_DIR = os.path.join("Data", "A_pdfs")
OUTPUT_DIR = os.path.join("Data", "D_pdfs")
LANG = "ukr+eng"
FONT_PATH = os.path.join("Fonts", "LiberationSans-Regular.ttf")
TESSERACT_CONFIG = "--oem 3 --psm 3"
DEBUG = False
DEBUG_DIR = "Debug"

COMMON_WORDS = {
    "що", "як", "для", "або", "та", "які", "але", "він", "вона", "вони", "ми", "ви", "це", "той", "ця", "яка","який", "про", "при", "від", "до", "на", "не", "за", "із", "по", "між", "над", "під", "без", "через", "після", "університеті", "університет", "студент", "навчання", "кафедра", "факультет", "наказ", "відповідно", "згідно", "затверджено", "розклад", "навчальний", "рік", "року", "наказу", "відділ", "декан", "ректор", "відомість", "протокол", "план", "львів", "львівський", "нацональний", "список", "списку", "року", "імені", "івана", "франка", "довідки", "ознайомлення", "копія", "копії", "документу", "документ", "академічної", "академічна", "пільги", "заява", "додаток", "даних", "фонд", "зберігання", "банку", "банк", "освіту", "освіта", "бюджет", "бюджеті", "прізвище", "форма", "одиниця", "заклад", "установа", "організація", "пункт"
}

BAD_CHARS = "|~^@#$%\\=`<>{}*"

def text_to_words(text:str)->list[str]:
    """Розділяє текст на окремі очищені слова"""
    for c in BAD_CHARS:
        text = text.replace(c, " ")# прибираю артефакти ocr
    words = text.split()
    return words

def is_correct_words(words:list[str],)->bool:
    """Перевіряє чи слова з однієї сторінки коректні"""
    if not words:
        return False

    if len(words) < 50:
        return False
            
    known_words = 0
    for w in words:
        if w.strip(".,;:!?()[]\"'") in COMMON_WORDS:
            known_words += 1

    if known_words >= 8:
        return True

    return False


def has_readable_text(file_path: str) -> bool:
    """Перевіряє чи є в PDF текст"""
    try:
        reader = PdfReader(file_path)
        num_pages_has_text = 0
        pages = reader.pages
        for page in pages:
            text = (page.extract_text() or "").lower()
            words = text_to_words(text)

            if is_correct_words(words):
                num_pages_has_text += 1
        
        if len(pages) > 0 and (num_pages_has_text/len(pages) >= 0.5):
            return True

        return False

    except Exception as e:
        print(f"\n[ПОМИЛКА] Помилка читання ({file_path}):\n{e}")
        return False

def is_valid_pdf(file_path: str) -> bool:
    """Перевіряє чи PDF файл не пошкоджений"""
    try:
        reader = PdfReader(file_path)
        _ = len(reader.pages)  # спроба прочитати сторінки
        return True
    except Exception as e:
        print(f"  [ПОПЕРЕДЖЕННЯ] Пошкоджений файл: {os.path.basename(file_path)}: {e}")
        return False

def ocr_page(image) -> None:
    """Обробляє і оцифровує одну сторінку"""
    #Збільшення контрасту
    if DEBUG:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        image.save(os.path.join(DEBUG_DIR, f"1.png"), format="PNG")

    def soft_threshold(p):
        if p < 30: return 0
        if p > 230: return 255
        return p

    image = ImageOps.autocontrast(image, cutoff=1)
    image = image.point(soft_threshold)
    
    if DEBUG:
        image.save(os.path.join(DEBUG_DIR, f"2.png"), format="PNG")
        input(f"Оброблено сторінку. Результат у {DEBUG_DIR}.\n")

    return pytesseract.image_to_string(image, lang=LANG, config=TESSERACT_CONFIG)

def ocr_pdf(file_path: str, output_path: str) -> bool:
    """Оцифровує один PDF-файл і зберігає результат у output_path"""
    try:
        # Обрахунок кількості сторінок файлу
        reader = PdfReader(file_path)
        original_metadata = reader.metadata
        num_pages = len(reader.pages)
        texts = []
        print(f"  Знайдено {num_pages} сторінок.")

        
        # для малих документів — потоки, для великих — процеси
        if num_pages <= max(MAX_WORKERS // 2, 1):
            ExecutorClass = ThreadPoolExecutor
        else:
            ExecutorClass = ProcessPoolExecutor

        workers = min(MAX_WORKERS, num_pages)
        with ExecutorClass(max_workers=workers) as executor:
            for i in range(0, num_pages, workers):
                last_page = min(i + workers, num_pages)
                print(f"  Обробка {i}-{last_page} сторінок...     ",end='\r')
            
                # Витягування сторінок файлу як зображення (poppler)
                chunk_images = convert_from_path(
                    file_path, 
                    dpi=400, 
                    first_page=i+1, 
                    last_page=last_page, 
                    grayscale=True
                )

                # Обробка зображень у текст
                chunk_texts = list(executor.map(ocr_page, chunk_images))
                texts.extend(chunk_texts)
            
                del chunk_images # Видалення оброблених зображень з ram
        

        # Записую результат в PDF
        pdf = FPDF()
        if os.path.exists(FONT_PATH):
            pdf.add_font("FreeSans", "", FONT_PATH)
            font_name = "FreeSans"
        else:
            font_name = "Arial"
            print(f"\n[ПОПЕРЕДЖЕННЯ] Шрифт не знайдено, використовується Arial")

        pdf.set_font(font_name, size=5)
        correct_page = 0
        for text in texts:
            # Очищення тексту
            words = text_to_words(text)
            if is_correct_words(words):
                correct_page += 1
                clean_text = " ".join(words)
                
                # Додавання у пдф
                pdf.add_page()
                pdf.multi_cell(0, 2, text=clean_text)
            else:
                pdf.add_page()

        if correct_page == 0:
            print(f"  [ПОПЕРЕДЖЕННЯ] Файл {os.path.basename(output_path)} не було записано. Не було знайдено корисної інформації.")
            return False

        print(f"  Корисних сторінок: {correct_page}/{num_pages}")
        pdf.output(output_path)

        new_reader = PdfReader(output_path)
        writer = PdfWriter()
        writer.append(new_reader)

        # Копіюю метадані
        new_metadata = {}
        if original_metadata:
            for key, value in original_metadata.items():
                new_metadata[key] = value
        writer.add_metadata(new_metadata)

        # Перезаписую файл з метаданими
        with open(output_path, 'wb') as f:
            writer.write(f)

        print(f"  Успішно оцифровано: {os.path.basename(output_path)}")
        return True

    except Exception as e:
        print(f"\n[ПОМИЛКА] Помилка OCR ({file_path}): {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def process_pdfs(source_dir: str = SOURCE_DIR, output_dir: str = OUTPUT_DIR) -> dict:
    """Основний код. Обробляє всі PDF — копіює текстові та оцифровує скани"""
    os.makedirs(output_dir, exist_ok=True) # Створюю вихідну папку якщо не існує
    stats: dict = {'copied': 0, 'ocred': 0, 'skipped': 0, 'errors': 0}

    # Пошук всіх pdf файлів з вхідної папки
    files = []
    for root, _, filenames in os.walk(source_dir):
        for f in filenames:
            if f.lower().endswith(".pdf"):
                files.append(os.path.join(root, f))

    for i in range(len(files)):
        src = files[i]
        dst = os.path.join(output_dir, os.path.basename(files[i]))

        print(f"[{i+1}/{len(files)}] {files[i]}")

        # Пропускаю якщо вже є  цей файл в output
        if os.path.exists(dst):
            print(f"  Вже існує, пропуск")
            stats['skipped'] += 1
            continue

        # Чи цілий
        if not is_valid_pdf(src):
            print(f"[ПОМИЛКА] Файл {src} пошкоджений")
            stats['errors'] += 1
            continue

        # Текстовий PDF — копіюю
        if has_readable_text(src):
            shutil.copy2(src, dst)
            print(f"  Текстовий — скопійовано")
            stats['copied'] += 1

        # Скан — оцифровую
        else:
            print(f"  Скан — запуск оцифрування...")
            result = ocr_pdf(src, dst)
            if result:
                stats['ocred'] += 1
            else:
                stats['errors'] += 1
    print_summary(stats,output_dir)
    return stats

def process_one_pdf(file_path:str, output_path: str|None = None) -> tuple[bool,str]:
    """Основний код. Обробляє  PDF — копіює якщо текстовий та оцифровує якщо скан. Повертає bool чи успішно оброблено та шлях куда оброблено"""
    if not output_path:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(file_path))
    
    # Створюю вихідну папку якщо не існує
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    # Пропускаю якщо вже є цей файл в output
    if os.path.exists(output_path):
        print(f"  Вже існує, пропуск")
        return False,output_path

    # Чи цілий
    if not is_valid_pdf(file_path):
        print(f"[ПОМИЛКА] Файл {file_path} пошкоджений")
        return False,output_path

    # Текстовий PDF — копіюю
    if has_readable_text(file_path):
        shutil.copy2(file_path, output_path)
        print(f"  Текстовий — скопійовано")
        return True,output_path

    # Скан — оцифровую
    print(f"  Скан — запуск оцифрування...")
    result = ocr_pdf(file_path, output_path)
    return result, output_path

def print_summary(stats: dict, save_dir: str) -> None:
    total = stats['copied'] + stats['ocred'] + stats['skipped'] + stats['errors']
    print("\n\n" + "=" * 80)
    print("Оцифровка завершена")
    print("=" * 80)
    print(f"Скопійовано        : {stats['copied']}")
    print(f"Оцифровано         : {stats['ocred']}")
    print(f"Пропущено (вже є)  : {stats['skipped']}")
    print(f"Помилок            : {stats['errors']}")
    print(f"Всього файлів      : {total}")
    print(f"Папка збереження   : {os.path.abspath(save_dir)}")
    print("=" * 80)

def delete_output(output_dir: str = OUTPUT_DIR) -> None:
    # Видалення всіх pdf файлів з вихідної папки
    if not os.path.exists(output_dir):
        return

    for f in os.listdir(output_dir):
        if f.lower().endswith(".pdf"):
            full_path = os.path.join(output_dir, f)
            try:
                print(f" Видалення {f}...",end="")
                os.remove(full_path)
                print("Успішно!")
            except Exception as e:
                print(f"Не вдалося видалити {full_path}: {e}")
    print(" Видалення завершено.")


if __name__ == "__main__":
    while True:
        print("Режими роботи:\n1. Обробити всі файли\n2. Видалення всіх pdf файлів з вихідної папки\n0. Вийти з програми")
        while True:
            mode = input("\nВаш вибір: ")
            if mode.isdigit() and int(mode) in [0, 1, 2]:
                mode = int(mode)
                break
            print("Некоректний вибір. Спробуйте ще раз.")
        
        match mode:
            case 0:
                exit()
            case 1:
                process_pdfs()
                print("\n"*10)
            case 2:
                delete_output()
                print("\n"*10)