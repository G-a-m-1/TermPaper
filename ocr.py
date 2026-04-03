import os
import pytesseract # (tesseract)
import shutil
from pypdf import PdfReader, PdfWriter
from pdf2image import convert_from_path # (poppler)
from fpdf import FPDF
from concurrent.futures import ProcessPoolExecutor
from PIL import ImageOps

MAX_WORKERS = 12  
SOURCE_DIR = os.path.join("Data", "A_pdfs")
OUTPUT_DIR = os.path.join("Data", "D_pdfs")
LANG = "ukr+eng"
FONT_PATH = os.path.join("Fonts", "LiberationSans-Regular.ttf")
TESSERACT_CONFIG = "--oem 3 --psm 3"
DEBUG = False
DEBUG_DIR = "Debug"


def has_text(file_path: str) -> bool:
    """Перевіряє чи є в PDF текст"""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text() or ""
            if len(text.strip()) > 30:
                return True
        return False
    except Exception as e:
        print(f"\n[ПОМИЛКА] Помилка читання ({file_path}):\n{e}")
        return False


def soft_threshold(p):
    if p < 50: return 0
    if p > 200: return 255
    return p

def ocr_page(image) -> None:
    """Обробляє і оцифровує одну сторінку"""
    #Збільшення контрасту
    if DEBUG:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        image.save(os.path.join(DEBUG_DIR, f"1.png"), format="PNG")

    image = ImageOps.autocontrast(image, cutoff=1)
    image = image.point(soft_threshold)
    
    if DEBUG:
        image.save(os.path.join(DEBUG_DIR, f"2.png"), format="PNG")
        input(f"Оброблено сторінку. Результат у {DEBUG_DIR}.")

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

        step = MAX_WORKERS
        for i in range(0, num_pages, step):
            last_page = min(i + step, num_pages)
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
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
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

        pdf.set_font(font_name, size=6)
        for text in texts:
            pdf.add_page()
            pdf.multi_cell(0, 5, text=text)

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

        print(f"  Оцифровано: {os.path.basename(output_path)}")
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
    for f in os.listdir(source_dir):
        if f.lower().endswith(".pdf"):
            files.append(f)
    print(f"Знайдено {len(files)} файлів\n")

    for i in range(len(files)):
        src = os.path.join(source_dir, files[i])
        dst = os.path.join(output_dir, files[i])

        print(f"[{i+1}/{len(files)}] {files[i]}")

        # Пропускаю якщо вже є  цей файл в output
        if os.path.exists(dst):
            print(f"  Вже існує, пропуск")
            stats['skipped'] += 1
            continue

        # Текстовий PDF — копіюю
        if has_text(src):
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

def print_summary(stats: dict, save_dir: str) -> None:
    total = stats['downloaded'] + stats['skipped']
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
                os.remove(full_path)
            except Exception as e:
                print(f"Не вдалося видалити {f}: {e}")


if __name__ == "__main__":
    process_pdfs()
    input("\n\nРоботу завершено. Натисніть Enter для виходу...")
