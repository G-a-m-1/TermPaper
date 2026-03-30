import os
import shutil
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
from fpdf import FPDF
from concurrent.futures import ThreadPoolExecutor

MAX_WORKERS = 12  
SOURCE_DIR = "Data/A_pdfs"
OUTPUT_DIR = "Data/D_pdfs"
LANG = "ukr+eng"
FONT_PATH = "Fonts/LiberationSans-Regular.ttf"
TESSERACT_CONFIG = "--oem 3 --psm 3"

def has_text(file_path: str) -> bool:
    """Перевіряє чи є в PDF текстовий шар"""
    try:
        reader = PdfReader(file_path, strict=False)
        for page in reader.pages:
            text = page.extract_text() or ""
            if len(text.strip()) > 20:
                return True
        return False
    except Exception as e:
        print(f"Помилка читання {file_path}: {e}")
        return False

def ocr_page(image):
    """Розпізнає одну сторінку"""
    return pytesseract.image_to_string(image, lang=LANG, config=TESSERACT_CONFIG)

def ocr_pdf(file_path: str, output_path: str) -> None:
    """Оцифровує PDF через OCR і зберігає результат"""
    try:
        images = convert_from_path(os.path.abspath(file_path), dpi=500)
        print(f"  Розпізнаю {len(images)} сторінок...")

        pdf = FPDF()

        if os.path.exists(FONT_PATH):
            pdf.add_font("FreeSans", "", FONT_PATH)
            font_name = "FreeSans"
        else:
            font_name = "Arial"
            print(f"Шрифт не знайдено, використовується Arial")

        # Розпізнаю всі сторінки паралельно
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            texts = list(executor.map(ocr_page, images))

        # Записую результат в PDF
        for text in texts:
            pdf.add_page()
            pdf.set_font(font_name, size=10)
            pdf.multi_cell(0, 5, text=text)

        pdf.output(output_path)
        print(f"  Оцифровано: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"Помилка OCR {file_path}: {e}")

def process_pdfs(source_dir: str = SOURCE_DIR, output_dir: str = OUTPUT_DIR) -> None:
    """Обробляє всі PDF — копіює текстові, оцифровує скани"""
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(source_dir) if f.lower().endswith(".pdf")]
    print(f"Знайдено {len(files)} файлів\n")

    copied = 0
    ocred = 0
    skipped = 0
    errors = 0

    for i, filename in enumerate(files, 1):
        src = os.path.join(source_dir, filename)
        dst = os.path.join(output_dir, filename)

        print(f"[{i}/{len(files)}] {filename}")

        # Пропускаю якщо вже є в output
        if os.path.exists(dst):
            print(f"  Вже існує, пропускаю")
            skipped += 1
            continue

        if has_text(src):
            # Текстовий PDF — просто копіюю
            shutil.copy2(src, dst)
            print(f"  Текстовий — скопійовано")
            copied += 1
        else:
            # Скан — оцифровую
            print(f"  Скан — запускаю OCR...")
            ocr_pdf(src, dst)
            ocred += 1

    print("\n" + "=" * 60)
    print("Готово")
    print("=" * 60)
    print(f"Скопійовано  : {copied}")
    print(f"Оцифровано   : {ocred}")
    print(f"Пропущено    : {skipped}")
    print("=" * 60)

if __name__ == "__main__":
    process_pdfs()
    input("\nРоботу завершено. Натисніть Enter для виходу...")
