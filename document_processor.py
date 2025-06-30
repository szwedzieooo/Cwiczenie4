import os
import pdfplumber
import pytesseract
from docx import Document
from PIL import Image
import csv
from transformers import pipeline


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Wczytanie modelu Hugging Face do wykrywania języka
language_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

# Ścieżka do folderu z plikami
FOLDER_PATH = "pliki"

def extract_text_from_pdf(filepath):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text, "PDF"

def extract_text_from_docx(filepath):
    doc = Document(filepath)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text, "DOCX"

def extract_text_from_image(filepath, lang='eng+pol'):
    image = Image.open(filepath)
    text = pytesseract.image_to_string(image, lang=lang)
    return text, "OCR"

def detect_language(text):
    if len(text.strip()) < 20:
        return "unknown"
    result = language_detector(text[:500])[0]
    return result['label']

def save_text_to_file(text, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

def save_report(report_data, output_csv="report.csv"):
    headers = ['File', 'Format', 'Extraction Method', 'Language', 'Word Count']
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(report_data)

def process_file(filepath):
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filepath)[1].lower()
    report = []

    if ext == '.pdf':
        text, method = extract_text_from_pdf(filepath)
    elif ext == '.docx':
        text, method = extract_text_from_docx(filepath)
    elif ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        text, method = extract_text_from_image(filepath)
    else:
        print(f"Nieobsługiwany format: {filename}")
        return None

    word_count = len(text.split())
    language = detect_language(text)
    save_text_to_file(text, f"{filename}.txt")

    report.append([filename, ext.upper(), method, language, word_count])
    return report

def batch_process():
    all_reports = []
    for file in os.listdir(FOLDER_PATH):
        full_path = os.path.join(FOLDER_PATH, file)
        if os.path.isfile(full_path):
            report = process_file(full_path)
            if report:
                all_reports.extend(report)
    save_report(all_reports)

if __name__ == "__main__":
    print(f"Rozpoczynanie przetwarzania plików z folderu: {FOLDER_PATH}")
    batch_process()
    print("Zakończono. Wyniki i raport zapisano w bieżącym folderze.")
