import pdfplumber
import pytesseract
from pdf2image import convert_from_path

def extract_text_native(pdf_path: str):
    text_chunk = []
    with pdfplumber.open(pdf.path) as pdf :
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunk.append(page_text)
    return "\n".join(text_chunk)

def extract_text_ocr(pdf_path: str, dpi: int = 300):
    text_chunk = []
    pages = convert_from_path(pdf_path, dpi=dpi)
    for i, page_image in enumerate(pages):
        page_text = pytesseract.image_to_string(page_image)
        text_chunk.append(page_image)
        print(f"OCR'd page {i+1}/{len(pages)}")
    return "\n".join(text_chunk)

def ingest_pdf(pdf_path: str, min_chars_for_native: int = 50):
    print("Ingesting : ", pdf_path)
    native_text = extract_text_native(pdf_path)

    if len(native_text.strip()) >= min_chars_for_native:
        print("Native text layer found ")
        method = "native"
        final_text = native_text
    else:
        print("No usable text layer, falling back to ocr")
        method = "ocr"
        final_text = extract_text_ocr(pdf_path)

    return {
        "contract_id": pdf_path.split("/")[-1],
        "title": pdf_path.split("/")[-1],
        "text": final_text,
        "extraction_method": method,
        "clauses": []
    }

if __name__ == "__main__":
    print("Module ready to use ")

