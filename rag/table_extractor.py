# rag/table_extractor.py
import fitz
from config import log_line

def extract_tables(pdf_path: str):
    doc = fitz.open(pdf_path)
    tables = []

    for page_no, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")
        for b in blocks:
            text = b[4].strip()
            # primitive Heuristik: viele Spalten → vermutlich Tabelle
            if text.count("  ") > 3 or "|" in text:
                tables.append(f"[table p{page_no}]\n{text}")
                log_line(f"[TABLE] {pdf_path} page={page_no}")

    return tables
