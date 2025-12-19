# rag/pdf_reader.py

import re

import fitz
from config import log_line


def _clean_page_text(text: str) -> str:
    """
    Führt eine performante Standard-Preprocessing-Pipeline für PDF-Text durch.

    Schritte:
    - Normalisierung von Zeilenumbrüchen.
    - Entfernen von häufigen PDF-Artefakten (übermäßige Leerzeichen).
    - Auflösen von Silbentrennungen über Zeilenumbrüche: "Bei-\nspiel" -> "Beispiel".
    - Ersetzen von verbleibenden Zeilenumbrüchen durch Leerzeichen.
    - Reduzieren mehrfacher Leerzeichen auf ein einzelnes.

    Hinweis:
    - Es werden bewusst keine heuristischen Header/Footer-Entfernungen vorgenommen,
      um keine Informationen zu verlieren.
    """

    if not text:
        return ""

    # \r -> \n normalisieren
    text = text.replace("\r", "\n")

    # Silbentrennungen über Zeilenumbrüche:
    # "Bei-\nspiel" -> "Beispiel"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Verbleibende Newlines zu Leerzeichen
    text = text.replace("\n", " ")

    # Mehrfache Leerzeichen reduzieren
    text = re.sub(r"\s+", " ", text)

    # Trim
    return text.strip()


def extract_pages(path: str):
    """
    Extrahiert Text pro Seite aus einem PDF und bereitet ihn grundlegend auf.

    Rückgabe:
    ---------
    list[tuple[int, str]]:
        Liste von (seiten_nummer, bereinigter_text).
    """
    doc = fitz.open(path)
    pages = []

    for i, page in enumerate(doc):
        raw_text = page.get_text("text") or ""
        cleaned_text = _clean_page_text(raw_text)
        pages.append((i + 1, cleaned_text))
        log_line(f"[PDF] {path} page={i + 1} raw_chars={len(raw_text)} cleaned_chars={len(cleaned_text)}")

    return pages