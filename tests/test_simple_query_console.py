# tests/test_simple_query_console.py

from config import set_global_seed, RAG_MODE, ENABLE_GAP_RETRIEVAL
from rag.pipeline import PDFRAG


def main():
    # Reproduzierbarkeit
    set_global_seed()

    print("=== SIMPLE QUERY TEST ===")
    print(f"Aktueller RAG_MODE: {RAG_MODE}")
    print(f"ENABLE_GAP_RETRIEVAL: {ENABLE_GAP_RETRIEVAL}")
    print()

    print("Hinweis:")
    print("  Für einen 'reinen' einfachen Test sollten idealerweise")
    print("  RAG_MODE = 'simple' oder ENABLE_GAP_RETRIEVAL = False gesetzt sein.")
    print("  (Anpassbar in config.py)")
    print()

    rag = PDFRAG()

    # Ingestion starten (falls noch nicht erfolgt)
    print("Starte Ingestion (Einlesen & Embedding der PDFs)...")
    rag.ingest()
    print("Ingestion abgeschlossen.")
    print()

    question = input("Frage (simple mode): ")
    print()
    print("Sende Anfrage an RAG-System...\n")

    answer = rag.query(question)

    print("=== ANTWORT (SIMPLE) ===")
    print(answer)
    print("=========================")


if __name__ == "__main__":
    main()