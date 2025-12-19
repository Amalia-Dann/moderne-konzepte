# tests/test_gap_query_console.py

from config import set_global_seed, RAG_MODE, ENABLE_GAP_RETRIEVAL
from rag.pipeline import PDFRAG


def main():
    # Reproduzierbarkeit
    set_global_seed()

    print("=== GAP-ANALYSE QUERY TEST ===")
    print(f"Aktueller RAG_MODE: {RAG_MODE}")
    print(f"ENABLE_GAP_RETRIEVAL: {ENABLE_GAP_RETRIEVAL}")
    print()

    print("Hinweis:")
    print("  Für einen vollen Gap-Analyse-Test sollten idealerweise")
    print("  RAG_MODE = 'enhanced' und ENABLE_GAP_RETRIEVAL = True gesetzt sein.")
    print("  (Anpassbar in config.py)")
    print()

    rag = PDFRAG()

    # Ingestion starten (falls noch nicht erfolgt)
    print("Starte Ingestion (Einlesen & Embedding der PDFs)...")
    rag.ingest()
    print("Ingestion abgeschlossen.")
    print()

    question = input("Frage (enhanced mit Gap-Analyse): ")
    print()
    print("Sende Anfrage an RAG-System mit Gap-Analyse...\n")

    answer = rag.query(question)

    print("=== ANTWORT (ENHANCED / GAP) ===")
    print(answer)
    print("=================================")


if __name__ == "__main__":
    main()