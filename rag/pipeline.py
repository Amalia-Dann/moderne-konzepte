# rag/pipeline.py

from pathlib import Path

from rag.pdf_reader import extract_pages
from rag.table_extractor import extract_tables
from rag.chunker import chunk_page
from rag.embeddings import Embedder
from rag.retriever import Retriever
from rag.reranker import Reranker
from rag.gap_analyzer import analyze_gap
from rag.answer_combiner import (
    combine,
    is_not_found_answer,
    collect_relevant_snippets,
    choose_best_answer,
)
from config import (
    PDF_DIR,
    DB_PATH,
    EMBED_MODEL,
    RERANK_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    RERANK_TOP_N,
    RAG_MODE,
    ENABLE_GAP_RETRIEVAL,
    log_line,
)


class PDFRAG:
    def __init__(self):
        """
        Initialisiert Embedder, Retriever und Reranker.
        """
        log_line("[PIPELINE] Initialisiere PDFRAG-Komponenten")
        self.embedder = Embedder(EMBED_MODEL)
        self.retriever = Retriever(DB_PATH)
        self.reranker = Reranker(RERANK_MODEL)

    def ingest(self):
        """
        Liest alle PDFs aus PDF_DIR ein, extrahiert Text und Tabellen,
        chunked sie, erzeugt Embeddings und speichert alles im Vector-Store.
        """
        log_line(f"[PIPELINE] Starte Ingestion aus Verzeichnis: {PDF_DIR}")

        docs: list[str] = []

        # Debug: Welche Einträge sieht Python im PDF_DIR?
        log_line(f"[PIPELINE] Ingestion: Liste Dateien in {PDF_DIR}")
        for entry in Path(PDF_DIR).iterdir():
            log_line(
                f"[PIPELINE] Ingestion: gefundenes Entry: {entry} "
                f"(is_file={entry.is_file()}, suffix={entry.suffix})"
            )

        # Nur echte Dateien mit .pdf (case-insensitive) verarbeiten
        for pdf in Path(PDF_DIR).iterdir():
            if not pdf.is_file():
                continue
            if pdf.suffix.lower() != ".pdf":
                continue

            pdf_path = str(pdf)
            pdf_name = pdf.name
            log_line(f"[PIPELINE] Verarbeite PDF: {pdf_path}")

            # Text-Seiten extrahieren
            pages = extract_pages(pdf_path)

            # Tabellen extrahieren
            tables = extract_tables(pdf_path)

            # Seiten chunking
            for pno, text in pages:
                for c in chunk_page(text, CHUNK_SIZE, CHUNK_OVERLAP):
                    # Page-Information im Text belassen (wie bisher)
                    docs.append(f"[file {pdf_name}] [page {pno}] {c}")

            # Tabellen als eigenständige Chunks
            for t in tables:
                docs.append(f"[file {pdf_name}] [table]\n{t}")

        if not docs:
            log_line("[PIPELINE] WARNUNG: Keine Dokumente gefunden, Ingestion beendet.")
            return

        # Embeddings berechnen
        embs = self.embedder.encode(docs)

        # Einfache IDs auf Basis des Hashes des Textes
        ids = [str(hash(d)) for d in docs]

        # In Vector-DB speichern
        self.retriever.add(ids, docs, embs)

        log_line(f"[PIPELINE] Ingestion abgeschlossen. Dokumente: {len(docs)}")

    def query(self, question: str) -> str:
        """
        Beantwortet eine Frage auf Basis der indizierten PDFs.

        Ablauf:
        1. Embedding der Frage
        2. Erster Retrieval-Pass (Vector-DB)
        3. Reranking
        4. (Optional) Gap-Analyse mit LLM
        5. (Optional) Zweiter Retrieval-Pass auf Basis der Gap-Queries
        6. Kombination aller relevanten Chunks zu einer finalen Antwort (LLM)

        Rückgabe:
        ---------
        str: Finale Antwort auf Deutsch.
        """
        log_line(f"[PIPELINE] QUERY_START Frage: {question}")

        # ===== 1) Embedding der Frage =====
        qemb = self.embedder.encode([question])[0]

        # ===== 2) Erster Retrieval-Pass =====
        first_docs = self.retriever.search(qemb, TOP_K)
        log_line(
            "[PIPELINE] FIRST_RETRIEVAL Ergebnisse START\n"
            + "\n---\n".join(first_docs)
            + "\n[PIPELINE] FIRST_RETRIEVAL Ergebnisse ENDE"
        )

        # ===== 3) Reranking =====
        reranked_docs = self.reranker.rerank(question, first_docs[:RERANK_TOP_N])
        log_line(
            "[PIPELINE] RERANKED Ergebnisse START\n"
            + "\n---\n".join(reranked_docs)
            + "\n[PIPELINE] RERANKED Ergebnisse ENDE"
        )

        # ===== 4) Einfacher Modus oder Gap-Analyse deaktiviert =====
        if RAG_MODE == "simple" or not ENABLE_GAP_RETRIEVAL:
            log_line(
                f"[PIPELINE] SIMPLE_MODE oder GAP_ANALYSE deaktiviert "
                f"(RAG_MODE={RAG_MODE}, ENABLE_GAP_RETRIEVAL={ENABLE_GAP_RETRIEVAL})"
            )
            answer = combine(question, reranked_docs)
            log_line("[PIPELINE] QUERY_END (simple / no-gap)")
            return answer

        # ===== 5) Gap-Analyse =====
        gap_queries = analyze_gap(question, reranked_docs)
        if not gap_queries:
            log_line("[PIPELINE] GAP_ANALYSE: NONE -> nutze Originalfrage als zusätzliche Gap-Query.")
            gap_queries = [question]
        log_line(
            "[PIPELINE] GAP_ANALYSE: Zusätzliche Suchanfragen START\n"
            + "\n".join(gap_queries)
            + "\n[PIPELINE] GAP_ANALYSE: Zusätzliche Suchanfragen ENDE"
        )

        # ===== 6) Zweiter Retrieval-Pass auf Basis der Gap-Queries =====
        extra_docs: list[str] = []

        for nq in gap_queries:
            log_line(f"[PIPELINE] SECOND_RETRIEVAL für Gap-Query: {nq}")
            nq_emb = self.embedder.encode([nq])[0]
            hits = self.retriever.search(nq_emb, TOP_K)

            log_line(
                "[PIPELINE] SECOND_RETRIEVAL Ergebnisse START\n"
                + "\n---\n".join(hits)
                + "\n[PIPELINE] SECOND_RETRIEVAL Ergebnisse ENDE"
            )

            extra_docs.extend(hits)

        # Kombinieren der Chunks aus erstem und zweitem Retrieval-Pass
        combined_docs = reranked_docs + extra_docs

        # Duplikate entfernen, Reihenfolge beibehalten
        seen = set()
        unique_docs: list[str] = []
        for d in combined_docs:
            if d not in seen:
                seen.add(d)
                unique_docs.append(d)

        log_line(
            "[PIPELINE] COMBINED_CONTEXT START\n"
            + "\n---\n".join(unique_docs)
            + "\n[PIPELINE] COMBINED_CONTEXT ENDE"
        )

        # ===== 7) Finale Antwort-Kombination (erster Versuch) =====
        answer = combine(question, unique_docs)
        log_line("[PIPELINE] FIRST_ANSWER")
        log_line(answer)

        # ===== 8) Fail-Safe NUR im enhanced: wenn Antwort sagt, dass Infos fehlen =====
        if is_not_found_answer(answer):
            log_line("[PIPELINE] FAILSAFE_TRIGGER: Antwort meldet fehlende Informationen. Starte Sammel-Pass.")

            snippets = collect_relevant_snippets(question, unique_docs)
            log_line("[PIPELINE] COLLECTED_SNIPPETS_START")
            log_line(snippets)
            log_line("[PIPELINE] COLLECTED_SNIPPETS_END")

            improved_answer = choose_best_answer(question, snippets)
            log_line("[PIPELINE] IMPROVED_ANSWER")
            log_line(improved_answer)

            if not is_not_found_answer(improved_answer):
                log_line("[PIPELINE] QUERY_END (enhanced, mit Fail-Safe-Verbesserung)")
                return improved_answer
            else:
                log_line("[PIPELINE] FAILSAFE_NO_IMPROVEMENT, gebe ursprüngliche Antwort zurück.")
                log_line("[PIPELINE] QUERY_END (enhanced, ohne Verbesserung)")
                return answer

        log_line("[PIPELINE] QUERY_END (enhanced, ohne Fail-Safe)")
        return answer

