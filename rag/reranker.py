# rag/reranker.py

from sentence_transformers import CrossEncoder
from config import log_line


class Reranker:
    def __init__(self, model_path: str):
        """
        Initialisiert einen CrossEncoder für die Re-Ranking-Phase.
        """
        self.model = CrossEncoder(
            model_path,
            local_files_only=True
        )
        log_line(f"[RERANK_INIT_OFFLINE] model={model_path}")

    def rerank(self, query: str, docs: list[str]) -> list[str]:
        """
        Re-rankt eine Liste von Dokument-Texten (Chunks) auf Basis einer Query.

        Parameter
        ---------
        query : str
            Die Benutzerfrage.
        docs : list[str]
            Liste der Dokument-Strings, die gerankt werden sollen.

        Rückgabe
        --------
        list[str]
            Die Dokumente, sortiert nach absteigender Relevanz.
        """
        if not docs:
            log_line("[RERANK] keine Dokumente übergeben, Rückgabe: []")
            return []

        log_line(f"[RERANK] START query={query} doc_count={len(docs)}")

        pairs = [(query, d) for d in docs]
        scores = self.model.predict(pairs)

        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        reranked_docs = [docs[i] for i in idx]

        log_lines = ["[RERANK] RESULTS:"]
        for rank, i in enumerate(idx, start=1):
            score = scores[i]
            doc_text = docs[i]
            log_lines.append(
                f"  rank={rank} index={i} score={score:.4f} "
                f"text_START\n{doc_text}\ntext_END"
            )
        log_line("\n".join(log_lines))

        log_line("[RERANK] END")
        return reranked_docs