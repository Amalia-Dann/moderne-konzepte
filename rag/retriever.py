# rag/retriever.py

import chromadb
from config import log_line


class Retriever:
    def __init__(self, path: str):
        """
        Initialisiert einen persistenten Chroma-Client und eine Collection
        für PDF-Dokumente.
        """
        self.client = chromadb.PersistentClient(path=path)
        # Name der Collection: "pdf" (konstant)
        self.col = self.client.get_or_create_collection("pdf")
        log_line(f"[VDB] init path={path}, collection=pdf")

    def add(self, ids, docs, embs):
        """
        Fügt Dokumente samt Embeddings in die Vektor-Datenbank ein.

        Parameter
        ---------
        ids : list[str]
            Eindeutige IDs für jedes Dokument / jeden Chunk.
        docs : list[str]
            Die eigentlichen Textinhalte (Chunks).
        embs : list[list[float]] oder np.ndarray
            Embeddings zu den Dokumenten.
        """
        n = len(docs)
        log_line(f"[VDB] add START count={n}")
        self.col.add(ids=ids, documents=docs, embeddings=embs)
        # Hinweis: PersistentClient speichert automatisch, kein persist() mehr nötig
        log_line(f"[VDB] add DONE count={n}")

    def search(self, emb, k: int):
        """
        Führt eine Ähnlichkeitssuche in der Vektor-Datenbank durch.

        Parameter
        ---------
        emb : list[float] oder np.ndarray
            Embedding der Query.
        k : int
            Anzahl der gewünschten Top-Ergebnisse.

        Rückgabe
        --------
        list[str]
            Liste der gefundenen Dokument-Texte (Chunks), sortiert nach Relevanz.
        """
        log_line(f"[VDB] search START k={k}")

        res = self.col.query(query_embeddings=[emb], n_results=k)

        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0] if "distances" in res else []

        # Vollständiges Logging der Treffer (IDs, Distanzen, Texte)
        log_lines = ["[VDB] search RESULTS:"]
        for i, d_id in enumerate(ids):
            dist_str = f"{dists[i]:.4f}" if i < len(dists) else "n/a"
            doc_text = docs[i] if i < len(docs) else ""
            log_lines.append(
                f"  rank={i+1} id={d_id} distance={dist_str} "
                f"text_START\n{doc_text}\ntext_END"
            )

        log_line("\n".join(log_lines))
        log_line("[VDB] search END")

        return docs