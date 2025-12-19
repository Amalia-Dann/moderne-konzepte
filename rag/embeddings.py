# rag/embeddings.py

import numpy as np
import ollama

from config import log_line


class Embedder:
    def __init__(self, model_name: str):
        """
        Embedder auf Basis eines Ollama-Embedding-Modells
        (z.B. "qwen3-embedding:0.6b").
        """
        self.model_name = model_name
        log_line(f"[EMBED_INIT_OLLAMA] model={model_name}")

    def encode(self, texts):
        """
        Berechnet Embeddings für eine Liste von Texten über Ollama.

        Verhalten:
        - Ruft für jeden Text `ollama.embeddings(model=self.model_name, prompt=...)` auf.
        - Normalisiert die Embeddings (L2-Norm), damit die Kosinus-Ähnlichkeit
          gut mit Chroma funktioniert.

        Parameter
        ---------
        texts : list[str]
            Liste von Texten (Chunks oder Query).

        Rückgabe
        --------
        np.ndarray
            Array der Form (n_texts, dim) mit Float32-Embeddings.
        """
        if not texts:
            log_line("[EMBED_OLLAMA] encode aufgerufen mit leerer Textliste")
            return np.zeros((0, 0), dtype=np.float32)

        embeddings = []
        for idx, t in enumerate(texts):
            res = ollama.embeddings(model=self.model_name, prompt=t)
            emb = np.array(res["embedding"], dtype=np.float32)
            embeddings.append(emb)

        embs = np.vstack(embeddings)

        # L2-Normalisierung (wie vorher mit normalize_embeddings=True)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms

        log_line(
            f"[EMBED_OLLAMA] model={self.model_name} "
            f"items={len(texts)} dim={embs.shape[1]}"
        )

        return embs