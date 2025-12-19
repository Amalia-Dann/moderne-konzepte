# rag/chunker.py

import re


def _split_into_sentences(text: str) -> list[str]:
    """
    Heuristische, performante Satzsegmentierung.

    Ansatz:
    - Split nach Satzendzeichen (. ? !) gefolgt von Whitespace.
    - Funktioniert nicht perfekt (v.a. bei Abkürzungen), ist aber ausreichend
      performant und verbessert die Chunk-Kohärenz gegenüber reinem Wort-Split.
    """
    if not text:
        return []

    # Einfacher Regex-Split an Satzenden
    # Beispiel: "Das ist ein Satz. Das ist der nächste!" -> ["Das ist ein Satz.", "Das ist der nächste!"]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Leere Einträge entfernen und trimmen
    return [s.strip() for s in sentences if s.strip()]


def chunk_page(text: str, size: int, overlap: int) -> list[str]:
    """
    Erzeugt Chunks aus einer Seiten-Textpassage.

    Ziele:
    - Alle Informationen sollen im RAG wiederfindbar sein -> keine Chunks werden verworfen.
    - Chunks sollen möglichst satzkohärent sein, um die Retrieval-Qualität zu erhöhen.
    - Wortbasiertes Fenster mit Overlap, gesteuert über `size` und `overlap`.

    Parameter
    ---------
    text : str
        Der bereinigte Text einer PDF-Seite.
    size : int
        Zielgröße eines Chunks in Anzahl von Wörtern.
    overlap : int
        Anzahl der Wörter, die zwischen aufeinanderfolgenden Chunks überlappen sollen.

    Rückgabe
    --------
    list[str]
        Liste von Text-Chunks.
    """
    if not text:
        return []

    sentences = _split_into_sentences(text)
    if not sentences:
        # Fallback: wenn keine Sätze erkannt wurden, word-basiertes Chunking
        words = text.split()
        if not words:
            return []
        chunks = []
        i = 0
        step = max(1, size - overlap)
        while i < len(words):
            chunk_words = words[i:i + size]
            chunks.append(" ".join(chunk_words))
            i += step
        return chunks

    chunks: list[str] = []
    current_words: list[str] = []
    current_len = 0

    # Wir merken uns die letzten Wörter des vorherigen Chunks
    # für das Overlap-Fenster.
    last_chunk_words: list[str] = []

    for sent in sentences:
        sent_words = sent.split()
        if not sent_words:
            continue

        # Wenn der aktuelle Satz alleine größer als die Zielgröße ist,
        # splitten wir ihn word-basiert, um nichts zu verlieren.
        if len(sent_words) > size:
            # Erst aktuellen Chunk abschließen, falls er schon Inhalt hat
            if current_words:
                chunks.append(" ".join(current_words))
                last_chunk_words = current_words[-overlap:] if overlap > 0 else []
                current_words = []
                current_len = 0

            # Den langen Satz word-basiert in kleinere Chunks splitten
            i = 0
            step = max(1, size - overlap)
            while i < len(sent_words):
                chunk_words = sent_words[i:i + size]
                chunks.append(" ".join(chunk_words))
                i += step

            # letzte Overlap-Wörter für evtl. folgenden normalen Chunk
            last_chunk_words = sent_words[-overlap:] if overlap > 0 else []
            continue

        # Wenn der aktuelle Satz noch in den bestehenden Chunk passt
        if current_len + len(sent_words) <= size:
            current_words.extend(sent_words)
            current_len += len(sent_words)
        else:
            # aktuellen Chunk abschließen
            if current_words:
                chunks.append(" ".join(current_words))
                last_chunk_words = current_words[-overlap:] if overlap > 0 else []
            else:
                last_chunk_words = []

            # neuen Chunk mit Overlap starten
            current_words = []
            current_len = 0

            if last_chunk_words:
                current_words.extend(last_chunk_words)
                current_len += len(last_chunk_words)

            current_words.extend(sent_words)
            current_len += len(sent_words)

    # letzten Chunk hinzufügen, falls vorhanden
    if current_words:
        chunks.append(" ".join(current_words))

    return chunks