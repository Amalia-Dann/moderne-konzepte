import json
from rag.pipeline import PDFRAG
from tests.metrics import recall_at_k, mrr

with open("tests/test_data.json", encoding="utf-8") as f:
    data = json.load(f)

rag = PDFRAG()
rag.ingest()

r, m = 0, 0
for d in data:
    question = d["question"]
    gold_chunk = d["gold_chunk"]

    # Query-Embedding
    q_emb = rag.embedder.encode([question])[0]

    # Top-20 Chunks aus der Vector-DB holen
    docs = rag.retriever.search(q_emb, k=20)

    # Sicherstellen, dass der Gold-Chunk überhaupt im Index vorkommen KANN:
    # Optional: Debug-Ausgabe, falls nie gefunden
    if gold_chunk not in docs:
        print("WARNUNG: gold_chunk nicht in Top-20 für Frage:")
        print("Frage:", question)
        # Du kannst hier auch weiterdebuggen, aber für die Metrik werten wir trotzdem aus.

    r += recall_at_k(docs, gold_chunk)
    m += mrr(docs, gold_chunk)

print("Recall@", r / len(data), "MRR", m / len(data))