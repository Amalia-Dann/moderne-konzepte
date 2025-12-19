from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("Downloading embedding model...")
SentenceTransformer(
    "intfloat/e5-base-v2",
    cache_folder=str(MODEL_DIR / "e5-base-v2")
)

print("Downloading reranker model...")
CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    cache_folder=str(MODEL_DIR / "ms-marco-MiniLM-L-12-v2")
)

print("Models downloaded successfully.")
