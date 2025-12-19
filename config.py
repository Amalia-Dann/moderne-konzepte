# config.py

from datetime import datetime
import os
import uuid
import random

import numpy as np
import torch

# ===== OFFLINE ENFORCEMENT =====
#s.environ["HF_HUB_OFFLINE"] = "1"
#os.environ["TRANSFORMERS_OFFLINE"] = "1"
#os.environ["HF_DATASETS_OFFLINE"] = "1"

# ===== RUN IDENTIFICATION =====
# Eindeutige ID pro Programmlauf (nützlich für spätere Auswertung)
RUN_ID = str(uuid.uuid4())

# ===== REPRODUCIBILITY / SEED =====
# Fester Seed für Reproduzierbarkeit von Experimenten
RANDOM_SEED = 42


def set_global_seed(seed: int = RANDOM_SEED):
    """
    Setzt globale Zufalls-Seeds für Python, NumPy und PyTorch,
    um Experimente möglichst reproduzierbar zu machen.
    Diese Funktion sollte zu Beginn des Programms aufgerufen werden.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Für zusätzliche Deterministik (kann etwas Performance kosten):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Falls Torch in einer Umgebung nicht verfügbar oder eingeschränkt ist,
        # soll das nicht den gesamten Run blockieren.
        pass


# ===== MODELS =====
EMBED_MODEL = "qwen3-embedding:0.6b"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# ===== PATHS =====
PDF_DIR = "./pdfs"
DB_PATH = "./vector_db"

# ===== CHUNKING =====
CHUNK_SIZE = 120
CHUNK_OVERLAP = 50

# ===== RETRIEVAL =====
TOP_K = 20
RERANK_TOP_N = 10

# ===== RAG MODES =====
RAG_MODE = "enhanced"        # "simple" | "enhanced"
ENABLE_GAP_RETRIEVAL = True


def set_rag_mode(mode: str, enable_gap: bool):
    """
    Erlaubt es, RAG_MODE und ENABLE_GAP_RETRIEVAL zur Laufzeit
    im selben Python-Prozess umzuschalten (z.B. für Evaluation).

    Beispiel:
        set_rag_mode("simple", False)
        set_rag_mode("enhanced", True)
    """
    global RAG_MODE, ENABLE_GAP_RETRIEVAL
    RAG_MODE = mode
    ENABLE_GAP_RETRIEVAL = enable_gap


# ===== OLLAMA =====
OLLAMA_MODEL = "llama3.2"
OLLAMA_TEMPERATURE = 0.25

# ===== LOGGING =====
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Genau eine Logdatei pro Lauf (über Timestamp im Dateinamen)
LOG_FILE = os.path.join(
    LOG_DIR,
    f"pdf_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)


def log_line(msg: str):
    """
    Schreibt eine einzelne Logzeile in die aktuelle Logdatei.
    Keine Truncation – der aufrufende Code entscheidet selbst,
    wie viel Inhalt geloggt wird.
    """
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"[RUN={RUN_ID}] {msg}\n"
        )
