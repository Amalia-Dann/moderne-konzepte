# rag/llm.py

import uuid

import ollama
from config import log_line, OLLAMA_MODEL, OLLAMA_TEMPERATURE


def call_llm(prompt: str, tag: str = "GENERIC") -> str:
    """
    Führt einen LLM-Call über Ollama aus und loggt dabei
    den vollständigen Prompt und die vollständige Antwort.

    Parameter
    ---------
    prompt : str
        Der vollständige Prompt, der an das Modell geschickt wird.
    tag : str, optional
        Ein kurzer Tag zur Kennzeichnung des Aufrufs im Log
        (z.B. "GAP_ANALYSIS", "ANSWER_COMBINE"). Default: "GENERIC".

    Rückgabe
    --------
    str
        Die vom Modell generierte Antwort (Content-Feld).
    """
    qid = str(uuid.uuid4())

    # Vollständigen Prompt loggen
    log_line(
        f"[LLM_CALL] [QID={qid}] [TAG={tag}] PROMPT_START\n"
        f"{prompt}\n"
        f"PROMPT_END"
    )

    # LLM-Aufruf
    res = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": OLLAMA_TEMPERATURE}
    )

    out = (res.get("message", {}).get("content") or "").strip()

    # Vollständige Antwort loggen
    log_line(
        f"[LLM_RESP] [QID={qid}] [TAG={tag}] RESP_START\n"
        f"{out}\n"
        f"RESP_END"
    )

    return out