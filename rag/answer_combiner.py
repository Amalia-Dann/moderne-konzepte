# rag/answer_combiner.py

from rag.llm import call_llm

PROMPT = """
Du erhältst eine Benutzerfrage und mehrere Textausschnitte aus PDF-Dokumenten.

Aufgabe:
- Beantworte die Frage so gut wie möglich NUR auf Basis der bereitgestellten Informationen.
- Erfinde KEINE Fakten, die im Widerspruch zu den Texten stehen.
- Antworte so kurz wie möglich, maximal ZWEI Sätze.
- Wenn die Frage nach einer Beschlussempfehlung, einem TOP oder einem konkreten Beschluss fragt,
  gib nach Möglichkeit GENAU den entsprechenden Satz oder die entsprechende Zeile aus dem Text wieder,
  ohne zusätzliche Erläuterungen oder Aufzählungen.
- Wenn wirklich entscheidende Informationen fehlen, um die Frage korrekt zu beantworten,
  dann sage das explizit und formuliere eine möglichst knappe Antwort,
  die klar macht, welche Informationen fehlen.

Frage:
{question}

Informationen (Ausschnitte aus den Dokumenten):
{info}

Antwort (kurz, maximal zwei Sätze):
"""

PROMPT_COLLECT = """
Du erhältst eine Benutzerfrage und mehrere Textausschnitte aus PDF-Dokumenten.

Aufgabe:
- Identifiziere ALLE Textstellen, die für die Beantwortung der Frage relevant sein könnten.
- Gib NUR die relevanten Textstellen zurück, jeweils mit einer kurzen Überschrift.
- Wenn mehrere Stellen ähnliche Informationen enthalten, fasse sie sinnvoll zusammen.
- Antworte nur mit den relevanten Ausschnitten, keine zusätzlichen Erklärungen.

Frage:
{question}

Informationen (Ausschnitte aus den Dokumenten):
{info}

Gib jetzt ALLE relevanten Textstellen zurück:
"""

PROMPT_CHOOSE = """
Du erhältst eine Benutzerfrage und eine Sammlung relevanter Textstellen aus PDF-Dokumenten.

Aufgabe:
- Beantworte die Frage so gut wie möglich NUR auf Basis dieser Textstellen.
- Wenn mehrere Textstellen unterschiedliche Beschlussempfehlungen oder Entscheidungen betreffen,
  wähle diejenige, die am besten zur Frage passt (z.B. zum richtigen TOP oder Thema).
- Sei kurz, präzise und sachlich.
- Erfinde keine Informationen, die im Widerspruch zu den Textstellen stehen.

Frage:
{question}

Relevante Textstellen:
{snippets}

Antwort (kurz und präzise):
"""

def combine(question: str, chunks: list[str]) -> str:
    """
    Kombiniert mehrere Text-Chunks zu einer finalen, knappen Antwort
    mit Hilfe des LLMs.

    Parameter
    ---------
    question : str
        Die Benutzerfrage.
    chunks : list[str]
        Liste von Text-Chunks (z.B. aus Retrieval + zweitem Retrieval-Pass).

    Rückgabe
    --------
    str
        Eine kurze, präzise Antwort auf Deutsch, die sich nur auf die
        gegebenen Informationen stützt, oder ein expliziter Hinweis darauf,
        dass die Informationen nicht ausreichen.
    """
    # Aus Performance-Gründen verwenden wir eine begrenzte Anzahl von Chunks.
    # Falls du später mehr Kontext zulassen möchtest, kann dieser Wert erhöht
    # oder konfigurierbar gemacht werden.
    max_chunks = 15
    selected_chunks = chunks[:max_chunks]

    info = "\n---\n".join(selected_chunks)

    return call_llm(
        PROMPT.format(question=question, info=info),
        tag="ANSWER_COMBINE"
    )

def collect_relevant_snippets(question: str, chunks: list[str]) -> str:
    """
    Lässt das LLM alle potentiell relevanten Textstellen aus den Chunks sammeln.
    Gibt einen großen String mit nur relevanten Ausschnitten zurück.
    """
    max_chunks = 30  # hier bewusst größer, damit wir mehr Kontext anbieten
    selected_chunks = chunks[:max_chunks]
    info = "\n---\n".join(selected_chunks)

    return call_llm(
        PROMPT_COLLECT.format(question=question, info=info),
        tag="ANSWER_COLLECT"
    )

def choose_best_answer(question: str, snippets: str) -> str:
    """
    Lässt das LLM aus den gesammelten relevanten Textstellen
    die bestpassende Antwort generieren.
    """
    return call_llm(
        PROMPT_CHOOSE.format(question=question, snippets=snippets),
        tag="ANSWER_CHOOSE"
    )

def is_not_found_answer(answer: str) -> bool:
    """
    Heuristik: erkennt Antworten, die ausdrücken, dass
    die nötigen Informationen im Kontext fehlen.
    """
    if not answer:
        return True

    patterns = [
        "leider fehlen mir entscheidende informationen",
        "die informationen reichen nicht aus",
        "aus den bereitgestellten texten nicht beantworten",
        "liegen mir keine ausreichenden informationen vor",
        "im gegebenen kontext nicht enthalten",
    ]

    ans_lower = answer.lower()
    return any(p in ans_lower for p in patterns)