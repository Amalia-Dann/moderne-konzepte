# rag/gap_analyzer.py

from rag.llm import call_llm

PROMPT = """
Du bewertest, ob die vorhandenen Informationen ausreichen, um eine Frage zu beantworten.

Aufgabe:
1. Prüfe, ob die Frage mit dem gegebenen Kontext wirklich vollständig, präzise und inhaltlich korrekt beantwortet werden kann.
2. Wenn du dir NICHT sicher bist, ob alle wichtigen Informationen im Kontext enthalten sind,
   dann gehe davon aus, dass zusätzliche Informationen nötig sind.
3. Wenn die Informationen eindeutig ausreichen, gib EXAKT folgendes zurück:
   NONE
4. Wenn wichtige Informationen fehlen oder unklar sind, dann:
   - formuliere bis zu DREI kurze, präzise Suchanfragen,
   - jede Suchanfrage in einer EIGENEN ZEILE,
   - KEINE Erklärungen, KEINE Begründungen, KEINE vollständigen Sätze,
   - maximal 10 bis 12 Wörter pro Zeile,
   - KEINE Anführungszeichen, KEINE Doppelpunkte,
   - KEINE Nummerierung, KEINE Bulletpoints,
   - auf DEUTSCH,
   - fokussiere dich nur auf Schlüsselbegriffe, die für die Beantwortung der Frage fehlen.

BEISPIELE FÜR DAS FORMAT:

Frage:
Was ist die Beschlussempfehlung von TOP 4?

Gültige Ausgaben, wenn Informationen fehlen:
Beschlussempfehlung TOP 4 Senat DHBW
TOP 4 Studienschwerpunkte Soziale Arbeit Beschlussempfehlung
Beschluss Nr. 2025-04-29-3 Fachkommission Sozialwesen

Ungültige Ausgaben (NICHT machen):
- Längere Erklärungen
- Vollständige Sätze
- Bulletpoints oder Nummerierungen

Frage:
{question}

Kontext (Auszüge aus Dokumenten):
{context}

WICHTIGES OUTPUT-FORMAT:
- Wenn alles ausreichend beantwortet ist:
  NONE
- Wenn etwas Wichtiges fehlt:
  (bis zu drei Suchanfragen, jeweils in einer eigenen Zeile, ohne zusätzliche Erklärungen)
"""


def analyze_gap(question: str, contexts: list[str]) -> list[str]:
    """
    Analysiert, ob die vorhandenen Kontexte ausreichen, um die Frage zu beantworten.
    Falls nicht, werden bis zu drei zusätzliche Suchanfragen erzeugt.

    Parameter
    ---------
    question : str
        Die Benutzerfrage.
    contexts : list[str]
        Liste von Kontext-Strings (z.B. die Top-Dokument-Chunks aus dem Retrieval).

    Rückgabe
    --------
    list[str]
        - Leere Liste, wenn die vorhandenen Informationen als ausreichend angesehen werden.
        - Sonst: bis zu drei Suchanfragen (Deutsch), jeweils ein String pro Anfrage.
    """
    # Wir nehmen nur die ersten 5 Kontexte, um den Prompt kompakt zu halten.
    ctx = "\n---\n".join(contexts[:5])

    out = call_llm(
        PROMPT.format(question=question, context=ctx),
        tag="GAP_ANALYSIS"
    )

    # Falls das Modell sich korrekt an die Instruktion hält:
    if out.strip().upper() == "NONE":
        return []

    # Allgemeines Parsing:
    # - Zeilenweise aufsplitten
    # - Leere Zeilen entfernen
    # - Potenzielle Bullet-Zeichen und Nummerierungen am Anfang entfernen
    candidates: list[str] = []
    for line in out.splitlines():
        raw = line.strip()
        if not raw:
            continue

        cleaned = raw

        # Häufige Bullet-/Nummerierungs-Patterns entfernen
        for prefix in ("- ", "* ", "• ", "· "):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()

        # Einfache Nummerierungen wie "1. " oder "2) "
        if len(cleaned) > 2 and cleaned[0].isdigit():
            if cleaned[1:3] in (". ", ") "):
                cleaned = cleaned[3:].strip()

        if cleaned:
            candidates.append(cleaned)

    # Postprocessing: zu lange/erklärende Zeilen kürzen und normalisieren
    normalized: list[str] = []
    for c in candidates:
        # Nur die erste "Satzhälfte" vor Punkt/Fragezeichen/Ausrufezeichen nehmen
        for sep in [".", "?", "!"]:
            if sep in c:
                c = c.split(sep)[0].strip()

        # Auf max. 12 Wörter begrenzen
        words = c.split()
        if len(words) > 12:
            c = " ".join(words[:12])
            words = c.split()

        # Zeilen mit sehr wenig Informationsgehalt wegwerfen
        if len(words) < 2:
            continue

        normalized.append(c)

    # Maximal 3 Suchanfragen zurückgeben
    return normalized[:3]