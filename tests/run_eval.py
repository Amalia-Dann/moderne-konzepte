# tests/run_eval.py

import json
import os

from bert_score import score as bertscore

from config import (
    set_global_seed,
    set_rag_mode,
    RAG_MODE,
    ENABLE_GAP_RETRIEVAL,
    log_line,
)
from rag.pipeline import PDFRAG


EVAL_DATA_PATH = "tests/eval_data.json"
EVAL_RESULTS_PATH = "tests/eval_results.json"


def load_eval_data(path: str = EVAL_DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_rag_for_mode(mode: str, enable_gap: bool):
    """
    Führt den RAG für alle Fragen im angegebenen Modus aus.
    Gibt eine Liste von Dicts zurück: id, mode, question, gold, answer.
    """
    set_rag_mode(mode, enable_gap)
    log_line(f"[EVAL] Starte Run für mode={mode}, enable_gap={enable_gap}")

    # Für Reproduzierbarkeit
    set_global_seed()

    rag = PDFRAG()

    data = load_eval_data()
    results = []

    for item in data:
        qid = item["id"]
        question = item["question"]
        gold = item["gold"]

        log_line(f"[EVAL] QUERY id={qid} mode={mode} question={question}")
        answer = rag.query(question)

        results.append(
            {
                "id": qid,
                "mode": mode,
                "question": question,
                "gold": gold,
                "answer": answer,
            }
        )

    return results


def add_bertscore(results):
    """
    Berechnet BERTScore(F1) für alle Einträge in 'results' und
    fügt pro Eintrag ein Feld 'bertscore_f1' hinzu.
    """
    golds = [r["gold"] for r in results]
    preds = [r["answer"] for r in results]

    # Deutsch
    P, R, F1 = bertscore(
        preds,
        golds,
        lang="de",
        rescale_with_baseline=True,
    )

    for r, f1 in zip(results, F1):
        r["bertscore_f1"] = float(f1.item())


def main():
    # Optional: Ingestion einmal hier machen.
    # Wenn deine Vector-DB schon existiert, kannst du das auskommentieren.
    set_global_seed()
    rag_ingest = PDFRAG()
    rag_ingest.ingest()

    # Run A: simple (ohne Gap-Analyse)
    results_simple = run_rag_for_mode(mode="simple", enable_gap=False)

    # Run B: enhanced (mit Gap-Analyse)
    results_enhanced = run_rag_for_mode(mode="enhanced", enable_gap=True)

    # Ergebnisse zusammenführen
    all_results = results_simple + results_enhanced

    # BERTScore berechnen
    add_bertscore(all_results)

    # Durchschnitt pro Modus ausgeben
    simple_scores = [r["bertscore_f1"] for r in all_results if r["mode"] == "simple"]
    enhanced_scores = [r["bertscore_f1"] for r in all_results if r["mode"] == "enhanced"]

    if simple_scores:
        mean_simple = sum(simple_scores) / len(simple_scores)
    else:
        mean_simple = 0.0

    if enhanced_scores:
        mean_enhanced = sum(enhanced_scores) / len(enhanced_scores)
    else:
        mean_enhanced = 0.0

    print("=== EVAL RESULTATE ===")
    print(f"BERTScore F1 (simple):   {mean_simple:.4f}")
    print(f"BERTScore F1 (enhanced): {mean_enhanced:.4f}")
    print(f"Detailergebnisse in: {EVAL_RESULTS_PATH}")

    # Ergebnisse als JSON speichern
    os.makedirs(os.path.dirname(EVAL_RESULTS_PATH), exist_ok=True)
    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()