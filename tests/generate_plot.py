import json
import matplotlib.pyplot as plt


def generate_scientific_plot(filepath):
    # 1. Daten laden
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Fehler: {filepath} nicht gefunden.")
        return

    # 2. Daten verarbeiten
    simple_scores = {}
    enhanced_scores = {}
    for entry in data:
        idx = int(entry['id'])
        if entry['mode'] == 'simple':
            simple_scores[idx] = entry['bertscore_f1']
        elif entry['mode'] == 'enhanced':
            enhanced_scores[idx] = entry['bertscore_f1']

    ids = sorted(list(set(simple_scores.keys()) | set(enhanced_scores.keys())))
    y_simple = [simple_scores.get(i, 0) for i in ids]
    y_enhanced = [enhanced_scores.get(i, 0) for i in ids]

    # 3. Plot-Design für wissenschaftliche Arbeiten
    # Wir nutzen 'serif' Schriftarten für eine bessere Einbettung in Dokumente
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11

    fig, ax = plt.subplots(figsize=(10, 5))

    # Linien mit unterschiedlichen Markern und Stilen für Schwarz-Weiß-Druck-Kompatibilität
    ax.plot(ids, y_simple, label='Baseline (Simple)', color='#1f77b4',
            marker='o', markersize=6, linestyle='--', linewidth=1.5)
    ax.plot(ids, y_enhanced, label='Enhanced (RAG/Optimized)', color='#d62728',
            marker='s', markersize=6, linestyle='-', linewidth=1.5)

    # Beschriftungen (Deutsch oder Englisch je nach Arbeit anpassen)
    ax.set_title('Evaluierung der Antwortqualität (BERTScore F1)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Testfall-ID (Question ID)', fontsize=12)
    ax.set_ylabel('BERTScore F1-Wert', fontsize=12)

    # Achsen-Setup
    ax.set_xticks(ids)
    ax.set_ylim(0, 1.0)  # BERTScore liegt immer zwischen 0 und 1
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', frameon=True)

    # Layout optimieren
    plt.tight_layout()

    # 4. Speichern in verschiedenen Formaten
    # PDF ist ideal für LaTeX (Vektorgrafik, kein Qualitätsverlust beim Zoomen)
    plt.savefig('evaluation_results_plot.pdf', bbox_inches='tight')
    # PNG mit 300 DPI ist ideal für Word/PowerPoint
    plt.savefig('evaluation_results_plot.png', dpi=300, bbox_inches='tight')

    print("Grafiken wurden als 'evaluation_results_plot.pdf' und '.png' gespeichert.")
    plt.show()


if __name__ == "__main__":
    generate_scientific_plot('eval_results.json')