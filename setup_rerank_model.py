# setup_rerank_model.py

from sentence_transformers import CrossEncoder

def main():
    model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    target_path = "./models/ms-marco-MiniLM-L-12-v2"

    print(f"Lade Rerank-Modell '{model_name}' ...")
    model = CrossEncoder(model_name)
    model.save(target_path)
    print(f"Rerank-Modell gespeichert unter {target_path}")

if __name__ == "__main__":
    main()