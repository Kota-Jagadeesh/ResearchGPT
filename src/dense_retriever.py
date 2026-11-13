import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==========================================================
# Folder
# ==========================================================

VECTOR_FOLDER = "vector_db"

INDEX_PATH = os.path.join(
    VECTOR_FOLDER,
    "research_index.faiss"
)

METADATA_PATH = os.path.join(
    VECTOR_FOLDER,
    "metadata.pkl"
)

EMBEDDINGS_PATH = os.path.join(
    VECTOR_FOLDER,
    "embeddings.pkl"
)

# ==========================================================
# Lazy-Loaded Resources
# ==========================================================
# These used to load at import time, which crashed the whole
# Streamlit app on startup (before any papers were processed,
# since app.py imports agmr_rerank -> retrieve before the UI
# even renders). Now they only load the first time retrieve()
# is actually called.

_model = None
_index = None
_metadata = None
_embeddings = None


def _load_resources():

    global _model, _index, _metadata, _embeddings

    if _model is None:

        print("=" * 60)
        print("Loading Sentence-BERT...")
        print("=" * 60)

        _model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Sentence-BERT Loaded Successfully.\n")

    if _index is None:

        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found: {INDEX_PATH}. Please process papers first."
            )

        _index = faiss.read_index(INDEX_PATH)

    if _metadata is None:

        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(
                f"Metadata file not found: {METADATA_PATH}. Please process papers first."
            )

        with open(METADATA_PATH, "rb") as file:
            _metadata = pickle.load(file)

    if _embeddings is None:

        if not os.path.exists(EMBEDDINGS_PATH):
            raise FileNotFoundError(
                f"Embeddings file not found: {EMBEDDINGS_PATH}. Please process papers first."
            )

        with open(EMBEDDINGS_PATH, "rb") as file:
            _embeddings = pickle.load(file)

        _embeddings = np.asarray(
            _embeddings,
            dtype=np.float32
        )


# ==========================================================
# Retrieve Function
# ==========================================================

def retrieve(question, top_k=100):

    if not question.strip():
        return []

    _load_resources()

    top_k = min(top_k, _index.ntotal)

    question_embedding = _model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    distances, indices = _index.search(
        question_embedding,
        top_k
    )

    results = []

    for distance, idx in zip(distances[0], indices[0]):

        if idx < 0 or idx >= len(_metadata):
            continue

        record = _metadata[idx].copy()

        record["distance"] = float(distance)

        record["embedding"] = _embeddings[idx]

        results.append(record)

    return results


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    question = input("Enter Question : ").strip()

    results = retrieve(question)

    print("\n")
    print("=" * 70)
    print("TOP RETRIEVED CHUNKS")
    print("=" * 70)

    if len(results) == 0:

        print("No results found.")

    else:

        for i, item in enumerate(results[:5], start=1):

            print(f"\nRank : {i}")
            print(f"Paper : {item['paper_name']}")
            print(f"Chunk : {item['chunk_id']}")
            print(f"Distance : {item['distance']:.4f}")

            print("-" * 60)

            print(item["chunk_text"][:500])

            print("-" * 60)