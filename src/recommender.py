import os
import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

# ==========================================================
# Folder Paths
# ==========================================================

VECTOR_FOLDER = "vector_db"

METADATA_FILE = os.path.join(
    VECTOR_FOLDER,
    "metadata.pkl"
)

INDEX_FILE = os.path.join(
    VECTOR_FOLDER,
    "research_index.faiss"
)

# ==========================================================
# Lazy-Loaded Resources
# ==========================================================
# These used to load at import time, which crashed the whole
# app on startup (before any papers were processed, since
# app.py imports this module before checking paper_count).
# Now they only load the first time recommend_similar_papers()
# is actually called.

_model = None
_index = None
_metadata = None


def _load_resources():

    global _model, _index, _metadata

    if _model is None:

        print("=" * 60)
        print("Loading Sentence-BERT...")
        print("=" * 60)

        _model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Model Loaded Successfully.\n")

    if _metadata is None:

        if not os.path.exists(METADATA_FILE):

            raise FileNotFoundError(
                "metadata.pkl not found. Please process papers first."
            )

        with open(
            METADATA_FILE,
            "rb"
        ) as f:

            _metadata = pickle.load(f)

    if _index is None:

        if not os.path.exists(INDEX_FILE):

            raise FileNotFoundError(
                "research_index.faiss not found. Please process papers first."
            )

        _index = faiss.read_index(
            INDEX_FILE
        )


# ==========================================================
# Recommend Similar Papers
# ==========================================================

def recommend_similar_papers(
    query,
    top_k=20
):

    if query.strip() == "":

        return []

    _load_resources()

    query_embedding = _model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = _index.search(
        query_embedding,
        top_k
    )

    paper_scores = {}

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < 0:
            continue

        if idx >= len(_metadata):
            continue

        paper = _metadata[idx]["paper_name"]

        similarity = float(
            1 / (1 + distance)
        )

        if paper not in paper_scores:

            paper_scores[paper] = []

        paper_scores[paper].append(
            similarity
        )

    recommendations = []

    for paper, scores in paper_scores.items():

        recommendations.append(

            {

                "paper": paper,

                "score": round(
                    float(np.mean(scores)),
                    4
                ),

                "matches": len(scores)

            }

        )

    recommendations.sort(
        key=lambda x: (
            -x["score"],
            x["paper"]
        )
    )

    return recommendations


# ==========================================================
# Print Recommendations
# ==========================================================

def print_recommendations(
    recommendations
):

    print("\n")
    print("=" * 60)
    print("SIMILAR PAPER RECOMMENDATIONS")
    print("=" * 60)

    if len(recommendations) == 0:

        print("No similar papers found.")

        return

    for i, paper in enumerate(
        recommendations,
        start=1
    ):

        print(f"\nRank : {i}")

        print(
            f"Paper          : {paper['paper']}"
        )

        print(
            f"Similarity     : {paper['score']:.4f}"
        )

        print(
            f"Matched Chunks : {paper['matches']}"
        )


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    query = input(
        "Enter Topic : "
    ).strip()

    papers = recommend_similar_papers(
        query=query,
        top_k=20
    )

    print_recommendations(
        papers
    )