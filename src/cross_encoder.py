import numpy as np

from sentence_transformers import CrossEncoder

# ==========================================================
# Lazy-Loaded Model
# ==========================================================
# CrossEncoder used to load at import time, which crashes or
# slows down app startup unnecessarily. Now it only loads the
# first time rerank() is actually called.

_model = None


def _load_model():

    global _model

    if _model is None:

        print("=" * 60)
        print("Loading CrossEncoder...")
        print("=" * 60)

        _model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        print("CrossEncoder Loaded Successfully.\n")


# ==========================================================
# CrossEncoder Re-ranking
# ==========================================================

def rerank(

    question,

    retrieved_chunks,

    top_k=5

):

    if len(retrieved_chunks) == 0:

        return []

    _load_model()

    pairs = []

    for chunk in retrieved_chunks:

        pairs.append(

            (

                question,

                chunk["chunk_text"]

            )

        )

    scores = _model.predict(

        pairs

    )

    for score, chunk in zip(

        scores,

        retrieved_chunks

    ):

        chunk["cross_score"] = float(score)

    retrieved_chunks.sort(

        key=lambda x: x["cross_score"],

        reverse=True

    )

    return retrieved_chunks[:top_k]

# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    from hybrid_retriever import hybrid_search

    question = input(

        "Question : "

    )

    retrieved = hybrid_search(

        question

    )

    reranked = rerank(

        question,

        retrieved

    )

    print()

    print("=" * 70)
    print("CROSSENCODER RESULTS")
    print("=" * 70)

    for i, item in enumerate(

        reranked,

        start=1

    ):

        print()

        print(f"Rank : {i}")

        print(f"Paper : {item['paper_name']}")

        print(f"Page : {item['page']}")

        print(f"Chunk : {item['chunk_id']}")

        print(f"Cross Score : {item['cross_score']:.4f}")

        print("-"*60)

        print(item["chunk_text"][:350])

        print("-"*60)