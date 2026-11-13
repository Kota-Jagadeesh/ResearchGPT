import os
import pickle
import re

from rank_bm25 import BM25Okapi

# ==========================================================
# Folder
# ==========================================================

VECTOR_FOLDER = "vector_db"

METADATA_FILE = os.path.join(

    VECTOR_FOLDER,

    "metadata.pkl"

)

# ==========================================================
# Tokenizer
# ==========================================================

def tokenize(text):

    text = text.lower()

    words = re.findall(

        r"\w+",

        text

    )

    return words

# ==========================================================
# Lazy-Loaded Resources
# ==========================================================
# These used to load at import time, which crashes the whole
# app on startup (before any papers were processed) since
# hybrid_retriever.py imports bm25_search at module level.
# Now metadata/corpus/index only build the first time
# bm25_search() is actually called.

_metadata = None
_bm25 = None


def _load_resources():

    global _metadata, _bm25

    if _metadata is None:

        if not os.path.exists(METADATA_FILE):
            raise FileNotFoundError(
                f"Metadata file not found: {METADATA_FILE}. Please process papers first."
            )

        with open(
            METADATA_FILE,
            "rb"
        ) as file:

            _metadata = pickle.load(file)

    if _bm25 is None:

        corpus = []

        for item in _metadata:

            corpus.append(
                tokenize(
                    item["chunk_text"]
                )
            )

        _bm25 = BM25Okapi(corpus)


# ==========================================================
# Sparse Retrieval
# ==========================================================

def bm25_search(

    question,

    top_k=20

):

    if not question.strip():
        return []

    _load_resources()

    tokens = tokenize(

        question

    )

    scores = _bm25.get_scores(

        tokens

    )

    ranked = sorted(

        zip(scores, _metadata),

        reverse=True,

        key=lambda x: x[0]

    )

    results = []

    for score, item in ranked[:top_k]:

        result = item.copy()

        result["bm25_score"] = float(score)

        results.append(

            result

        )

    return results

# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    question = input(

        "Question : "

    )

    results = bm25_search(

        question

    )

    print()

    print("=" * 70)

    print("BM25 RESULTS")

    print("=" * 70)

    for i, item in enumerate(

        results,

        start=1

    ):

        print()

        print(f"Rank : {i}")

        print(f"Paper : {item['paper_name']}")

        print(f"Page : {item['page']}")

        print(f"Chunk : {item['chunk_id']}")

        print(f"Score : {item['bm25_score']:.3f}")

        print("-"*60)

        print(item["chunk_text"][:350])

        print("-"*60)