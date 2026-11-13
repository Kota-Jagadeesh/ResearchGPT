import numpy as np

from dense_retriever import retrieve
from bm25_retriever import bm25_search

# ==========================================================
# Normalize Score
# ==========================================================

def normalize(scores):

    scores = np.asarray(scores, dtype=np.float32)

    if len(scores) == 0:

        return scores

    minimum = scores.min()

    maximum = scores.max()

    if maximum == minimum:

        return np.ones_like(scores)

    return (scores - minimum) / (maximum - minimum)

# ==========================================================
# Hybrid Search
# ==========================================================

def hybrid_search(

    question,

    dense_top_k=100,

    sparse_top_k=100,

    alpha=0.6,

    paper_filter=None

):

    # --------------------------------------------------------
    # Metadata Filtering (Scoped Retrieval)
    # --------------------------------------------------------
    # paper_filter: optional list of paper filenames to restrict
    # results to (e.g. ["RAG.pdf", "FiD.pdf"]). Since dense/sparse
    # retrieval return a fixed top_k BEFORE filtering, most of
    # that top_k may belong to papers outside the filter — so
    # when a filter is active, we retrieve a larger candidate
    # pool first, then filter, so the effective top_k after
    # filtering still has enough real candidates left.
    # --------------------------------------------------------

    effective_dense_top_k = dense_top_k

    effective_sparse_top_k = sparse_top_k

    if paper_filter:

        effective_dense_top_k = min(dense_top_k * 3, 500)

        effective_sparse_top_k = min(sparse_top_k * 3, 500)

    dense_results = retrieve(

        question,

        top_k=effective_dense_top_k

    )

    sparse_results = bm25_search(

        question,

        top_k=effective_sparse_top_k

    )

    if paper_filter:

        paper_filter_set = set(paper_filter)

        dense_results = [

            item for item in dense_results

            if item["paper_name"] in paper_filter_set

        ]

        sparse_results = [

            item for item in sparse_results

            if item["paper_name"] in paper_filter_set

        ]

    # ------------------------------------------------------
    # IMPORTANT: faiss_db.py builds an IndexFlatIP index on
    # L2-normalized vectors, so dense_retriever's "distance"
    # field is actually COSINE SIMILARITY — HIGHER = better
    # match (opposite of true L2 distance). The old code here
    # applied 1/(1+distance), which is only correct for true
    # L2 distance; applying it to an already-good similarity
    # score inverted the ranking (a great match would get a
    # LOW transformed score). We now use the raw similarity
    # directly — it's already on a higher-is-better scale, so
    # it just needs the same min-max normalization as BM25.
    # ------------------------------------------------------

    dense_scores = normalize(
        [item["distance"] for item in dense_results]
    )

    sparse_scores = normalize(

        [item["bm25_score"] for item in sparse_results]

    )

    merged = {}

    # ---------------- Dense ----------------

    for score, item in zip(

        dense_scores,

        dense_results

    ):

        key = (

            item["paper_name"],

            item["chunk_id"]

        )

        merged[key] = item.copy()

        merged[key]["hybrid_score"] = (

            alpha * float(score)

        )

    # ---------------- Sparse ----------------

    for score, item in zip(

        sparse_scores,

        sparse_results

    ):

        key = (

            item["paper_name"],

            item["chunk_id"]

        )

        if key not in merged:

            merged[key] = item.copy()

            merged[key]["hybrid_score"] = 0

        merged[key]["hybrid_score"] += (

            (1-alpha) * float(score)

        )

    # ------------------------------------------------------
    # Ensure every merged item carries a "distance" field.
    # Items that came ONLY from sparse (BM25) results won't
    # have one (BM25 has no notion of vector similarity), and
    # downstream code (calculate_confidence, AGMR clustering)
    # expects "distance" to exist on every chunk, consistently
    # meaning higher-is-better similarity. hybrid_score is
    # already on that same higher-is-better scale, so it's
    # used directly as the proxy — NOT (1 - hybrid_score),
    # which would flip the convention backwards.
    # ------------------------------------------------------

    for key, item in merged.items():

        if "distance" not in item:

            item["distance"] = item["hybrid_score"]

    results = list(

        merged.values()

    )

    results.sort(

        key=lambda x:

        x["hybrid_score"],

        reverse=True

    )

    return results

# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    question = input(

        "Question : "

    )

    results = hybrid_search(

        question

    )

    print()

    print("="*70)

    print("HYBRID SEARCH RESULTS")

    print("="*70)

    for i, item in enumerate(

        results[:10],

        start=1

    ):

        print()

        print(f"Rank : {i}")

        print(f"Paper : {item['paper_name']}")

        print(f"Page : {item['page']}")

        print(f"Chunk : {item['chunk_id']}")

        print(f"Hybrid Score : {item['hybrid_score']:.4f}")

        print("-"*60)

        print(item["chunk_text"][:300])

        print("-"*60)