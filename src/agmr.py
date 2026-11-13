import numpy as np
from sklearn.mixture import GaussianMixture

from hybrid_retriever import hybrid_search
from cross_encoder import rerank as cross_encoder_rerank
from query_expansion import build_query
from confidence import calculate_confidence


# ==========================================================
# GMM Diversity Clustering
# ==========================================================
# Groups the cross-encoder-reranked candidates into clusters
# and picks the most confident, diverse chunk from each
# cluster, so the final selection isn't just N near-duplicate
# passages of the same point.

def _diversify_with_gmm(

    chunks,

    num_clusters,

    final_chunks

):

    # NOTE: "distance" here is actually cosine similarity
    # (higher = better match), inherited from dense_retriever.py
    # (IndexFlatIP on normalized vectors) and hybrid_retriever.py.
    # All sorting/tie-breaking below prefers HIGHER distance.

    embeddings = np.asarray(
        [chunk["embedding"] for chunk in chunks],
        dtype=np.float32
    )

    num_clusters = min(
        num_clusters,
        len(embeddings)
    )

    if num_clusters <= 0:
        return chunks[:final_chunks]

    try:

        gmm = GaussianMixture(
            n_components=num_clusters,
            covariance_type="full",
            random_state=42
        )

        gmm.fit(embeddings)

    except Exception as e:

        print(e)

        return chunks[:final_chunks]

    labels = gmm.predict(embeddings)

    probabilities = gmm.predict_proba(embeddings)

    cluster_scores = []

    for cluster in range(num_clusters):

        ids = np.where(labels == cluster)[0]

        if len(ids) == 0:
            continue

        confidence = float(
            probabilities[ids, cluster].mean()
        )

        average_distance = float(
            np.mean([chunks[idx]["distance"] for idx in ids])
        )

        cluster_scores.append(
            {
                "cluster": cluster,
                "confidence": confidence,
                "average_distance": average_distance,
                "ids": ids
            }
        )

    cluster_scores.sort(
        key=lambda x: (-x["confidence"], -x["average_distance"])
    )

    print("\nCluster Confidence\n")

    for cluster in cluster_scores:

        print(
            f"Cluster {cluster['cluster']} | "
            f"Confidence = {cluster['confidence']:.4f} | "
            f"Average Distance = {cluster['average_distance']:.4f} | "
            f"Chunks = {len(cluster['ids'])}"
        )

    selected = []

    used = set()

    for cluster in cluster_scores:

        cluster_items = sorted(
            cluster["ids"],
            key=lambda idx: chunks[idx]["distance"],
            reverse=True
        )

        for idx in cluster_items:

            key = (
                chunks[idx]["paper_name"],
                chunks[idx]["chunk_id"]
            )

            if key in used:
                continue

            used.add(key)

            selected.append(chunks[idx])

            if len(selected) >= final_chunks:
                break

        if len(selected) >= final_chunks:
            break

    return selected


# ==========================================================
# AGMR Re-Ranking (Hybrid Search + CrossEncoder + Diversity)
# ==========================================================

def agmr_rerank(
    question,
    retrieve_top_k=100,
    num_clusters=5,
    final_chunks=5,
    rerank_top_k=10,
    paper_filter=None
):

    print("=" * 70)
    print("Hybrid Retrieval + CrossEncoder Reranking + AGMR")
    print("=" * 70)

    if paper_filter:
        print(f"Scoped to papers : {paper_filter}")

    expanded_query = build_query(question)

    print(f"\nExpanded Query : {expanded_query}")

    candidates = hybrid_search(
        expanded_query,
        dense_top_k=retrieve_top_k,
        sparse_top_k=retrieve_top_k,
        paper_filter=paper_filter
    )

    if len(candidates) == 0:

        print("No chunks retrieved.")

        return [], {
            "confidence": 0,
            "label": "No Confidence"
        }

    print(f"Hybrid Candidates : {len(candidates)}")

    # --------------------------------------------------------
    # CrossEncoder reranking uses the ORIGINAL question, not the
    # expanded one — expansion widens recall for retrieval, but
    # scoring relevance against a synonym-stuffed query would
    # distort the reranker's judgement.
    # --------------------------------------------------------

    reranked = cross_encoder_rerank(
        question,
        candidates,
        top_k=min(rerank_top_k, len(candidates))
    )

    print(f"Reranked Top {len(reranked)} (CrossEncoder)")

    has_embeddings = all(
        "embedding" in chunk
        for chunk in reranked
    )

    if has_embeddings and len(reranked) >= 2:

        selected = _diversify_with_gmm(
            reranked,
            num_clusters=min(num_clusters, len(reranked)),
            final_chunks=final_chunks
        )

    else:

        # Some/all reranked candidates came only from BM25 (sparse)
        # and have no dense embedding, so GMM clustering can't run.
        # Fall back to taking the top CrossEncoder-scored chunks.
        print("Skipping GMM diversity step (embeddings unavailable for some candidates).")

        selected = reranked[:final_chunks]

    confidence_result = calculate_confidence(
        selected
    )

    print("\n")
    print("=" * 70)
    print("Answer Confidence")
    print("=" * 70)

    print(
        f"Confidence : {confidence_result['confidence']} %"
    )

    print(
        f"Level      : {confidence_result['label']}"
    )

    print(f"\nSelected Chunks : {len(selected)}")

    return selected, confidence_result


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    question = input(
        "\nEnter Question : "
    ).strip()

    results, confidence = agmr_rerank(
        question
    )

    print("\n")
    print("=" * 70)
    print("FINAL AGMR RESULTS")
    print("=" * 70)

    if len(results) == 0:

        print("No Results Found.")

    else:

        for i, item in enumerate(
            results,
            start=1
        ):

            print(f"\nRank : {i}")

            print(
                f"Paper : {item['paper_name']}"
            )

            print(
                f"Page  : {item.get('page', 'N/A')}"
            )

            print(
                f"Chunk : {item['chunk_id']}"
            )

            print(
                f"Distance : {item['distance']:.4f}"
            )

            if "cross_score" in item:

                print(
                    f"CrossEncoder Score : {item['cross_score']:.4f}"
                )

            print("-" * 60)

            print(
                item["chunk_text"][:400]
            )

            print("-" * 60)

    print("\nConfidence Summary")

    print(confidence)