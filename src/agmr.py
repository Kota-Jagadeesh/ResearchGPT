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
    """
    Diversifies top Cross-Encoder reranked candidates using GMM clustering.
    Guarantees the #1 Cross-Encoder match is always selected, then incorporates
    high-quality candidates from other semantic clusters to prevent redundancy,
    while filling remaining slots with top reranked evidence.
    """
    if len(chunks) <= final_chunks:
        return chunks

    embeddings = np.asarray(
        [chunk["embedding"] for chunk in chunks],
        dtype=np.float32
    )

    num_clusters = min(num_clusters, len(embeddings))

    if num_clusters <= 0:
        return chunks[:final_chunks]

    try:
        gmm = GaussianMixture(
            n_components=num_clusters,
            covariance_type="full",
            random_state=42
        )
        labels = gmm.fit_predict(embeddings)
    except Exception as e:
        print(f"GMM clustering fallback: {e}")
        return chunks[:final_chunks]

    # Assign cluster labels
    for idx, c in enumerate(chunks):
        c["_cluster"] = int(labels[idx])

    selected = [chunks[0]]
    selected_keys = {(chunks[0]["paper_name"], chunks[0]["chunk_id"])}
    seen_clusters = {chunks[0]["_cluster"]}

    # Try to pick the best chunk from distinct clusters if relevance is good
    for c in chunks[1:]:
        if len(selected) >= final_chunks:
            break
        key = (c["paper_name"], c["chunk_id"])
        if key in selected_keys:
            continue
        # Only take diverse clusters if the candidate is not deeply negative
        if c["_cluster"] not in seen_clusters and c.get("cross_score", 0.0) > -2.0:
            seen_clusters.add(c["_cluster"])
            selected.append(c)
            selected_keys.add(key)

    # Fill remaining slots with the highest-ranked Cross-Encoder candidates
    for c in chunks[1:]:
        if len(selected) >= final_chunks:
            break
        key = (c["paper_name"], c["chunk_id"])
        if key not in selected_keys:
            selected.append(c)
            selected_keys.add(key)

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

        if paper_filter:
            try:
                from dense_retriever import _metadata, _embeddings, _load_resources
                _load_resources()
                filter_set = set(paper_filter)
                for idx, m in enumerate(_metadata):
                    if m.get("paper_name") in filter_set:
                        item = m.copy()
                        item["distance"] = 0.40
                        if _embeddings is not None and idx < len(_embeddings):
                            item["embedding"] = _embeddings[idx]
                        candidates.append(item)
                if candidates:
                    print(f"Safety fallback recovered {len(candidates)} candidates from metadata for {paper_filter}")
            except Exception as e:
                print(f"Metadata fallback notice: {e}")

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
        selected,
        question=question
    )
    confidence_result["candidates_pool"] = candidates

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