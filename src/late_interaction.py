import numpy as np
import torch

from sentence_transformers import SentenceTransformer

# ==========================================================
# Late Interaction (MaxSim) Reranking
# ==========================================================
# HONEST SCOPE NOTE: this is NOT a true ColBERT integration.
# Real ColBERT uses a model specifically trained for late
# interaction and requires its own token-level index built
# and stored separately from a standard FAISS index. Building
# and maintaining that is a significant infra project on its
# own.
#
# What this module actually does: it pulls TOKEN-level
# embeddings (not the usual single pooled vector) out of the
# same all-MiniLM-L6-v2 bi-encoder you're already using, then
# scores query/document pairs with MaxSim — the same scoring
# mechanism ColBERT uses. This gets you the core idea (matching
# at the token level instead of collapsing everything into one
# vector) without a second model or a second index, at the
# cost of using an encoder that wasn't specifically trained for
# this. Treat it as an experimental extra reranking stage, not
# a drop-in ColBERT replacement.

MODEL_NAME = "all-MiniLM-L6-v2"

MAX_TOKENS = 256

_model = None
_tokenizer = None


def _load():

    global _model, _tokenizer

    if _model is None:

        print("=" * 60)
        print("Loading model for Late-Interaction (MaxSim) scoring...")
        print("=" * 60)

        _model = SentenceTransformer(MODEL_NAME)

        _tokenizer = _model.tokenizer

        print("Model Loaded Successfully.\n")


def _token_embeddings(text):

    _load()

    features = _tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        max_length=MAX_TOKENS,

        padding=True

    )

    with torch.no_grad():

        output = _model._first_module().auto_model(**features)

    token_embeds = output.last_hidden_state[0]

    token_embeds = torch.nn.functional.normalize(
        token_embeds,
        p=2,
        dim=1
    )

    return token_embeds.numpy()


def _maxsim_score(query_embeds, doc_embeds):

    similarities = query_embeds @ doc_embeds.T

    max_per_query_token = similarities.max(axis=1)

    return float(np.sum(max_per_query_token))


# ==========================================================
# Late Interaction Rerank
# ==========================================================

def late_interaction_rerank(

    question,

    chunks,

    top_k=10

):

    if len(chunks) == 0:
        return []

    query_embeds = _token_embeddings(question)

    scored = []

    for chunk in chunks:

        doc_text = chunk.get("chunk_text", "")[:1000]

        doc_embeds = _token_embeddings(doc_text)

        score = _maxsim_score(
            query_embeds,
            doc_embeds
        )

        chunk = dict(chunk)

        chunk["late_interaction_score"] = score

        scored.append(chunk)

    scored.sort(

        key=lambda x: x["late_interaction_score"],

        reverse=True

    )

    return scored[:top_k]


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    from hybrid_retriever import hybrid_search

    question = input("Question : ").strip()

    candidates = hybrid_search(question)

    reranked = late_interaction_rerank(

        question,

        candidates,

        top_k=10

    )

    print()
    print("=" * 70)
    print("LATE INTERACTION (MaxSim) RESULTS")
    print("=" * 70)

    for i, item in enumerate(reranked, start=1):

        print(f"\nRank : {i}")

        print(f"Paper : {item['paper_name']}")

        print(f"Page : {item.get('page', 'N/A')}")

        print(f"MaxSim Score : {item['late_interaction_score']:.4f}")

        print("-" * 60)

        print(item["chunk_text"][:300])

        print("-" * 60)