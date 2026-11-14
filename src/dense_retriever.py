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
_last_loaded_mtime = 0.0


def reload_resources():
    """Forces reloading of FAISS index, metadata, and embeddings from disk."""
    global _index, _metadata, _embeddings, _last_loaded_mtime
    _index = None
    _metadata = None
    _embeddings = None
    _last_loaded_mtime = 0.0
    _load_resources()


def get_indexed_paper_names():
    """Returns set of paper names currently present in the metadata index."""
    _load_resources()
    if not _metadata:
        return set()
    return {item.get("paper_name") for item in _metadata if item.get("paper_name")}


def get_paper_title(paper_name):
    """Extracts a readable academic title for a paper from its first metadata chunks."""
    _load_resources()
    clean_fallback = paper_name.replace(".pdf", "").replace("_", " ").strip()
    if not _metadata:
        return clean_fallback

    for m in _metadata:
        if m.get("paper_name") == paper_name and m.get("chunk_id") in (1, 2):
            text = m.get("chunk_text", "")
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            for l in lines:
                l_upper = l.upper()
                if (
                    len(l) > 15
                    and not l.startswith("http")
                    and not l_upper.startswith("ISSN")
                    and not l_upper.startswith("DOI")
                    and not l_upper.startswith("COMMENT")
                    and not l_upper.startswith("RECEIVED")
                    and not l_upper.startswith("ACCEPTED")
                    and not l_upper.startswith("PUBLISHED")
                    and not l_upper.startswith("IJARSCT")
                    and not l_upper.startswith("INTERNATIONAL JOURNAL")
                    and not l_upper.startswith("JOURNAL OF")
                    and not l_upper.startswith("PROCEEDINGS OF")
                    and not l_upper.startswith("VOLUME")
                    and not l_upper.startswith("ISSUE")
                    and not l_upper.startswith("COPYRIGHT")
                    and not l_upper.startswith("WWW.")
                    and not l_upper.startswith("IMPACT FACTOR")
                    and not l_upper.startswith("SUBMISSION ID")
                    and not l_upper.startswith("DOCUMENT DETAILS")
                    and not l_upper.startswith("PAGE")
                    and not l_upper.startswith("AMRITA")
                    and not l_upper.startswith("QUICK SUBMIT")
                    and not l_upper.startswith("SUBMISSION DATE")
                ):
                    return l[:100]

    return clean_fallback


def _load_resources():

    global _model, _index, _metadata, _embeddings, _last_loaded_mtime

    if _model is None:

        print("=" * 60)
        print("Loading Sentence-BERT...")
        print("=" * 60)

        _model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Sentence-BERT Loaded Successfully.\n")

    # Check if disk files were updated since last load
    disk_mtime = 0.0
    if os.path.exists(INDEX_PATH):
        disk_mtime = max(disk_mtime, os.path.getmtime(INDEX_PATH))
    if os.path.exists(METADATA_PATH):
        disk_mtime = max(disk_mtime, os.path.getmtime(METADATA_PATH))

    if disk_mtime > _last_loaded_mtime:
        # Disk files updated, invalidate existing cached index
        _index = None
        _metadata = None
        _embeddings = None

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

    _last_loaded_mtime = disk_mtime


# ==========================================================
# Query Encoding Function
# ==========================================================

def encode_query(question):
    if not question.strip():
        return None
    _load_resources()
    embedding = _model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)
    return embedding[0]


# ==========================================================
# Retrieve Function
# ==========================================================

def retrieve(question, top_k=100, paper_filter=None):

    if not question.strip():
        return []

    _load_resources()

    if _index is None or len(_metadata) == 0:
        return []

    question_embedding = _model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    paper_filter_set = set(paper_filter) if paper_filter else None

    # When paper_filter is set, search all chunks in the index to guarantee
    # we find the best chunks belonging to the targeted paper(s).
    search_k = _index.ntotal if paper_filter_set else min(top_k, _index.ntotal)

    distances, indices = _index.search(
        question_embedding,
        search_k
    )

    results = []

    for distance, idx in zip(distances[0], indices[0]):

        if idx < 0 or idx >= len(_metadata):
            continue

        record = _metadata[idx]

        if paper_filter_set and record.get("paper_name") not in paper_filter_set:
            continue

        item = record.copy()
        item["distance"] = float(distance)

        if _embeddings is not None and idx < len(_embeddings):
            item["embedding"] = _embeddings[idx]

        results.append(item)

        if len(results) >= top_k:
            break

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