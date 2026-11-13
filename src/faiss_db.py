import os
import pickle
import numpy as np
import faiss

# ==========================================================
# Folder
# ==========================================================

VECTOR_FOLDER = "vector_db"

# ==========================================================
# Build FAISS Index
# ==========================================================

def build_faiss_index():

    print("=" * 60)
    print("Building FAISS Index")
    print("=" * 60)

    embeddings_file = os.path.join(
        VECTOR_FOLDER,
        "embeddings.pkl"
    )

    metadata_file = os.path.join(
        VECTOR_FOLDER,
        "metadata.pkl"
    )

    if not os.path.exists(embeddings_file):

        raise FileNotFoundError(
            f"Embedding file not found : {embeddings_file}"
        )

    if not os.path.exists(metadata_file):

        raise FileNotFoundError(
            f"Metadata file not found : {metadata_file}"
        )

    # ======================================================
    # Load Embeddings
    # ======================================================

    with open(

        embeddings_file,

        "rb"

    ) as file:

        embeddings = pickle.load(file)

    embeddings = np.asarray(

        embeddings,

        dtype=np.float32

    )

    # ======================================================
    # Load Metadata
    # ======================================================

    with open(

        metadata_file,

        "rb"

    ) as file:

        metadata = pickle.load(file)

    # ======================================================
    # Validation
    # ======================================================

    if embeddings.ndim != 2:

        raise ValueError(
            "Embeddings must be a 2D array."
        )

    if len(embeddings) == 0:

        raise ValueError(
            "No embeddings found."
        )

    if len(embeddings) != len(metadata):

        raise ValueError(
            "Metadata and embeddings size mismatch."
        )

    dimension = embeddings.shape[1]

    print(f"Embedding Dimension : {dimension}")
    print(f"Total Embeddings    : {len(embeddings)}")
    print(f"Metadata Records    : {len(metadata)}")

    # ======================================================
    # Normalize Embeddings
    # ======================================================

    faiss.normalize_L2(

        embeddings

    )

    # ======================================================
    # Build Index
    # ======================================================

    index = faiss.IndexFlatIP(

        dimension

    )

    index.add(

        embeddings

    )

    # ======================================================
    # Save Index
    # ======================================================

    index_file = os.path.join(

        VECTOR_FOLDER,

        "research_index.faiss"

    )

    faiss.write_index(

        index,

        index_file

    )

    # ======================================================
    # Summary
    # ======================================================

    print()

    print("=" * 60)
    print("FAISS Index Created Successfully")
    print("=" * 60)

    print(f"Vectors Stored      : {index.ntotal}")
    print(f"Embedding Dimension : {dimension}")
    print(f"Index Type          : IndexFlatIP")
    print(f"Similarity          : Cosine")
    print(f"Saved               : {index_file}")

    print()

    print("Metadata Stored")

    print("----------------------------")

    sample = metadata[0]

    for key in sample:

        print(f"✓ {key}")

    print()

    print("FAISS Database Ready.")

# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    build_faiss_index()