import os
import pickle

# ==========================================================
# Folder Paths
# ==========================================================

RAW_FOLDER = "data/raw"
PROCESSED_FOLDER = "data/processed"
CHUNK_FOLDER = "embeddings"
VECTOR_FOLDER = "vector_db"

# ==========================================================
# Count PDF Files
# ==========================================================

def get_total_papers():

    if not os.path.exists(RAW_FOLDER):
        return 0

    return len(
        [
            f
            for f in os.listdir(RAW_FOLDER)
            if f.endswith(".pdf")
        ]
    )

# ==========================================================
# Count Chunk Files
# ==========================================================

def get_total_chunk_files():

    if not os.path.exists(CHUNK_FOLDER):
        return 0

    return len(
        [
            f
            for f in os.listdir(CHUNK_FOLDER)
            if f.endswith("_chunks.txt")
        ]
    )

# ==========================================================
# Count Total Chunks
# ==========================================================

def get_total_chunks():

    metadata_path = os.path.join(
        VECTOR_FOLDER,
        "metadata.pkl"
    )

    if not os.path.exists(metadata_path):
        return 0

    with open(
        metadata_path,
        "rb"
    ) as f:

        metadata = pickle.load(f)

    return len(metadata)

# ==========================================================
# Embedding Dimension
# ==========================================================

def get_embedding_dimension():

    embedding_path = os.path.join(
        VECTOR_FOLDER,
        "embeddings.pkl"
    )

    if not os.path.exists(embedding_path):
        return 0

    with open(
        embedding_path,
        "rb"
    ) as f:

        embeddings = pickle.load(f)

    if len(embeddings) == 0:
        return 0

    return len(embeddings[0])

# ==========================================================
# Vector Count
# ==========================================================

def get_vector_count():

    embedding_path = os.path.join(
        VECTOR_FOLDER,
        "embeddings.pkl"
    )

    if not os.path.exists(embedding_path):
        return 0

    with open(
        embedding_path,
        "rb"
    ) as f:

        embeddings = pickle.load(f)

    return len(embeddings)

# ==========================================================
# Project Statistics
# ==========================================================

def get_project_statistics():

    stats = {

        "papers":
            get_total_papers(),

        "chunk_files":
            get_total_chunk_files(),

        "chunks":
            get_total_chunks(),

        "embedding_dimension":
            get_embedding_dimension(),

        "vectors":
            get_vector_count()

    }

    return stats

# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    stats = get_project_statistics()

    print("=" * 60)
    print("ResearchGPT Statistics")
    print("=" * 60)

    print(f"Papers              : {stats['papers']}")
    print(f"Chunk Files         : {stats['chunk_files']}")
    print(f"Total Chunks        : {stats['chunks']}")
    print(f"Embedding Dimension : {stats['embedding_dimension']}")
    print(f"Vectors Stored      : {stats['vectors']}")