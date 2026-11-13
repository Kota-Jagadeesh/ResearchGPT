import os
import re
import pickle
import hashlib
import numpy as np

from sentence_transformers import SentenceTransformer

# ==========================================================
# Folder Paths
# ==========================================================

INPUT_FOLDER = "embeddings"
OUTPUT_FOLDER = "vector_db"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

CACHE_FILE = os.path.join(
    OUTPUT_FOLDER,
    "embedding_cache.pkl"
)

# ==========================================================
# Lazy-Loaded Sentence-BERT
# ==========================================================
# Loading SentenceTransformer used to happen at import time,
# which meant just IMPORTING embedding.py (which app.py does
# at startup via pipeline.py, before Streamlit renders
# anything) would block on downloading/loading this model.
# Now it only loads the first time it's actually needed, i.e.
# when "Process Papers" is actually clicked.

_model = None


def _load_model():

    global _model

    if _model is None:

        print("=" * 60)
        print("Loading Sentence-BERT Model...")
        print("=" * 60)

        _model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Model Loaded Successfully.\n")

    return _model

# ==========================================================
# Read Chunk File
# ==========================================================

def read_chunks(file_path):

    with open(

        file_path,

        "r",

        encoding="utf-8"

    ) as file:

        text = file.read()

    pattern = (

        r"=+\n"

        r"Chunk\s*:\s*(\d+)\n"

        r"Paper\s*:\s*(.*?)\n"

        r"Page\s*:\s*(\d+)\n\n"

        r"(.*?)(?=\n=+|\Z)"

    )

    matches = re.findall(

        pattern,

        text,

        re.DOTALL

    )

    chunks = []

    for chunk_id, paper, page, chunk_text in matches:

        chunks.append(

            {

                "chunk_id": int(chunk_id),

                "paper_name": paper.strip(),

                "page": int(page),

                "chunk_text": chunk_text.strip()

            }

        )

    return chunks


# ==========================================================
# Embedding Cache
# ==========================================================
# Keyed by a hash of the chunk's own text, so identical chunk
# text always maps to the same cached vector regardless of
# which paper/page it came from. Cache self-invalidates: if a
# chunk's text changes at all (re-extracted, re-chunked), its
# hash changes and it's treated as new automatically — no
# manual invalidation logic needed.

def _hash_text(text):

    return hashlib.sha256(

        text.encode("utf-8")

    ).hexdigest()


def _load_cache():

    if not os.path.exists(CACHE_FILE):
        return {}

    try:

        with open(CACHE_FILE, "rb") as file:

            return pickle.load(file)

    except (pickle.UnpicklingError, EOFError, OSError):

        print("Cache file unreadable, starting fresh.")

        return {}


def _save_cache(cache):

    with open(CACHE_FILE, "wb") as file:

        pickle.dump(cache, file)


# ==========================================================
# Encode Chunks (Cache-Aware)
# ==========================================================

def encode_chunks(

    chunk_objects,

    cache

):

    texts = [

        chunk["chunk_text"]

        for chunk in chunk_objects

    ]

    hashes = [
        _hash_text(text)
        for text in texts
    ]

    to_encode_positions = []

    to_encode_texts = []

    for position, text_hash in enumerate(hashes):

        if text_hash not in cache:

            to_encode_positions.append(position)

            to_encode_texts.append(texts[position])

    if to_encode_texts:

        model = _load_model()

        new_vectors = model.encode(

            to_encode_texts,

            batch_size=32,

            show_progress_bar=True,

            convert_to_numpy=True,

            normalize_embeddings=True

        )

        for position, vector in zip(

            to_encode_positions,

            new_vectors

        ):

            cache[hashes[position]] = vector.astype(np.float32)

    vectors = np.asarray(

        [cache[text_hash] for text_hash in hashes],

        dtype=np.float32

    )

    cache_hits = len(texts) - len(to_encode_texts)

    newly_encoded = len(to_encode_texts)

    return vectors, cache_hits, newly_encoded


# ==========================================================
# Generate Embeddings
# ==========================================================

def generate_embeddings():

    files = sorted(

        [

            file

            for file in os.listdir(INPUT_FOLDER)

            if file.lower().endswith("_chunks.txt")

        ]

    )

    if len(files) == 0:

        print("No chunk files found.")

        return

    cache = _load_cache()

    metadata = []

    embeddings = []

    total_chunks = 0

    total_cache_hits = 0

    total_newly_encoded = 0

    print("=" * 60)
    print("Generating Sentence-BERT Embeddings")
    print(f"Existing Cache Entries : {len(cache)}")
    print("=" * 60)

    for paper_id, file in enumerate(

        files,

        start=1

    ):

        print(f"\nProcessing : {file}")

        chunk_file = os.path.join(

            INPUT_FOLDER,

            file

        )

        try:

            chunks = read_chunks(

                chunk_file

            )

            if len(chunks) == 0:

                print("No valid chunks found.")

                continue

            print(

                f"Chunks : {len(chunks)}"

            )

            vectors, cache_hits, newly_encoded = encode_chunks(

                chunks,

                cache

            )

            print(

                f"Cache Hits     : {cache_hits}"
            )

            print(

                f"Newly Encoded  : {newly_encoded}"
            )

            for chunk, vector in zip(

                chunks,

                vectors

            ):

                metadata.append(

                    {

                        "paper_id": paper_id,

                        "paper_name": chunk["paper_name"],

                        "page": chunk["page"],

                        "chunk_id": chunk["chunk_id"],

                        "chunk_text": chunk["chunk_text"]

                    }

                )

                embeddings.append(

                    vector.astype(

                        np.float32

                    )

                )

            total_chunks += len(

                chunks

            )

            total_cache_hits += cache_hits

            total_newly_encoded += newly_encoded

        except Exception as e:

            print(

                f"Failed : {file}"

            )

            print(

                f"Reason : {e}"

            )

    if len(embeddings) == 0:

        print("No embeddings generated.")

        return

    embeddings = np.asarray(

        embeddings,

        dtype=np.float32

    )

    metadata_file = os.path.join(

        OUTPUT_FOLDER,

        "metadata.pkl"

    )

    embedding_file = os.path.join(

        OUTPUT_FOLDER,

        "embeddings.pkl"

    )

    with open(

        metadata_file,

        "wb"

    ) as file:

        pickle.dump(

            metadata,

            file

        )

    with open(

        embedding_file,

        "wb"

    ) as file:

        pickle.dump(

            embeddings,

            file

        )

    _save_cache(cache)

    print("\n")

    print("=" * 60)
    print("Embedding Generation Completed")
    print("=" * 60)

    print(

        f"Total Papers          : {len(files)}"

    )

    print(

        f"Total Chunks          : {total_chunks}"

    )

    if total_chunks > 0:

        cache_hit_rate = round(

            (total_cache_hits / total_chunks) * 100,

            1

        )

    else:

        cache_hit_rate = 0.0

    print(

        f"Cache Hits             : {total_cache_hits} ({cache_hit_rate}%)"

    )

    print(

        f"Newly Encoded          : {total_newly_encoded}"

    )

    print(

        f"Embedding Dimension   : {embeddings.shape[1]}"

    )

    print(

        f"Embedding Shape       : {embeddings.shape}"

    )

    print("\nSaved Files")

    print(

        f"✓ {metadata_file}"

    )

    print(

        f"✓ {embedding_file}"

    )

    print(

        f"✓ {CACHE_FILE}"

    )

    print("\nMetadata Fields Stored")

    print(

        "✓ Paper Name"

    )

    print(

        "✓ Page Number"

    )

    print(

        "✓ Chunk Number"

    )

    print(

        "✓ Chunk Text"

    )

    print(

        "✓ Embedding Vector"

    )

    print("\n")

    print("=" * 60)
    print("Embedding Generation Completed Successfully.")
    print("=" * 60)


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    generate_embeddings()