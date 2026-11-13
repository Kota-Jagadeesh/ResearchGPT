import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# ==========================================================
# Folder Paths
# ==========================================================

INPUT_FOLDER = "data/chunks"
OUTPUT_FOLDER = "embeddings"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# Token-Based Chunk Sizing
# ==========================================================
# Chunk size / overlap are measured in TOKENS, not characters,
# using the same tokenizer as the embedding model
# (all-MiniLM-L6-v2) so the token count here matches what the
# embedding model actually sees.
#
# IMPORTANT CAVEAT: all-MiniLM-L6-v2 has a max sequence length
# of 256 tokens. A 500-token chunk will be silently TRUNCATED
# to its first ~256 tokens when it's embedded — the embedding
# vector used for retrieval will only represent roughly the
# first half of each chunk, even though the full chunk text is
# still shown to the LLM when generating the final answer.
#
# If you want the full 500-token chunk to be represented in
# the embedding (not just the first half), either:
#   1. Lower CHUNK_SIZE_TOKENS to ~250, or
#   2. Switch to a longer-context embedding model, e.g.
#      "sentence-transformers/all-mpnet-base-v2" (384 tokens) or
#      "BAAI/bge-base-en-v1.5" (512 tokens)
# ==========================================================

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_MAX_TOKENS = 256

if CHUNK_SIZE_TOKENS > EMBEDDING_MODEL_MAX_TOKENS:

    print(
        f"⚠ WARNING: CHUNK_SIZE_TOKENS ({CHUNK_SIZE_TOKENS}) exceeds "
        f"the embedding model's max sequence length "
        f"({EMBEDDING_MODEL_MAX_TOKENS}). Embeddings will only "
        f"represent the first {EMBEDDING_MODEL_MAX_TOKENS} tokens "
        f"of each chunk.\n"
    )

# ==========================================================
# Lazy-Loaded Tokenizer
# ==========================================================
# Loading AutoTokenizer used to happen at import time, which
# meant just IMPORTING chunking.py (which app.py does at
# startup via pipeline.py, before Streamlit renders anything)
# would block on downloading/loading this model. Now it only
# loads the first time count_tokens() is actually called,
# i.e. when "Process Papers" is actually clicked.

_tokenizer = None


def _load_tokenizer():

    global _tokenizer

    if _tokenizer is None:

        print("=" * 60)
        print("Loading Tokenizer for Token-Based Chunking...")
        print("=" * 60)

        _tokenizer = AutoTokenizer.from_pretrained(
            EMBEDDING_MODEL_NAME
        )

        print("Tokenizer Loaded Successfully.\n")


def count_tokens(text):

    _load_tokenizer()

    return len(
        _tokenizer.encode(
            text,
            add_special_tokens=False
        )
    )


# ==========================================================
# Chunk Splitter (Token-Aware)
# ==========================================================

splitter = RecursiveCharacterTextSplitter(

    chunk_size=CHUNK_SIZE_TOKENS,

    chunk_overlap=CHUNK_OVERLAP_TOKENS,

    length_function=count_tokens,

    separators=[

        "\n\n",

        "\n",

        ".",

        " ",

        ""

    ]

)

# ==========================================================
# Read Pages
# ==========================================================

def extract_pages(text):

    pattern = r"Paper\s*:\s*(.*?)\nPage\s*:\s*(\d+)\n\n(.*?)(?=\n={20,}|\Z)"

    matches = re.findall(

        pattern,

        text,

        re.DOTALL

    )

    pages = []

    for paper, page, content in matches:

        pages.append(

            {

                "paper": paper.strip(),

                "page": int(page),

                "text": content.strip()

            }

        )

    return pages

# ==========================================================
# Chunk All Files
# ==========================================================

def chunk_all_files():

    files = sorted(

        [

            file

            for file in os.listdir(INPUT_FOLDER)

            if file.lower().endswith(".txt")

        ]

    )

    if len(files) == 0:

        print("No processed text files found.")

        return

    print("=" * 60)
    print("ResearchGPT Chunk Generator")
    print(f"Chunk Size    : {CHUNK_SIZE_TOKENS} tokens")
    print(f"Chunk Overlap : {CHUNK_OVERLAP_TOKENS} tokens")
    print("=" * 60)

    total_chunks = 0

    for file in files:

        input_path = os.path.join(

            INPUT_FOLDER,

            file

        )

        try:

            with open(

                input_path,

                "r",

                encoding="utf-8"

            ) as f:

                text = f.read()

            pages = extract_pages(text)

            output_file = os.path.join(

                OUTPUT_FOLDER,

                os.path.splitext(file)[0] +

                "_chunks.txt"

            )

            chunk_id = 1

            with open(

                output_file,

                "w",

                encoding="utf-8"

            ) as out:

                for page in pages:

                    chunks = splitter.split_text(

                        page["text"]

                    )

                    for chunk in chunks:

                        out.write(

                            "=" * 70 + "\n"

                        )

                        out.write(

                            f"Chunk : {chunk_id}\n"

                        )

                        out.write(

                            f"Paper : {page['paper']}\n"

                        )

                        out.write(

                            f"Page : {page['page']}\n\n"

                        )

                        out.write(

                            chunk.strip()

                        )

                        out.write("\n\n")

                        chunk_id += 1

            total_chunks += (

                chunk_id - 1

            )

            print()

            print(

                f"Processed          : {file}"

            )

            print(

                f"Chunks Generated   : {chunk_id-1}"

            )

            print(

                f"Saved              : {output_file}"

            )

        except Exception as e:

            print()

            print(

                f"Failed             : {file}"

            )

            print(

                f"Reason             : {e}"

            )

    print()

    print("=" * 60)
    print("Chunk Generation Summary")
    print("=" * 60)

    print(

        f"Files Processed    : {len(files)}"

    )

    print(

        f"Total Chunks       : {total_chunks}"

    )

    print("=" * 60)
    print("Chunk Generation Completed Successfully.")

# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    chunk_all_files()