import os
import re

# ==========================================================
# Folder
# ==========================================================
# pdf_loader.py already saves the full page-level extracted
# text here (Paper: X / Page: Y / <full page text>). We reuse
# that as the "parent document" store for parent-document
# retrieval, so no changes to the ingestion pipeline are
# needed — chunks stay small and precise for matching, but the
# LLM gets the full parent page for context.

PROCESSED_FOLDER = "data/processed"

# Cap how much of a parent page we send to the LLM, so a single
# very dense page can't blow out the prompt budget.
MAX_PARENT_CHARS = 3000

# In-memory cache: paper_name -> {page_number: page_text}
_page_index_cache = {}


# ==========================================================
# Build Page Index For One Paper
# ==========================================================

def _build_page_index(paper_name):

    txt_filename = os.path.splitext(paper_name)[0] + ".txt"

    txt_path = os.path.join(
        PROCESSED_FOLDER,
        txt_filename
    )

    if not os.path.exists(txt_path):
        return {}

    with open(
        txt_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    pattern = r"Paper\s*:\s*(.*?)\nPage\s*:\s*(\d+)\n\n(.*?)(?=\n={20,}|\Z)"

    matches = re.findall(
        pattern,
        text,
        re.DOTALL
    )

    page_index = {}

    for paper, page, content in matches:

        page_index[int(page)] = content.strip()

    return page_index


# ==========================================================
# Get Parent Page Text
# ==========================================================

def get_parent_page_text(paper_name, page_number):

    if paper_name not in _page_index_cache:

        _page_index_cache[paper_name] = _build_page_index(
            paper_name
        )

    return _page_index_cache[paper_name].get(page_number)


# ==========================================================
# Attach Parent Context To Retrieved Chunks
# ==========================================================

def attach_parent_context(chunks, max_parent_chars=MAX_PARENT_CHARS):

    for chunk in chunks:

        paper = chunk.get("paper_name")

        page = chunk.get("page")

        if paper is None or page is None:

            chunk["parent_text"] = chunk.get("chunk_text", "")

            continue

        parent_text = get_parent_page_text(
            paper,
            page
        )

        if parent_text is None:

            # Fall back to the chunk's own text if the parent
            # page can't be found (e.g. data/processed was
            # cleared after embeddings were generated).
            chunk["parent_text"] = chunk.get("chunk_text", "")

        else:

            chunk["parent_text"] = parent_text[:max_parent_chars]

    return chunks


# ==========================================================
# De-Duplicate Parents
# ==========================================================
# When multiple retrieved chunks come from the same page, we
# don't want to send that page's full text to the LLM more
# than once. This collapses to one entry per (paper, page).

def get_unique_parents(chunks):

    seen = set()

    unique = []

    for chunk in chunks:

        key = (
            chunk.get("paper_name"),
            chunk.get("page")
        )

        if key in seen:
            continue

        seen.add(key)

        unique.append(
            {
                "paper_name": chunk.get("paper_name"),
                "page": chunk.get("page"),
                "parent_text": chunk.get(
                    "parent_text",
                    chunk.get("chunk_text", "")
                )
            }
        )

    return unique


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    paper = input("Paper filename (e.g. RAG.pdf) : ").strip()

    page = int(input("Page number : ").strip())

    text = get_parent_page_text(paper, page)

    print()
    print("=" * 60)
    print("PARENT PAGE TEXT")
    print("=" * 60)

    if text is None:

        print("Not found.")

    else:

        print(text[:1500])