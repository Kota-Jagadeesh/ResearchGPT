import pickle
import os

# ==========================================================
# Folder
# ==========================================================

VECTOR_FOLDER = "vector_db"

METADATA_FILE = os.path.join(
    VECTOR_FOLDER,
    "metadata.pkl"
)

# ==========================================================
# Lazy-Loaded Metadata
# ==========================================================
# Used to load at import time, which crashes anything that
# imports this module before papers are processed. Now it
# only loads the first time a filter function is called.

_metadata = None


def _load_metadata():

    global _metadata

    if _metadata is None:

        if not os.path.exists(METADATA_FILE):
            raise FileNotFoundError(
                f"Metadata file not found: {METADATA_FILE}. Please process papers first."
            )

        with open(
            METADATA_FILE,
            "rb"
        ) as file:

            _metadata = pickle.load(file)

    return _metadata


# ==========================================================
# Filter by Paper
# ==========================================================

def filter_by_paper(
    paper_name
):

    metadata = _load_metadata()

    results = []

    for item in metadata:

        if item["paper_name"].lower() == paper_name.lower():

            results.append(item)

    return results

# ==========================================================
# Filter by Page
# ==========================================================

def filter_by_page(
    page
):

    metadata = _load_metadata()

    results = []

    for item in metadata:

        if item["page"] == page:

            results.append(item)

    return results

# ==========================================================
# General Metadata Filter
# ==========================================================

def filter_metadata(

    paper_name=None,

    page=None

):

    metadata = _load_metadata()

    results = metadata

    if paper_name is not None:

        results = [

            item

            for item in results

            if item["paper_name"].lower()

            == paper_name.lower()

        ]

    if page is not None:

        results = [

            item

            for item in results

            if item["page"] == page

        ]

    return results

# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    print("="*60)
    print("METADATA FILTER")
    print("="*60)

    paper = input(
        "Paper Name (leave blank to skip): "
    ).strip()

    page = input(
        "Page Number (leave blank to skip): "
    ).strip()

    if page == "":
        page = None
    else:
        page = int(page)

    if paper == "":
        paper = None

    results = filter_metadata(
        paper_name=paper,
        page=page
    )

    print()

    print("="*60)
    print("RESULTS")
    print("="*60)

    print(f"Total Results : {len(results)}")

    for item in results[:10]:

        print()

        print(f"Paper : {item['paper_name']}")
        print(f"Page  : {item['page']}")
        print(f"Chunk : {item['chunk_id']}")

        print("-"*50)

        print(item["chunk_text"][:250])

        print("-"*50)