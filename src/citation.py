import textwrap
from datetime import datetime


# ==========================================================
# Format Sources
# ==========================================================

def format_sources(sources):

    formatted = []

    seen = set()

    if not sources:
        return formatted

    for source in sources:

        # --------------------------------------------------
        # CASE 1:
        # Source is a dictionary
        #
        # Example:
        # {
        #     "paper": "RAG.pdf",
        #     "page": 5,
        #     "chunk": 12
        # }
        # --------------------------------------------------

        if isinstance(source, dict):

            paper = source.get(
                "paper",
                source.get(
                    "paper_name",
                    "Unknown Paper"
                )
            )

            page = source.get(
                "page",
                "N/A"
            )

            chunk = source.get(
                "chunk",
                source.get(
                    "chunk_id",
                    "N/A"
                )
            )

            citation = (
                f"{paper} "
                f"(Page {page}, Chunk {chunk})"
            )

        # --------------------------------------------------
        # CASE 2:
        # Source is already a string
        #
        # Example:
        # "RAG.pdf (Page 5, Chunk 12)"
        # --------------------------------------------------

        elif isinstance(source, str):

            citation = source

        # --------------------------------------------------
        # CASE 3:
        # Unknown source type
        # --------------------------------------------------

        else:

            citation = str(source)

        # --------------------------------------------------
        # Remove duplicate citations
        # --------------------------------------------------

        if citation not in seen:

            formatted.append(citation)

            seen.add(citation)

    return formatted


# ==========================================================
# IEEE Citation
# ==========================================================

def ieee_citations(sources):

    citations = []

    year = datetime.now().year

    if not sources:
        return citations

    for i, source in enumerate(
        sources,
        start=1
    ):

        citations.append(
            f"[{i}] {source}. "
            f"ResearchGPT Retrieval, {year}."
        )

    return citations


# ==========================================================
# APA Citation
# ==========================================================

def apa_citations(sources):

    citations = []

    year = datetime.now().year

    if not sources:
        return citations

    for source in sources:

        citations.append(
            f"{source}. "
            f"({year}). "
            f"Retrieved using ResearchGPT."
        )

    return citations


# ==========================================================
# BibTeX Style
# ==========================================================

def bibtex_entries(sources):

    entries = []

    year = datetime.now().year

    if not sources:
        return entries

    for i, source in enumerate(
        sources,
        start=1
    ):

        entries.append(
f"""@misc{{paper{i},
  title={{ {source} }},
  author={{Unknown}},
  year={{ {year} }},
  note={{Retrieved using ResearchGPT}}
}}"""
        )

    return entries


# ==========================================================
# Pretty Print Sources
# ==========================================================

def print_sources(sources):

    print("\n")
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)

    if len(sources) == 0:

        print("No Sources Available")

        return

    for i, source in enumerate(
        sources,
        start=1
    ):

        print(
            f"{i}. {source}"
        )


# ==========================================================
# Build Citation Block
# ==========================================================

def build_citation_text(sources):

    if len(sources) == 0:

        return "No Sources"

    return "\n".join(
        f"[{i}] {source}"
        for i, source in enumerate(
            sources,
            start=1
        )
    )


# ==========================================================
# Citation Statistics
# ==========================================================

def citation_statistics(sources):

    if not sources:

        return {
            "total_sources": 0,
            "unique_sources": 0
        }

    return {
        "total_sources": len(sources),
        "unique_sources": len(set(sources))
    }


# ==========================================================
# Wrap Long Answer
# ==========================================================

def wrap_answer(
    answer,
    width=100
):

    return textwrap.fill(
        answer,
        width=width
    )


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    # ------------------------------------------------------
    # Test dictionary sources
    # ------------------------------------------------------

    demo_sources = [

        {
            "paper": "RAG.pdf",
            "page": 5,
            "chunk": 12
        },

        {
            "paper": "RePlug.pdf",
            "page": 3,
            "chunk": 5
        },

        {
            "paper": "RAG.pdf",
            "page": 5,
            "chunk": 12
        }

    ]

    # ------------------------------------------------------
    # Format sources
    # ------------------------------------------------------

    formatted = format_sources(
        demo_sources
    )

    print_sources(
        formatted
    )

    # ------------------------------------------------------
    # IEEE
    # ------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("IEEE CITATIONS")
    print("=" * 70)

    for citation in ieee_citations(
        formatted
    ):

        print(citation)

    # ------------------------------------------------------
    # APA
    # ------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("APA CITATIONS")
    print("=" * 70)

    for citation in apa_citations(
        formatted
    ):

        print(citation)

    # ------------------------------------------------------
    # BIBTEX
    # ------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("BIBTEX")
    print("=" * 70)

    for entry in bibtex_entries(
        formatted
    ):

        print(entry)

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    stats = citation_statistics(
        formatted
    )

    print("\n")
    print("=" * 70)
    print("STATISTICS")
    print("=" * 70)

    print(
        f"Total Sources : "
        f"{stats['total_sources']}"
    )

    print(
        f"Unique Sources : "
        f"{stats['unique_sources']}"
    )

    # ------------------------------------------------------
    # Wrapped Answer
    # ------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("WRAPPED ANSWER")
    print("=" * 70)

    print(
        wrap_answer(
            "Retrieval-Augmented Generation combines "
            "retrieval with language generation to "
            "improve factual correctness and reduce "
            "hallucinations in large language models."
        )
    )

    # ------------------------------------------------------
    # Test string sources
    # ------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STRING SOURCE TEST")
    print("=" * 70)

    string_sources = [
        "RAG.pdf (Page 5, Chunk 12)",
        "RePlug.pdf (Page 3, Chunk 5)",
        "RAG.pdf (Page 5, Chunk 12)"
    ]

    formatted_strings = format_sources(
        string_sources
    )

    print_sources(
        formatted_strings
    )

    print("\nCitation test completed successfully.")