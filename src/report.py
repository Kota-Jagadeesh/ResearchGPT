import os
from datetime import datetime

# ==========================================================
# Folder
# ==========================================================

RESULT_FOLDER = "results"

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)

# ==========================================================
# Build Research Report
# ==========================================================

def build_report(
    question,
    answer,
    confidence,
    confidence_label,
    sources,
    recommended_papers
):

    report = []

    report.append("=" * 80)
    report.append("ResearchGPT RESEARCH REPORT")
    report.append("=" * 80)
    report.append("")

    report.append(
        f"Generated On : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    report.append("")

    report.append("=" * 80)
    report.append("QUESTION")
    report.append("=" * 80)
    report.append(question)

    report.append("")
    report.append("=" * 80)
    report.append("ANSWER")
    report.append("=" * 80)
    report.append(answer)

    report.append("")
    report.append("=" * 80)
    report.append("CONFIDENCE")
    report.append("=" * 80)

    report.append(
        f"Confidence Score : {confidence:.2f}%"
    )

    report.append(
        f"Confidence Level : {confidence_label}"
    )

    report.append("")

    report.append("=" * 80)
    report.append("SOURCES")
    report.append("=" * 80)

    if len(sources) == 0:

        report.append("No Sources Available")

    else:

        for i, source in enumerate(
            sources,
            start=1
        ):

            report.append(
                f"{i}. {source}"
            )

    report.append("")

    report.append("=" * 80)
    report.append("SIMILAR PAPER RECOMMENDATIONS")
    report.append("=" * 80)

    if len(recommended_papers) == 0:

        report.append(
            "No Similar Papers Found"
        )

    else:

        for i, paper in enumerate(
            recommended_papers,
            start=1
        ):

            report.append(
                f"{i}. {paper['paper']}"
            )

            report.append(
                f"   Similarity Score : {paper['score']:.4f}"
            )

    report.append("")

    report.append("=" * 80)
    report.append("REPORT SUMMARY")
    report.append("=" * 80)

    report.append(
        f"Total Sources Used : {len(sources)}"
    )

    report.append(
        f"Recommended Papers : {len(recommended_papers)}"
    )

    report.append(
        f"Overall Confidence : {confidence_label}"
    )

    report.append("")

    return "\n".join(report)
    # ==========================================================
# Save Report
# ==========================================================

def save_report(report):

    filename = os.path.join(

        RESULT_FOLDER,

        f"ResearchGPT_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    )

    with open(

        filename,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(report)

    return filename


# ==========================================================
# Generate Complete Report
# ==========================================================

def generate_report(

    question,

    answer,

    confidence,

    confidence_label,

    sources,

    recommended_papers

):

    report = build_report(

        question=question,

        answer=answer,

        confidence=confidence,

        confidence_label=confidence_label,

        sources=sources,

        recommended_papers=recommended_papers

    )

    return save_report(report)


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    report_path = generate_report(

        question="What is Retrieval-Augmented Generation?",

        answer=(
            "Retrieval-Augmented Generation combines "
            "retrieval with language generation to improve "
            "the factual correctness of LLM responses."
        ),

        confidence=94.62,

        confidence_label="Very High",

        sources=[

            "RAG.pdf (Chunk 12)",

            "RAG.pdf (Chunk 18)",

            "FiD.pdf (Chunk 7)"

        ],

        recommended_papers=[

            {

                "paper": "RAG.pdf",

                "score": 0.9784

            },

            {

                "paper": "RePlug.pdf",

                "score": 0.9458

            },

            {

                "paper": "FiD.pdf",

                "score": 0.9327

            }

        ]

    )

    print("\n")

    print("=" * 70)

    print("RESEARCH REPORT GENERATED")

    print("=" * 70)

    print(report_path)