import os
from llm import generate_summary

# ==========================================================
# Folder
# ==========================================================

CHUNK_FOLDER = "embeddings"

# ==========================================================
# Summarize Papers
# ==========================================================

def summarize_papers():

    files = sorted(
        [
            file
            for file in os.listdir(CHUNK_FOLDER)
            if file.lower().endswith("_chunks.txt")
        ]
    )

    if not files:

        print("No chunk files found.")

        return []

    summaries = []

    print("=" * 60)
    print("ResearchGPT Paper Summarizer")
    print("=" * 60)

    for file in files:

        print(f"\nSummarizing : {file}")

        file_path = os.path.join(
            CHUNK_FOLDER,
            file
        )

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as f:

                text = f.read()

            if not text.strip():

                print("Skipped (Empty File)")

                continue

            # Limit input size for the LLM
            paper_text = text[:10000]

            summary = generate_summary(
                paper_text
            )

            summaries.append(
                {
                    "paper_name": os.path.splitext(file)[0].replace(
                        "_chunks",
                        ""
                    ) + ".pdf",

                    "summary": summary
                }
            )

            print("Completed")

        except Exception as e:

            print(f"Failed : {file}")
            print(f"Reason : {e}")

    print("\n")
    print("=" * 60)
    print("Summarization Completed")
    print("=" * 60)
    print(f"Total Papers : {len(summaries)}")

    return summaries


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    results = summarize_papers()

    print("\n")

    if len(results) == 0:

        print("No summaries generated.")

    else:

        for paper in results:

            print("=" * 80)
            print(paper["paper_name"])
            print("=" * 80)

            print(paper["summary"])

            print("\n")