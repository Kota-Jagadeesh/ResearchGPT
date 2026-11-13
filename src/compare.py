from agmr import agmr_rerank
from llm import generate_rag_answer


# ==========================================================
# Multi-Paper Comparison
# ==========================================================

def compare_papers(

    question,

    retrieve_top_k=100,

    final_chunks=10

):

    print("=" * 70)
    print("ResearchGPT Multi-Paper Comparison")
    print("=" * 70)

    retrieved_chunks, confidence = agmr_rerank(

        question,

        retrieve_top_k=retrieve_top_k,

        num_clusters=5,

        final_chunks=final_chunks

    )

    if len(retrieved_chunks) == 0:

        return {

            "question": question,

            "answer": "No relevant information found.",

            "papers": [],

            "sources": [],

            "confidence": confidence

        }

    answer, sources, groundedness = generate_rag_answer(

        question,

        retrieved_chunks

    )

    papers = []

    paper_summary = {}

    for source in sources:

        paper = source["paper"]

        if paper not in papers:

            papers.append(paper)

        if paper not in paper_summary:

            paper_summary[paper] = {

                "chunks": []

            }

        paper_summary[paper]["chunks"].append(

            source["chunk"]

        )

    result = {

        "question": question,

        "answer": answer,

        "papers": papers,

        "paper_summary": paper_summary,

        "sources": sources,

        "confidence": confidence

    }

    return result
    # ==========================================================
# Print Comparison
# ==========================================================

def print_comparison(result):

    print("\n")
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(result["question"])

    print("\n")
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    print("\n")
    print("=" * 70)
    print("CONFIDENCE")
    print("=" * 70)

    print(
        f"Confidence : {result['confidence']['confidence']} %"
    )

    print(
        f"Level      : {result['confidence']['label']}"
    )

    print("\n")
    print("=" * 70)
    print("PAPERS USED")
    print("=" * 70)

    if len(result["papers"]) == 0:

        print("No Papers Found")

    else:

        for i, paper in enumerate(

            result["papers"],

            start=1

        ):

            chunks = result["paper_summary"][paper]["chunks"]

            print(

                f"{i}. {paper}"

            )

            print(

                f"   Chunks Used : {', '.join(map(str, chunks))}"

            )

    print("\n")
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)

    if len(result["sources"]) == 0:

        print("No Sources")

    else:

        for source in result["sources"]:

            print(

                f"{source['paper']} "

                f"(Chunk {source['chunk']})"

            )


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    question = input(

        "Enter Comparison Question : "

    ).strip()

    comparison = compare_papers(

        question

    )

    print_comparison(

        comparison

    )