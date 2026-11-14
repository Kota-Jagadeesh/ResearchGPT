import re

from llm import call_groq

# ==========================================================
# Query Rewriting
# ==========================================================
# Distinct from query_expansion.py: expansion adds synonyms to
# widen RECALL for retrieval. This module resolves AMBIGUITY —
# vague pronouns, elliptical follow-up questions, or unclear
# scope — by rewriting the question into a clear, standalone
# one, optionally using recent conversation history for context.
#
# Example:
#   History: Q: "What is RAG?" A: "RAG combines retrieval..."
#   Current: "What are its limitations?"
#   Rewritten: "What are the limitations of Retrieval-Augmented
#               Generation (RAG)?"

_AMBIGUITY_INDICATORS = {

    "it", "this", "that", "these", "those", "they", "them",
    "its", "their", "such", "above", "aforementioned",
    "paper", "papers", "here", "summarize", "summary", "overview",
    "explain", "findings", "contributions", "about"

}


def _looks_potentially_ambiguous(question, target_papers=None):

    words = re.findall(
        r"\w+",
        question.lower()
    )

    if len(words) == 0:
        return False

    has_ambiguous_word = any(
        word in _AMBIGUITY_INDICATORS
        for word in words
    )

    is_very_short = len(words) <= 5

    # If target papers are specified and question refers to "paper" or is short, rewrite it
    has_target = bool(target_papers) and any(w in words for w in ["paper", "papers", "here", "summary", "summarize", "about", "models", "model"])

    return has_ambiguous_word or is_very_short or has_target


def _build_paper_query(question, target_papers):
    """
    Fast deterministic query formulation for document-level meta-questions
    (models, summary, authors, datasets) when target papers are active.
    Avoids LLM latency, stop-token truncation, and rate-limits.
    """
    clean_q = question.lower().strip()
    if not target_papers:
        return None

    try:
        from dense_retriever import get_paper_title
        primary_paper = target_papers[0]
        title = get_paper_title(primary_paper)
    except Exception:
        title = target_papers[0].replace(".pdf", "").replace("_", " ")

    if any(w in clean_q for w in ["model", "models", "architecture", "algorithm"]):
        return f"What AI models, neural networks, algorithms, and architectures are used in {title}?"
    if any(w in clean_q for w in ["author", "authors", "who wrote", "researcher", "researchers"]):
        return f"Who are the authors and researchers of {title}?"
    if any(w in clean_q for w in ["dataset", "datasets", "benchmark", "data used"]):
        return f"What dataset, benchmark, and data was used in {title}?"
    if any(w in clean_q for w in ["about", "summarize", "summary", "overview", "what is this paper", "discuss", "contributions", "objective"]):
        return f"What is the summary, abstract, methodology, and key contributions of {title}?"

    return None


# ==========================================================
# Rewrite Query
# ==========================================================

def rewrite_query(question, recent_history=None, target_papers=None):

    """
    Returns (rewritten_question, was_rewritten).

    recent_history: optional list of {"question": ..., "answer": ...}
    dicts, oldest first, used as context to resolve references like
    "it" / "that method" / "the previous one".

    target_papers: optional list of target paper filenames
    (e.g. ["Online_Exam_Proctoring_System.pdf"]) to resolve meta-queries
    like "what is this paper about" or "summarize the paper".
    """

    question = question.strip()

    # 1. Fast deterministic check for document-level meta-queries
    meta_q = _build_paper_query(question, target_papers)
    if meta_q:
        return meta_q, True

    if not _looks_potentially_ambiguous(question, target_papers=target_papers):

        return question, False

    history_context = ""

    if recent_history:

        for item in recent_history[-3:]:

            history_context += (

                f"Previous Question: {item['question']}\n"
                f"Previous Answer: {item['answer'][:300]}\n\n"

            )

    papers_context = ""
    target_title = None
    if target_papers:
        paper_descriptors = []
        for p in target_papers:
            if not p:
                continue
            try:
                from dense_retriever import get_paper_title
                t = get_paper_title(p)
                if not target_title:
                    target_title = t
                paper_descriptors.append(f"{p} (Title: \"{t}\")")
            except Exception:
                clean_p = p.replace(".pdf", "").replace("_", " ")
                if not target_title:
                    target_title = clean_p
                paper_descriptors.append(clean_p)

        if paper_descriptors:
            papers_context = f"Active / Filtered Research Paper(s): {', '.join(paper_descriptors)}\n"

    prompt = f"""
You are rewriting a question for an academic research paper RAG retrieval system.
The rewritten question will be used to retrieve relevant passages using dense semantic embeddings and BM25 search.

{papers_context}
{("Conversation so far:" + chr(10) + history_context) if history_context else "No prior conversation history is available."}

Current Question: {question}

Instructions:
1. If the current question is already highly specific, contains technical domain terms, and names the subject explicitly, return it unchanged.
2. If the question refers to "this paper", "the paper", "these papers", "here", or asks for a summary/overview/models/contributions without specifying the paper, rewrite it into a clear, standalone academic query that explicitly incorporates the active research paper title or subject (e.g., "What is the primary topic, abstract, and methodology of the Online Exam Proctoring System paper?").
3. If it relies on prior conversation context (pronouns like "it", "that model"), resolve them into specific terms using the conversation history.
4. Keep the question natural, focused, complete (never cut off), and concise (under 25 words). Do not invent false facts.

Return ONLY the rewritten search question. No explanation, no quotes, no prefix.
"""

    try:
        rewritten = call_groq(
            prompt,
            temperature=0.1,
            max_tokens=100
        ).strip().strip('"').strip("'")
    except Exception as e:
        rewritten = ""

    # Sanitize and guard against truncated or empty LLM outputs
    if (
        not rewritten
        or rewritten.endswith(",")
        or len(rewritten.split()) < 3
        or ("this paper" in rewritten.lower() and target_title)
    ):
        if target_title:
            q_clean = question.lower()
            if any(w in q_clean for w in ["model", "models", "architecture"]):
                return f"What AI models, neural network architectures, and algorithms are used in {target_title}?", True
            elif any(w in q_clean for w in ["author", "who wrote", "researcher"]):
                return f"Who are the authors and researchers of {target_title}?", True
            else:
                return f"What is the summary, abstract, methodology, and key contributions of {target_title}?", True
        return question, False

    if rewritten == "":

        return question, False

    was_rewritten = (

        rewritten.strip().lower()
        != question.strip().lower()

    )

    return rewritten, was_rewritten


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    history = [

        {
            "question": "What is Retrieval-Augmented Generation?",
            "answer": "RAG combines a retriever with a language model, "
                      "fetching relevant passages to condition generation on."
        }

    ]

    test_question = input("Question : ").strip()

    rewritten, was_rewritten = rewrite_query(

        test_question,

        recent_history=history

    )

    print()
    print("=" * 60)
    print("QUERY REWRITE")
    print("=" * 60)

    print(f"Original  : {test_question}")
    print(f"Rewritten : {rewritten}")
    print(f"Changed   : {was_rewritten}")