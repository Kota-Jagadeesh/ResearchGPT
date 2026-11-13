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
    "its", "their", "such", "above", "aforementioned"

}


def _looks_potentially_ambiguous(question):

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

    is_very_short = len(words) <= 4

    return has_ambiguous_word or is_very_short


# ==========================================================
# Rewrite Query
# ==========================================================

def rewrite_query(question, recent_history=None):

    """
    Returns (rewritten_question, was_rewritten).

    recent_history: optional list of {"question": ..., "answer": ...}
    dicts, oldest first, used as context to resolve references like
    "it" / "that method" / "the previous one".
    """

    question = question.strip()

    if not _looks_potentially_ambiguous(question):

        return question, False

    history_context = ""

    if recent_history:

        for item in recent_history[-3:]:

            history_context += (

                f"Previous Question: {item['question']}\n"
                f"Previous Answer: {item['answer'][:300]}\n\n"

            )

    prompt = f"""
You are rewriting a potentially ambiguous or vague question so it
can be understood entirely on its own, without needing the
conversation history.

{("Conversation so far:" + chr(10) + history_context) if history_context else "No prior conversation history is available."}

Current Question: {question}

If the current question is already clear and specific on its own,
return it EXACTLY unchanged, word for word.

If it relies on prior context (uses "it", "that", "this method", is
very short, or is otherwise vague), rewrite it into a standalone,
specific question that captures the same intent, using the
conversation history above if it helps resolve the ambiguity. If
there is no history and the question is still ambiguous, make your
best reasonable interpretation and rewrite it to be more specific,
without inventing unrelated details.

Return ONLY the question itself. No explanation, no quotes, no
prefix like "Rewritten question:".
"""

    rewritten = call_groq(

        prompt,

        temperature=0.1,

        max_tokens=100

    ).strip().strip('"')

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