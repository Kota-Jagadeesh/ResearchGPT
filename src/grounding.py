import re

# ==========================================================
# Citation Grounding
# ==========================================================
# NOTE: This is a lightweight LEXICAL overlap check, not a
# semantic entailment model. It flags answers that contain a
# lot of content whose words don't appear anywhere in the
# retrieved context — a reasonable, cheap signal for "the
# model may be answering from its own knowledge instead of
# the papers" — but it is not a guarantee of factual
# correctness or a substitute for human review.

_STOPWORDS = {

    "the", "a", "an", "is", "are", "was", "were", "of", "to",
    "in", "on", "and", "or", "for", "with", "as", "by", "that",
    "this", "it", "its", "be", "which", "from", "at", "these",
    "those", "can", "will", "not", "has", "have", "had", "but",
    "if", "so", "than", "then", "there", "their", "they", "you",
    "your", "i", "we", "our", "also", "into", "such", "may",

}


def _tokenize(text):

    return set(

        re.findall(
            r"\w+",
            text.lower()
        )

    )


def _meaningful_tokens(text):

    tokens = _tokenize(text)

    return {

        token
        for token in tokens
        if token not in _STOPWORDS and len(token) > 2

    }


# ==========================================================
# Compute Groundedness
# ==========================================================

def compute_groundedness(answer, context_chunks_or_text):

    if not answer or not answer.strip():

        return {
            "groundedness": 0.0,
            "label": "No Answer"
        }

    if answer.strip() == "I could not find the answer in the uploaded papers.":

        return {
            "groundedness": 100.0,
            "label": "N/A (No Answer Given)"
        }

    meaningful_tokens = _meaningful_tokens(answer)

    if len(meaningful_tokens) == 0:

        return {
            "groundedness": 100.0,
            "label": "Grounded"
        }

    if isinstance(context_chunks_or_text, str):

        context_text = context_chunks_or_text

    else:

        context_text = " ".join(

            chunk.get(
                "parent_text",
                chunk.get("chunk_text", "")
            )

            for chunk in context_chunks_or_text

        )

    context_tokens = _tokenize(context_text)

    matched = meaningful_tokens & context_tokens

    ratio = len(matched) / len(meaningful_tokens)

    score = round(ratio * 100, 2)

    if score >= 80:
        label = "Well Grounded"

    elif score >= 60:
        label = "Mostly Grounded"

    elif score >= 40:
        label = "Partially Grounded"

    else:
        label = "Weakly Grounded / Possible Hallucination"

    return {
        "groundedness": score,
        "label": label
    }


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    context = """
    Retrieval-Augmented Generation combines a retriever with a
    language model. The retriever fetches relevant passages
    from an external knowledge base, and the language model
    conditions its generation on those passages.
    """

    good_answer = (
        "Retrieval-Augmented Generation combines a retriever "
        "with a language model, where the retriever fetches "
        "relevant passages that the model conditions on."
    )

    hallucinated_answer = (
        "Retrieval-Augmented Generation was invented in 2010 by "
        "a team at Stanford and uses quantum computing to "
        "search billions of documents in microseconds."
    )

    print("Good answer  :", compute_groundedness(good_answer, context))

    print("Bad answer   :", compute_groundedness(hallucinated_answer, context))