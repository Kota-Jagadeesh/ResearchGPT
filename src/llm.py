import os
import time
from dotenv import load_dotenv
from groq import Groq

from parent_retriever import attach_parent_context, get_unique_parents
from grounding import compute_groundedness


# ==========================================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================================

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Please check your .env file."
    )


# ==========================================================
# GROQ CLIENT
# ==========================================================

# max_retries=0 is intentional.
# We handle rate-limit errors ourselves so that the application
# does not repeatedly retry and finally show
# "Maximum retry attempts exceeded."

client = Groq(
    api_key=API_KEY,
    max_retries=0
)


# ==========================================================
# WORKING MODEL
# ==========================================================

MODEL_NAME = "openai/gpt-oss-20b"


# ==========================================================
# OPTIONAL FALLBACK MODEL
# ==========================================================

# This model is used only if the primary model is temporarily
# rate limited and the fallback is available.

FALLBACK_MODEL = "openai/gpt-oss-120b"


# ==========================================================
# RETRY SETTINGS
# ==========================================================

MAX_RETRIES = 2


# ==========================================================
# GROQ REQUEST
# ==========================================================

def call_groq(
    prompt,
    temperature=0.2,
    max_tokens=700
):

    models_to_try = [
        MODEL_NAME,
        FALLBACK_MODEL
    ]

    last_error = None

    for model_index, model in enumerate(models_to_try):

        for attempt in range(MAX_RETRIES):

            try:

                print()
                print("=" * 60)
                print("Calling Groq")
                print("=" * 60)
                print("Model :", model)
                print("Attempt :", attempt + 1)

                response = client.chat.completions.create(

                    model=model,

                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    temperature=temperature,

                    max_tokens=max_tokens,

                    # GPT-OSS models can return reasoning separately.
                    # We only need the final answer.
                    include_reasoning=False

                )

                if not response.choices:

                    print("No choices returned by Groq.")

                    last_error = "No response choices returned."

                    continue

                content = response.choices[0].message.content

                if content is None:

                    print("Groq returned empty content.")

                    last_error = "Empty response returned by Groq."

                    continue

                content = content.strip()

                if not content:

                    print("Groq returned an empty answer.")

                    last_error = "Empty response returned by Groq."

                    continue

                print("Groq request successful.")

                return content

            except Exception as e:

                error_text = str(e)

                last_error = error_text

                print()
                print("Groq Error:")
                print(error_text)

                # --------------------------------------------------
                # RATE LIMIT / QUOTA
                # --------------------------------------------------

                if "429" in error_text:

                    print()
                    print("Groq rate limit or quota reached.")

                    # If this is the primary model, try the fallback
                    # model instead of repeatedly waiting.
                    if model_index == 0:

                        print()
                        print(
                            "Trying fallback model:",
                            FALLBACK_MODEL
                        )

                        break

                    # Fallback model also failed.
                    print()
                    print(
                        "Fallback model also reached a rate limit."
                    )

                    return (
                        "TEMPORARY LIMIT: The Groq API quota has "
                        "been reached. Please wait for the quota "
                        "to reset or use a higher Groq billing tier."
                    )

                # --------------------------------------------------
                # MODEL NOT FOUND
                # --------------------------------------------------

                if "404" in error_text or "model_not_found" in error_text:

                    print()
                    print("Model not available:", model)

                    # Try next model.
                    break

                # --------------------------------------------------
                # OTHER ERRORS
                # --------------------------------------------------

                if attempt < MAX_RETRIES - 1:

                    wait_time = 2 * (attempt + 1)

                    print(
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                else:

                    print(
                        "Request failed after retries."
                    )

        # Move to next model.

    # ==========================================================
    # FINAL ERROR
    # ==========================================================

    print()
    print("=" * 60)
    print("Groq request failed")
    print("=" * 60)

    if last_error:

        print(last_error)

    return (
        "TEMPORARY ERROR: The AI model could not generate "
        "a response at this time. Please try again."
    )


# ==========================================================
# GENERATE RAG ANSWER
# ==========================================================

def generate_rag_answer(
    question,
    retrieved_chunks,
    allow_general_knowledge=False
):

    GENERAL_KNOWLEDGE_MARKER = (
        "[GENERAL KNOWLEDGE — NOT FROM YOUR UPLOADED PAPERS]"
    )

    # ==========================================================
    # NO RETRIEVED CHUNKS
    # ==========================================================

    if len(retrieved_chunks) == 0:

        if allow_general_knowledge:

            prompt = f"""
You are ResearchGPT.

No relevant passages were found in the user's uploaded papers
for this question.

Answer using your general knowledge.

You MUST start your answer with this exact marker:

{GENERAL_KNOWLEDGE_MARKER}

Question:
{question}

Answer:
"""

            answer = call_groq(
                prompt,
                temperature=0.2,
                max_tokens=700
            )

            return (
                answer,
                [],
                {
                    "groundedness": 0.0,
                    "label": (
                        "General Knowledge "
                        "(Not Grounded In Papers)"
                    )
                }
            )

        return (
            "I could not find the answer in the uploaded papers.",
            [],
            {
                "groundedness": 100.0,
                "label": "N/A (No Answer Given)"
            }
        )

    # ==========================================================
    # ATTACH PARENT DOCUMENT CONTEXT
    # ==========================================================

    retrieved_chunks = attach_parent_context(
        retrieved_chunks
    )

    unique_parents = get_unique_parents(
        retrieved_chunks
    )

    # ==========================================================
    # BUILD CONTEXT
    # ==========================================================

    context = ""

    sources = []

    for parent in unique_parents:

        context += f"""
Paper : {parent['paper_name']}
Page  : {parent.get('page', 'N/A')}

{parent['parent_text']}

------------------------------------------------------------

"""

    # ==========================================================
    # BUILD SOURCES
    # ==========================================================

    for chunk in retrieved_chunks:

        sources.append(
            {
                "paper": chunk["paper_name"],
                "page": chunk.get("page", "N/A"),
                "chunk": chunk["chunk_id"]
            }
        )

    # ==========================================================
    # GROUNDING INSTRUCTIONS
    # ==========================================================

    if allow_general_knowledge:

        grounding_instructions = f"""

Answer the user's question primarily using the retrieved
research paper passages.

If the passages fully answer the question, answer using
ONLY the passages.

If multiple passages discuss the same topic, combine them
into one clear answer.

If the passages do NOT contain the answer, you may use
general knowledge.

However, any general knowledge MUST be clearly marked.

Start that portion with:

{GENERAL_KNOWLEDGE_MARKER}

Do not mix paper-grounded information and general knowledge
without clearly distinguishing them.
"""

    else:

        grounding_instructions = """

Answer the user's question ONLY using the retrieved
research paper passages.

If multiple passages discuss the same topic, combine
them into one clear answer.

Do not hallucinate.

If the answer cannot be found in the papers, reply:

I could not find the answer in the uploaded papers.
"""

    # ==========================================================
    # FINAL PROMPT
    # ==========================================================

    prompt = f"""
You are ResearchGPT, an AI research-paper assistant.

{grounding_instructions}

==========================
RESEARCH PAPERS
==========================

{context}

==========================
QUESTION
==========================

{question}

==========================
ANSWER
==========================
"""

    # ==========================================================
    # CALL LLM
    # ==========================================================

    answer = call_groq(
        prompt,
        temperature=0.1,
        max_tokens=700
    )

    # ==========================================================
    # GROUNDEDNESS
    # ==========================================================

    try:

        groundedness = compute_groundedness(
            answer,
            retrieved_chunks
        )

    except Exception as e:

        print()
        print("Groundedness calculation error:")
        print(e)

        groundedness = {
            "groundedness": 0.0,
            "label": "Unable to calculate"
        }

    # ==========================================================
    # RETURN
    # ==========================================================

    return (
        answer,
        sources,
        groundedness
    )


# ==========================================================
# GENERATE PAPER SUMMARY
# ==========================================================

def generate_summary(paper_text):

    if not paper_text:

        return "No paper text available."

    if not paper_text.strip():

        return "No paper text available."

    prompt = f"""
You are an expert research assistant.

Read the research paper text below and produce a clear,
structured academic summary.

Use exactly these sections:

1. Executive Summary

2. Key Contributions

3. Methodology

4. Results

5. Limitations

6. Future Work

Important instructions:

- Use only information available in the provided paper text.
- Do not invent results.
- Do not invent limitations.
- Do not invent future work.
- If a section is not available in the provided text,
  write "Not clearly stated in the provided paper text."
- Keep the summary concise but informative.
- Use bullet points where appropriate.

==========================
PAPER TEXT
==========================

{paper_text}

==========================
SUMMARY
==========================
"""

    summary = call_groq(
        prompt,
        temperature=0.2,
        max_tokens=800
    )

    return summary


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("ResearchGPT LLM Module")
    print("=" * 70)

    print()
    print("Using Groq model:", MODEL_NAME)

    print()
    print("Testing generate_summary()...")

    sample_text = """
Retrieval-Augmented Generation (RAG) combines
information retrieval with language generation.

The system retrieves relevant passages and provides
them as context to a language model.

This can improve factuality and reduce hallucination.
"""

    summary = generate_summary(
        sample_text
    )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(summary)

    print()
    print("=" * 70)
    print("LLM TEST COMPLETED")
    print("=" * 70)