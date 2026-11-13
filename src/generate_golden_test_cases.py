import os
import json
import random
import pickle

# ==========================================================
# Anchor Working Directory To Project Root
# ==========================================================
# This script lives in src/, and uses bare relative paths like
# "vector_db" and "data/eval/...". Anchoring to the project
# root (one level above this file's own folder) means it
# behaves the same regardless of which directory you run it
# from.

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

os.chdir(PROJECT_ROOT)

from llm import call_groq

# ==========================================================
# Config
# ==========================================================

VECTOR_FOLDER = "vector_db"

METADATA_FILE = os.path.join(
    VECTOR_FOLDER,
    "metadata.pkl"
)

OUTPUT_FILE = os.path.join(
    "data",
    "eval",
    "golden_test_cases.json"
)

NUM_POSITIVE_CASES = 50

MIN_CHUNK_LENGTH = 200

# Deliberately generic, unrelated-to-research-papers questions,
# used as the adversarial "should return nothing" test cases.
ADVERSARIAL_QUESTIONS = [
    "What is the population of Antarctica?",
    "Who won the 2022 FIFA World Cup final?",
    "What is the chemical formula for table salt?",
    "What year did the Berlin Wall fall?",
    "What is the top speed of a cheetah?",
    "What is the capital city of New Zealand?",
    "How many strings does a standard violin have?"
]


# ==========================================================
# Load Metadata
# ==========================================================

def load_metadata():

    if not os.path.exists(METADATA_FILE):

        raise FileNotFoundError(
            f"{METADATA_FILE} not found. Process at least one paper first "
            f"(click 'Process Papers' in the app)."
        )

    with open(METADATA_FILE, "rb") as file:

        return pickle.load(file)


# ==========================================================
# Sample Chunks (Spread Across Papers Where Possible)
# ==========================================================

def sample_chunks(metadata, num_cases):

    eligible = [

        chunk for chunk in metadata

        if len(chunk["chunk_text"]) >= MIN_CHUNK_LENGTH

    ]

    if len(eligible) == 0:

        raise ValueError(
            "No chunks long enough to generate questions from. "
            "Try lowering MIN_CHUNK_LENGTH."
        )

    random.shuffle(eligible)

    by_paper = {}

    for chunk in eligible:

        by_paper.setdefault(
            chunk["paper_name"],
            []
        ).append(chunk)

    papers = list(by_paper.keys())

    random.shuffle(papers)

    selected = []

    i = 0

    safety_limit = num_cases * 20

    while len(selected) < num_cases and i < safety_limit:

        paper = papers[i % len(papers)]

        if by_paper[paper]:

            selected.append(
                by_paper[paper].pop()
            )

        i += 1

    return selected[:num_cases]


# ==========================================================
# Generate a Question For a Chunk
# ==========================================================

def generate_question_for_chunk(chunk_text):

    prompt = f"""
You are creating a test question for a retrieval evaluation system.

Read the passage below and write ONE clear, specific,
self-contained factual question that can be answered using
ONLY this passage.

Do not reference "the passage" or "the text" in your question.
Do not include the answer.
Return ONLY the question, nothing else — no quotes, no preamble.

Passage:

{chunk_text}
"""

    question = call_groq(

        prompt,

        temperature=0.3,

        max_tokens=100

    )

    return question.strip().strip('"')


# ==========================================================
# Build Test Cases
# ==========================================================

def build_test_cases():

    metadata = load_metadata()

    print("=" * 60)
    print(f"Loaded {len(metadata)} chunks from {METADATA_FILE}")
    print("=" * 60)

    sampled_chunks = sample_chunks(
        metadata,
        NUM_POSITIVE_CASES
    )

    print(f"\nGenerating {len(sampled_chunks)} questions from your papers...\n")

    test_cases = []

    for i, chunk in enumerate(sampled_chunks, start=1):

        print(

            f"[{i}/{len(sampled_chunks)}] "
            f"{chunk['paper_name']} (chunk {chunk['chunk_id']})"

        )

        try:

            question = generate_question_for_chunk(

                chunk["chunk_text"]

            )

        except Exception as e:

            print(f"  Failed to generate question: {e}")

            continue

        print(f"  Q: {question}\n")

        test_cases.append(

            {

                "id": f"tc_{i:03d}",

                "question": question,

                "expected_paper": chunk["paper_name"],

                "expected_chunk_ids": [chunk["chunk_id"]],

                "should_retrieve": True

            }

        )

    for i, question in enumerate(ADVERSARIAL_QUESTIONS, start=1):

        test_cases.append(

            {

                "id": f"tc_adv_{i:03d}",

                "question": question,

                "expected_paper": None,

                "expected_chunk_ids": [],

                "should_retrieve": False

            }

        )

    os.makedirs(

        os.path.dirname(OUTPUT_FILE),

        exist_ok=True

    )

    with open(

        OUTPUT_FILE,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            test_cases,

            file,

            indent=2,

            ensure_ascii=False

        )

    print("=" * 60)
    print(f"Saved {len(test_cases)} test cases to {OUTPUT_FILE}")
    print(f"  Positive cases    : {len(sampled_chunks)}")
    print(f"  Adversarial cases : {len(ADVERSARIAL_QUESTIONS)}")
    print("=" * 60)
    print("\nNext step: python src/eval_harness.py")


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    build_test_cases()