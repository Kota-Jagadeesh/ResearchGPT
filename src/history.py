import os
import json
from datetime import datetime

# ==========================================================
# Folder Configuration
# ==========================================================

HISTORY_FOLDER = "results"
HISTORY_FILE = os.path.join(HISTORY_FOLDER, "history.json")

os.makedirs(HISTORY_FOLDER, exist_ok=True)


# ==========================================================
# Load History
# ==========================================================

def load_history():

    if not os.path.exists(HISTORY_FILE):
        return []

    try:

        with open(HISTORY_FILE, "r", encoding="utf-8") as file:

            return json.load(file)

    except (json.JSONDecodeError, OSError):

        return []


# ==========================================================
# Save History
# ==========================================================

def save_history(question, answer, sources):

    history = load_history()

    history.append(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question.strip(),
            "answer": answer.strip(),
            "sources": sources
        }
    )

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
            ensure_ascii=False
        )


# ==========================================================
# Get History
# ==========================================================

def get_history():

    return load_history()


# ==========================================================
# Clear History
# ==========================================================

def clear_history():

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump([], file, indent=4)


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    save_history(
        "What is Retrieval-Augmented Generation?",
        "Retrieval-Augmented Generation combines retrieval with language generation.",
        [
            "RAG.pdf (Chunk 12)"
        ]
    )

    history = get_history()

    print("=" * 60)
    print("ResearchGPT History")
    print("=" * 60)

    for item in history:

        print(f"\nTime      : {item['time']}")
        print(f"Question  : {item['question']}")
        print(f"Answer    : {item['answer']}")
        print(f"Sources   : {item['sources']}")