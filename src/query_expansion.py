import re

# ==========================================================
# Dictionary
# ==========================================================

EXPANSIONS = {

    "rag": [

        "retrieval augmented generation",

        "retrieval-augmented generation",

        "knowledge augmented generation",

        "retrieval based language model"

    ],

    "llm": [

        "large language model",

        "foundation model"

    ],

    "bert": [

        "bidirectional encoder representations from transformers"

    ],

    "faiss": [

        "facebook ai similarity search"

    ],

    "bm25": [

        "best matching 25"

    ],

    "embedding": [

        "vector embedding",

        "dense embedding"

    ]

}

# ==========================================================
# Expand Query
# ==========================================================

def expand_query(question):

    expanded = [question]

    tokens = re.findall(

        r"\w+",

        question.lower()

    )

    for token in tokens:

        if token in EXPANSIONS:

            expanded.extend(

                EXPANSIONS[token]

            )

    expanded = list(

        dict.fromkeys(expanded)

    )

    return expanded

# ==========================================================
# Build Final Query
# ==========================================================

def build_query(question):

    expanded = expand_query(

        question

    )

    return " ".join(expanded)

# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    question = input(

        "Question : "

    )

    print()

    print("="*60)

    print("QUERY EXPANSION")

    print("="*60)

    queries = expand_query(

        question

    )

    for i, q in enumerate(

        queries,

        start=1

    ):

        print(

            f"{i}. {q}"

        )

    print()

    print("="*60)

    print("FINAL QUERY")

    print("="*60)

    print(

        build_query(question)

    )