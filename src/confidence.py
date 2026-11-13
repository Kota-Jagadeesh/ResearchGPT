import numpy as np


# ==========================================================
# Normalize Similarity Score
# ==========================================================
# FAISS uses IndexFlatIP with L2-normalized embeddings.
# Therefore the value stored as "distance" is actually
# cosine similarity.
#
# Higher score = better match
# Lower score  = weaker match
#
# The score is clamped to [0, 1].

def normalize_distance(distance):

    return max(
        0.0,
        min(1.0, float(distance))
    )


# ==========================================================
# Calculate Confidence
# ==========================================================

def calculate_confidence(retrieved_chunks):

    if not retrieved_chunks:

        return {
            "confidence": 0.0,
            "label": "No Confidence",
            "scores": []
        }

    scores = []

    for chunk in retrieved_chunks:

        score = normalize_distance(
            chunk.get("distance", 0.0)
        )

        scores.append(score)

    # Average cosine similarity
    average = np.mean(scores)

    # Convert to percentage
    confidence = round(
        average * 100,
        2
    )

    if confidence >= 90:

        label = "Very High"

    elif confidence >= 75:

        label = "High"

    elif confidence >= 60:

        label = "Medium"

    elif confidence >= 40:

        label = "Low"

    else:

        label = "Very Low"

    return {
        "confidence": confidence,
        "label": label,
        "scores": scores
    }


# ==========================================================
# Print Confidence
# ==========================================================

def print_confidence(result):

    print()
    print("=" * 60)
    print("ANSWER CONFIDENCE")
    print("=" * 60)

    print(
        f"Confidence : {result['confidence']} %"
    )

    print(
        f"Level      : {result['label']}"
    )

    if result.get("scores"):

        print(
            f"Scores     : {result['scores']}"
        )


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    sample_chunks = [

        {"distance": 0.91},
        {"distance": 0.95},
        {"distance": 0.89},
        {"distance": 0.87},
        {"distance": 0.92}

    ]

    result = calculate_confidence(
        sample_chunks
    )

    print_confidence(result)