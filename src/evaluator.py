import time
import numpy as np

# ==========================================================
# Timer Functions
# ==========================================================

def start_timer():

    return time.time()


def stop_timer(start_time):

    return round(

        time.time() - start_time,

        3

    )


# ==========================================================
# Similarity Score
# ==========================================================

def average_similarity(retrieved_chunks):

    if not retrieved_chunks:

        return 0.0

    similarities = []

    for chunk in retrieved_chunks:

        similarity = 1 / (1 + chunk["distance"])

        similarities.append(similarity)

    return round(

        float(np.mean(similarities)),

        4

    )


# ==========================================================
# Retrieval Efficiency
# ==========================================================

def retrieval_efficiency(

    retrieved_chunks,

    retrieval_time

):

    if retrieval_time <= 0:

        return 0.0

    efficiency = (

        len(retrieved_chunks)

        / retrieval_time

    )

    return round(

        efficiency,

        2

    )


# ==========================================================
# Recommendation Coverage
# ==========================================================

def recommendation_coverage(

    recommended_papers,

    retrieved_chunks

):

    if len(retrieved_chunks) == 0:

        return 0.0

    coverage = (

        len(recommended_papers)

        / len(retrieved_chunks)

    ) * 100

    return round(

        coverage,

        2

    )


# ==========================================================
# Overall Rating
# ==========================================================

def overall_rating(

    confidence,

    similarity,

    total_time

):

    score = 0

    if confidence >= 90:

        score += 40

    elif confidence >= 75:

        score += 30

    elif confidence >= 60:

        score += 20

    else:

        score += 10

    if similarity >= 0.90:

        score += 30

    elif similarity >= 0.80:

        score += 20

    elif similarity >= 0.70:

        score += 10

    if total_time <= 2:

        score += 30

    elif total_time <= 4:

        score += 20

    elif total_time <= 6:

        score += 10

    if score >= 90:

        return "Excellent"

    elif score >= 75:

        return "Very Good"

    elif score >= 60:

        return "Good"

    elif score >= 45:

        return "Average"

    return "Needs Improvement"
    # ==========================================================
# Evaluate System
# ==========================================================

def evaluate_system(

    retrieval_time,

    llm_time,

    retrieved_chunks,

    recommended_papers,

    confidence

):

    total_time = round(

        retrieval_time + llm_time,

        3

    )

    similarity = average_similarity(

        retrieved_chunks

    )

    efficiency = retrieval_efficiency(

        retrieved_chunks,

        retrieval_time

    )

    coverage = recommendation_coverage(

        recommended_papers,

        retrieved_chunks

    )

    rating = overall_rating(

        confidence["confidence"],

        similarity,

        total_time

    )

    result = {

        "retrieval_time": retrieval_time,

        "llm_time": llm_time,

        "total_time": total_time,

        "retrieved_chunks": len(

            retrieved_chunks

        ),

        "recommended_papers": len(

            recommended_papers

        ),

        "average_similarity": similarity,

        "retrieval_efficiency": efficiency,

        "coverage": coverage,

        "confidence": confidence["confidence"],

        "confidence_label": confidence["label"],

        "overall_rating": rating

    }

    return result


# ==========================================================
# Print Evaluation
# ==========================================================

def print_evaluation(result):

    print()

    print("=" * 70)

    print("RESEARCHGPT SYSTEM EVALUATION")

    print("=" * 70)

    print(

        f"Retrieval Time           : {result['retrieval_time']} sec"

    )

    print(

        f"LLM Time                 : {result['llm_time']} sec"

    )

    print(

        f"Total Response Time      : {result['total_time']} sec"

    )

    print(

        f"Retrieved Chunks         : {result['retrieved_chunks']}"

    )

    print(

        f"Recommended Papers       : {result['recommended_papers']}"

    )

    print(

        f"Average Similarity       : {result['average_similarity']}"

    )

    print(

        f"Retrieval Efficiency     : {result['retrieval_efficiency']} chunks/sec"

    )

    print(

        f"Recommendation Coverage  : {result['coverage']} %"

    )

    print(

        f"Confidence Score         : {result['confidence']} %"

    )

    print(

        f"Confidence Level         : {result['confidence_label']}"

    )

    print(

        f"Overall Rating           : {result['overall_rating']}"

    )
    # ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    retrieved = [

        {
            "distance": 0.91
        },

        {
            "distance": 0.95
        },

        {
            "distance": 1.02
        },

        {
            "distance": 0.88
        },

        {
            "distance": 0.97
        }

    ]

    recommended = [

        {
            "paper": "RAG.pdf",
            "score": 0.97
        },

        {
            "paper": "RePlug.pdf",
            "score": 0.94
        },

        {
            "paper": "FiD.pdf",
            "score": 0.91
        }

    ]

    confidence = {

        "confidence": 94.83,

        "label": "Very High"

    }

    retrieval_start = start_timer()

    time.sleep(0.25)

    retrieval_time = stop_timer(
        retrieval_start
    )

    llm_start = start_timer()

    time.sleep(1.42)

    llm_time = stop_timer(
        llm_start
    )

    result = evaluate_system(

        retrieval_time=retrieval_time,

        llm_time=llm_time,

        retrieved_chunks=retrieved,

        recommended_papers=recommended,

        confidence=confidence

    )

    print_evaluation(result)

    print()

    print("=" * 70)
    print("SYSTEM STATUS")
    print("=" * 70)

    print(
        f"System Performance : {result['overall_rating']}"
    )

    print(
        f"Average Similarity : {result['average_similarity']}"
    )

    print(
        f"Retrieval Speed    : {result['retrieval_efficiency']} chunks/sec"
    )

    print(
        f"Coverage           : {result['coverage']} %"
    )

    print(
        f"Confidence         : {result['confidence']} %"
    )

    print("=" * 70)