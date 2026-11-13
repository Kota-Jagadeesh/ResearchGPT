import os
import json
import math
from datetime import datetime

# ==========================================================
# Anchor Working Directory To Project Root
# ==========================================================
# This script lives in src/, and uses bare relative paths like
# "data/eval/..." and "results". Anchoring to the project root
# (one level above this file's own folder) means it behaves
# the same regardless of which directory you run it from.

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

os.chdir(PROJECT_ROOT)

from dense_retriever import retrieve as dense_retrieve
from hybrid_retriever import hybrid_search
from cross_encoder import rerank as cross_encoder_rerank

# ==========================================================
# Configuration
# ==========================================================

TEST_CASES_FILE = os.path.join(
    "data",
    "eval",
    "golden_test_cases.json"
)

RESULTS_FOLDER = "results"

os.makedirs(
    RESULTS_FOLDER,
    exist_ok=True
)

TOP_K = 10

# "distance" is actually cosine similarity (higher = better
# match) — see dense_retriever.py / faiss_db.py. Adversarial
# cases (should_retrieve = False) are judged as "correctly
# handled" if the top result's similarity is LOW (at or below
# this threshold), meaning the system found nothing genuinely
# relevant — the right outcome for a question with no real
# answer in the papers. 0.35 is a starting point for
# all-MiniLM-L6-v2; genuinely unrelated text pairs typically
# score well below this, while topically related chunks score
# well above it. Tune empirically if adversarial accuracy looks
# consistently off against your own papers.
ADVERSARIAL_SIMILARITY_THRESHOLD = 0.35


# ==========================================================
# Load Test Cases
# ==========================================================

def load_test_cases(path=TEST_CASES_FILE):

    if not os.path.exists(path):

        raise FileNotFoundError(

            f"Test cases file not found: {path}. "
            f"Create it with real questions from your papers first."

        )

    with open(path, "r", encoding="utf-8") as file:

        cases = json.load(file)

    return cases


# ==========================================================
# Retrieval Strategies Under Test
# ==========================================================

def naive_dense_only(question, top_k=TOP_K):

    return dense_retrieve(
        question,
        top_k=top_k
    )


def hybrid_dense_sparse(question, top_k=TOP_K):

    results = hybrid_search(
        question,
        dense_top_k=max(top_k, 50),
        sparse_top_k=max(top_k, 50)
    )

    return results[:top_k]


def hybrid_plus_cross_encoder(question, top_k=TOP_K):

    candidates = hybrid_search(
        question,
        dense_top_k=100,
        sparse_top_k=100
    )

    if len(candidates) == 0:
        return []

    reranked = cross_encoder_rerank(
        question,
        candidates,
        top_k=top_k
    )

    return reranked


STRATEGIES = {

    "Naive (Dense Only)": naive_dense_only,

    "Hybrid (Dense + BM25)": hybrid_dense_sparse,

    "Hybrid + CrossEncoder (Full Pipeline)": hybrid_plus_cross_encoder

}


# ==========================================================
# Matching Logic
# ==========================================================

def is_match(result, case):

    expected_paper = case.get("expected_paper")

    expected_chunk_ids = case.get("expected_chunk_ids") or []

    if expected_paper is None or len(expected_chunk_ids) == 0:
        return False

    return (

        result.get("paper_name") == expected_paper

        and result.get("chunk_id") in expected_chunk_ids

    )


def find_rank(results, case):

    """Returns 1-indexed rank of the first matching result, or None if no match."""

    for rank, result in enumerate(results, start=1):

        if is_match(result, case):

            return rank

    return None


# ==========================================================
# Metrics
# ==========================================================

def reciprocal_rank(rank):

    if rank is None:
        return 0.0

    return 1.0 / rank


def ndcg_at_k(rank, k):

    """
    Binary relevance NDCG. If the (single) relevant item is
    found at position `rank`, DCG = 1 / log2(rank + 1).
    IDCG (best possible case, relevant item at rank 1) = 1.
    """

    if rank is None or rank > k:
        return 0.0

    dcg = 1.0 / math.log2(rank + 1)

    idcg = 1.0 / math.log2(1 + 1)

    return dcg / idcg


# ==========================================================
# Run Harness For One Strategy
# ==========================================================

def evaluate_strategy(strategy_name, strategy_fn, test_cases, top_k=TOP_K):

    positive_cases = [
        case for case in test_cases
        if case.get("should_retrieve", True)
    ]

    adversarial_cases = [
        case for case in test_cases
        if not case.get("should_retrieve", True)
    ]

    hits = 0

    reciprocal_ranks = []

    ndcg_scores = []

    per_case_results = []

    for i, case in enumerate(positive_cases, start=1):

        print(
            f"  [{i}/{len(positive_cases)}] {case['question'][:70]}",
            flush=True
        )

        results = strategy_fn(
            case["question"],
            top_k=top_k
        )

        rank = find_rank(results, case)

        hit = rank is not None

        if hit:
            hits += 1

        rr = reciprocal_rank(rank)

        ndcg = ndcg_at_k(rank, top_k)

        reciprocal_ranks.append(rr)

        ndcg_scores.append(ndcg)

        per_case_results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "hit": hit,
                "rank": rank,
                "reciprocal_rank": round(rr, 4),
                "ndcg": round(ndcg, 4)
            }
        )

    adversarial_correct = 0

    for case in adversarial_cases:

        results = strategy_fn(
            case["question"],
            top_k=top_k
        )

        # "Correctly handled" = either nothing came back, or the
        # top result's similarity is LOW enough to indicate a
        # genuinely weak/irrelevant match (nothing in the papers
        # actually answers this adversarial question).
        if len(results) == 0:

            correctly_handled = True

        else:

            top_similarity = results[0].get("distance", 0.0)

            correctly_handled = top_similarity <= ADVERSARIAL_SIMILARITY_THRESHOLD

        if correctly_handled:
            adversarial_correct += 1

        per_case_results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "adversarial": True,
                "correctly_handled": correctly_handled
            }
        )

    num_positive = len(positive_cases)

    num_adversarial = len(adversarial_cases)

    hit_rate = round(
        (hits / num_positive) * 100, 2
    ) if num_positive > 0 else None

    mrr = round(
        sum(reciprocal_ranks) / num_positive, 4
    ) if num_positive > 0 else None

    mean_ndcg = round(
        sum(ndcg_scores) / num_positive, 4
    ) if num_positive > 0 else None

    adversarial_accuracy = round(
        (adversarial_correct / num_adversarial) * 100, 2
    ) if num_adversarial > 0 else None

    return {

        "strategy": strategy_name,

        "num_positive_cases": num_positive,

        "num_adversarial_cases": num_adversarial,

        f"hit_rate_at_{top_k}": hit_rate,

        "mrr": mrr,

        f"ndcg_at_{top_k}": mean_ndcg,

        "adversarial_accuracy": adversarial_accuracy,

        "per_case_results": per_case_results

    }


# ==========================================================
# Run Full Harness
# ==========================================================

def run_harness(top_k=TOP_K):

    test_cases = load_test_cases()

    print("=" * 70)
    print("ResearchGPT Retrieval Eval Harness")
    print("=" * 70)
    print(f"Test Cases : {len(test_cases)}")
    print(f"Top K      : {top_k}")

    if len(test_cases) < 50:

        print(

            f"\n⚠ WARNING: Only {len(test_cases)} test cases loaded. "
            f"For a statistically meaningful evaluation, populate "
            f"{TEST_CASES_FILE} with 50+ real question/answer pairs "
            f"drawn from your actual uploaded papers."

        )

    all_results = []

    for strategy_name, strategy_fn in STRATEGIES.items():

        print(f"\nRunning strategy : {strategy_name}...")

        result = evaluate_strategy(

            strategy_name,

            strategy_fn,

            test_cases,

            top_k=top_k

        )

        all_results.append(result)

    print_comparison_table(all_results, top_k)

    save_report(all_results)

    return all_results


# ==========================================================
# Print Comparison Table
# ==========================================================

def print_comparison_table(all_results, top_k):

    print("\n")
    print("=" * 90)
    print("RETRIEVAL QUALITY COMPARISON")
    print("=" * 90)

    header = (
        f"{'Strategy':<40}"
        f"{'Hit@' + str(top_k):<12}"
        f"{'MRR':<10}"
        f"{'NDCG@' + str(top_k):<12}"
        f"{'Adversarial':<12}"
    )

    print(header)
    print("-" * 90)

    for result in all_results:

        hit_rate = result.get(f"hit_rate_at_{top_k}")

        ndcg = result.get(f"ndcg_at_{top_k}")

        row = (
            f"{result['strategy']:<40}"
            f"{(str(hit_rate) + '%') if hit_rate is not None else 'N/A':<12}"
            f"{(str(result['mrr'])) if result['mrr'] is not None else 'N/A':<10}"
            f"{(str(ndcg)) if ndcg is not None else 'N/A':<12}"
            f"{(str(result['adversarial_accuracy']) + '%') if result['adversarial_accuracy'] is not None else 'N/A':<12}"
        )

        print(row)

    print("=" * 90)

    # ------------------------------------------------------
    # Improvement vs naive baseline
    # ------------------------------------------------------

    naive = next(
        (r for r in all_results if r["strategy"] == "Naive (Dense Only)"),
        None
    )

    best = next(
        (r for r in all_results if r["strategy"] == "Hybrid + CrossEncoder (Full Pipeline)"),
        None
    )

    if naive and best and naive.get(f"hit_rate_at_{top_k}") is not None:

        naive_hit = naive[f"hit_rate_at_{top_k}"]

        best_hit = best[f"hit_rate_at_{top_k}"]

        improvement = round(best_hit - naive_hit, 2)

        print(

            f"\nHit Rate Improvement (Full Pipeline vs Naive Dense) : "
            f"{'+' if improvement >= 0 else ''}{improvement} percentage points"

        )


# ==========================================================
# Save Report
# ==========================================================

def save_report(all_results):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = os.path.join(

        RESULTS_FOLDER,

        f"eval_report_{timestamp}.json"

    )

    with open(report_path, "w", encoding="utf-8") as file:

        json.dump(

            all_results,

            file,

            indent=2,

            ensure_ascii=False

        )

    print(f"\nFull report saved to : {report_path}")


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    run_harness()