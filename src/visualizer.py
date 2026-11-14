import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from confidence import calibrate_cosine_similarity, calibrate_cross_score


FIGURES_DIR = os.path.join("results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set clean academic style defaults
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


def _sanitize_filename(text, max_len=30):
    """Sanitizes query text for safe filenames."""
    if not text:
        return "query"
    clean = re.sub(r"[^\w\s-]", "", text).strip().lower()
    clean = re.sub(r"[-\s]+", "_", clean)
    return clean[:max_len]


# ==========================================================
# Figure 1: Multi-Stage Retrieval & Reranking Scores
# ==========================================================

def plot_retrieval_relevance(retrieved_chunks, query=None, save_dir=FIGURES_DIR):
    """
    Publication-quality grouped horizontal bar chart comparing:
    - Calibrated Cross-Encoder Relevance (%)
    - Calibrated Dense Semantic Similarity (%)
    - Normalized BM25 Sparse Score (%)
    - Combined Chunk Confidence Score (%)
    """
    if not retrieved_chunks:
        return None, {}

    n = len(retrieved_chunks)
    chunk_labels = []
    cross_scores = []
    dense_scores = []
    bm25_scores = []
    combined_scores = []

    # Find max BM25 score for relative normalization
    max_bm25 = max(
        [float(c.get("bm25_score", 0.0) or 0.0) for c in retrieved_chunks] or [1.0]
    )
    if max_bm25 <= 0:
        max_bm25 = 1.0

    for i, c in enumerate(retrieved_chunks, start=1):
        paper = c.get("paper_name", "Paper")
        # Shorten paper name for neat y-axis labels
        short_paper = paper[:18] + ".." if len(paper) > 20 else paper
        page = c.get("page", "?")
        cid = c.get("chunk_id", i)
        label = f"#{i}: {short_paper}\n(p.{page}, ch.{cid})"
        chunk_labels.append(label)

        # Calibrated Cross-Encoder
        if "cross_score" in c and c["cross_score"] is not None:
            p_cross = calibrate_cross_score(c["cross_score"]) * 100.0
        else:
            p_cross = 0.0
        cross_scores.append(round(p_cross, 1))

        # Calibrated Dense
        dist = c.get("distance", 0.0)
        p_dense = calibrate_cosine_similarity(dist) * 100.0
        dense_scores.append(round(p_dense, 1))

        # Normalized BM25
        raw_bm25 = float(c.get("bm25_score", 0.0) or 0.0)
        norm_bm25 = (raw_bm25 / max_bm25) * 100.0 if max_bm25 > 0 else 0.0
        bm25_scores.append(round(norm_bm25, 1))

        # Combined
        if "cross_score" in c and c["cross_score"] is not None:
            comb = (0.75 * (p_cross / 100.0) + 0.25 * (p_dense / 100.0)) * 100.0
        else:
            comb = p_dense
        combined_scores.append(round(comb, 1))

    # Reverse order so #1 is at the top of the horizontal bar chart
    chunk_labels = chunk_labels[::-1]
    cross_scores = cross_scores[::-1]
    dense_scores = dense_scores[::-1]
    bm25_scores = bm25_scores[::-1]
    combined_scores = combined_scores[::-1]

    y = np.arange(n)
    bar_height = 0.20

    fig, ax = plt.subplots(figsize=(9, max(4.5, n * 1.1)), dpi=300)

    bars_ce = ax.barh(y + 1.5 * bar_height, cross_scores, bar_height, label="Cross-Encoder Relevance", color="#1f77b4", edgecolor="#0f4c81", alpha=0.9)
    bars_dense = ax.barh(y + 0.5 * bar_height, dense_scores, bar_height, label="Dense Similarity (Sentence-BERT)", color="#2ca02c", edgecolor="#1b661b", alpha=0.9)
    bars_bm25 = ax.barh(y - 0.5 * bar_height, bm25_scores, bar_height, label="Sparse Match (BM25 Keyword)", color="#ff7f0e", edgecolor="#b85600", alpha=0.9)
    bars_comb = ax.barh(y - 1.5 * bar_height, combined_scores, bar_height, label="Blended Chunk Score", color="#343a40", edgecolor="#111111", alpha=0.95)

    # Value labels on top of bars
    for bars in [bars_ce, bars_comb]:
        for bar in bars:
            width = bar.get_width()
            if width > 5:
                ax.text(width + 1.2, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%",
                        va="center", ha="left", fontsize=7.5, color="#222222", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(chunk_labels, fontsize=8.5)
    ax.set_xlabel("Relevance / Match Score (%)", fontweight="bold")
    ax.set_xlim(0, 108)

    title = "Fig 1: Multi-Stage Retrieval & Reranking Score Distribution"
    if query:
        short_q = query[:55] + "..." if len(query) > 55 else query
        ax.set_title(f"{title}\nQuery: \"{short_q}\"", fontsize=10.5, pad=12, fontweight="bold")
    else:
        ax.set_title(title, fontsize=11, pad=12, fontweight="bold")

    ax.axvline(40, color="#d9534f", linestyle=":", linewidth=1.2, label="Fallback Threshold (40%)")
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=8)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    saved_files = {}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        q_tag = _sanitize_filename(query)
        png_path = os.path.join(save_dir, f"fig1_retrieval_relevance_{q_tag}.png")
        pdf_path = os.path.join(save_dir, f"fig1_retrieval_relevance_{q_tag}.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        saved_files["png"] = png_path
        saved_files["pdf"] = pdf_path

    return fig, saved_files


# ==========================================================
# Figure 2: AGMR 2D Semantic Latent Space Projection (PCA)
# ==========================================================

def plot_agmr_clusters_2d(retrieved_chunks, candidate_pool=None, query_vector=None, query_text=None, save_dir=FIGURES_DIR):
    """
    2D PCA projection of chunk embeddings showing:
    - Candidate pool color-coded by Gaussian Mixture cluster
    - User query embedding vector (star marker)
    - Selected diverse passages highlighted with numbered badges
    """
    # Collect all available embeddings
    all_chunks = []
    if candidate_pool:
        all_chunks.extend(candidate_pool)
    for c in retrieved_chunks:
        if not any(c.get("chunk_id") == x.get("chunk_id") and c.get("paper_name") == x.get("paper_name") for x in all_chunks):
            all_chunks.append(c)

    chunks_with_emb = [c for c in all_chunks if "embedding" in c and c["embedding"] is not None]

    if len(chunks_with_emb) < 3:
        return None, {}

    embeddings = [np.asarray(c["embedding"], dtype=np.float32) for c in chunks_with_emb]

    has_query = query_vector is not None
    if has_query:
        embeddings.append(np.asarray(query_vector, dtype=np.float32))

    X = np.asarray(embeddings)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    if has_query:
        query_2d = X_2d[-1]
        chunk_coords = X_2d[:-1]
    else:
        query_2d = None
        chunk_coords = X_2d

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Color by cluster if cluster label exists
    clusters = [c.get("_cluster", 0) for c in chunks_with_emb]
    unique_clusters = sorted(list(set(clusters)))
    palette = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_clusters))))

    # Plot candidate pool
    for k in unique_clusters:
        idx = [i for i, cl in enumerate(clusters) if cl == k]
        ax.scatter(
            chunk_coords[idx, 0], chunk_coords[idx, 1],
            color=palette[k % 10], alpha=0.45, s=60, edgecolors="none",
            label=f"GMM Cluster {k + 1}"
        )

    # Highlight selected chunks
    selected_keys = {(c.get("paper_name"), c.get("chunk_id")) for c in retrieved_chunks}
    sel_num = 1
    for i, c in enumerate(chunks_with_emb):
        key = (c.get("paper_name"), c.get("chunk_id"))
        if key in selected_keys:
            x_pt, y_pt = chunk_coords[i, 0], chunk_coords[i, 1]
            ax.scatter(
                x_pt, y_pt,
                s=180, facecolors="none", edgecolors="#111111", linewidth=2.2, zorder=5
            )
            ax.annotate(
                f"#{sel_num}", (x_pt, y_pt),
                textcoords="offset points", xytext=(6, 6),
                fontsize=8.5, fontweight="bold", color="#111111",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", alpha=0.85, edgecolor="#cccccc")
            )
            sel_num += 1

    # Plot Query vector
    if query_2d is not None:
        ax.scatter(
            query_2d[0], query_2d[1],
            color="#d62728", s=280, marker="*", edgecolors="#800000",
            linewidth=1.5, zorder=10, label="Query Embedding"
        )
        ax.annotate(
            "Query Vector", (query_2d[0], query_2d[1]),
            textcoords="offset points", xytext=(8, -12),
            fontsize=9, fontweight="bold", color="#d62728"
        )

    var1 = pca.explained_variance_ratio_[0] * 100.0
    var2 = pca.explained_variance_ratio_[1] * 100.0
    ax.set_xlabel(f"Principal Component 1 ({var1:.1f}% variance)", fontweight="bold")
    ax.set_ylabel(f"Principal Component 2 ({var2:.1f}% variance)", fontweight="bold")

    title = "Fig 2: AGMR Semantic Manifold & Gaussian Mixture Diversity Selection"
    ax.set_title(title, fontsize=11, pad=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=7.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    saved_files = {}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        q_tag = _sanitize_filename(query_text)
        png_path = os.path.join(save_dir, f"fig2_agmr_latent_space_{q_tag}.png")
        pdf_path = os.path.join(save_dir, f"fig2_agmr_latent_space_{q_tag}.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        saved_files["png"] = png_path
        saved_files["pdf"] = pdf_path

    return fig, saved_files


# ==========================================================
# Figure 3: Multi-Metric RAG Radar Chart
# ==========================================================

def plot_performance_radar(confidence, groundedness, evaluation, query=None, save_dir=FIGURES_DIR):
    """
    Academic Radar (Spider) chart showing multi-metric RAG performance:
    - Overall Confidence (%)
    - Lexical Groundedness (%)
    - Dense Semantic Match (%)
    - Retrieval Speed Index (%)
    - Recommendation Alignment (%)
    Includes the 40% Fallback Threshold benchmark ring.
    """
    conf_val = float(confidence.get("confidence", 0.0) or 0.0) if confidence else 0.0
    ground_val = float(groundedness.get("groundedness", 0.0) or 0.0) if groundedness else 0.0

    avg_sim = float(evaluation.get("average_similarity", 0.0) or 0.0) if evaluation else 0.0
    # Map cosine similarity (0.1 to 0.6) to percentage (0 to 100)
    sim_pct = calibrate_cosine_similarity(avg_sim) * 100.0

    # Efficiency index (faster = higher percentage, 1s = 90%, 5s = 40%)
    tot_time = float(evaluation.get("total_time", 2.0) or 2.0) if evaluation else 2.0
    speed_idx = max(10.0, min(100.0, 100.0 - (tot_time * 12.0)))

    # Coverage / recommendation alignment
    cov_val = float(evaluation.get("coverage", 50.0) or 50.0) if evaluation else 50.0
    cov_val = max(10.0, min(100.0, cov_val))

    categories = [
        "Retrieval\nConfidence",
        "Answer\nGroundedness",
        "Semantic\nAlignment",
        "System Speed\nEfficiency",
        "Recommendation\nAlignment"
    ]

    values = [conf_val, ground_val, sim_pct, speed_idx, cov_val]

    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # Close the loop
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 6.5), subplot_kw=dict(polar=True), dpi=300)

    # Plot metric polygon
    ax.plot(angles_closed, values_closed, color="#1f77b4", linewidth=2.2, linestyle="solid", label="ResearchGPT Evaluation")
    ax.fill(angles_closed, values_closed, color="#1f77b4", alpha=0.25)

    # Plot Fallback Threshold ring (40%)
    threshold_vals = [40.0] * (num_vars + 1)
    ax.plot(angles_closed, threshold_vals, color="#d62728", linewidth=1.5, linestyle="--", label="Min Confidence Fallback (40%)")

    # Grid rings
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=7.5, color="#666666")

    # Spoke labels
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=9, fontweight="bold")

    # Point annotations
    for a, v in zip(angles, values):
        ax.plot(a, v, "o", color="#1f77b4", markersize=6)
        ax.text(a, v + 6, f"{v:.1f}%", ha="center", va="center", fontsize=8, fontweight="bold", color="#0f4c81")

    title = "Fig 3: Multi-Dimensional Verification Profile"
    ax.set_title(title, fontsize=11, pad=20, fontweight="bold")
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0.0), frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=8)

    plt.tight_layout()

    saved_files = {}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        q_tag = _sanitize_filename(query)
        png_path = os.path.join(save_dir, f"fig3_rag_verification_profile_{q_tag}.png")
        pdf_path = os.path.join(save_dir, f"fig3_rag_verification_profile_{q_tag}.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        saved_files["png"] = png_path
        saved_files["pdf"] = pdf_path

    return fig, saved_files


# ==========================================================
# Figure 4: Pipeline Latency Breakdown
# ==========================================================

def plot_latency_breakdown(retrieval_time, llm_time, query=None, save_dir=FIGURES_DIR):
    """
    Stacked horizontal bar chart showing latency breakdown:
    - Retrieval + Reranking stage (ms)
    - LLM Answer Generation stage (ms)
    """
    fig, ax = plt.subplots(figsize=(8, 2.5), dpi=300)

    r_sec = float(retrieval_time or 0.0)
    l_sec = float(llm_time or 0.0)
    tot = r_sec + l_sec

    r_pct = (r_sec / tot * 100.0) if tot > 0 else 0.0
    l_pct = (l_sec / tot * 100.0) if tot > 0 else 0.0

    ax.barh([0], [r_sec], color="#2ca02c", edgecolor="#1b661b", height=0.45, label=f"Retrieval & AGMR ({r_sec:.2f}s, {r_pct:.0f}%)")
    ax.barh([0], [l_sec], left=[r_sec], color="#1f77b4", edgecolor="#0f4c81", height=0.45, label=f"LLM Generation ({l_sec:.2f}s, {l_pct:.0f}%)")

    ax.text(r_sec / 2, 0, f"{r_sec:.2f}s", ha="center", va="center", color="#ffffff", fontweight="bold", fontsize=9)
    ax.text(r_sec + (l_sec / 2), 0, f"{l_sec:.2f}s", ha="center", va="center", color="#ffffff", fontweight="bold", fontsize=9)

    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)", fontweight="bold")
    ax.set_xlim(0, max(tot * 1.15, 1.0))
    ax.set_title(f"Fig 4: End-to-End Pipeline Latency (Total: {tot:.2f}s)", fontsize=11, pad=10, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout()

    saved_files = {}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        q_tag = _sanitize_filename(query)
        png_path = os.path.join(save_dir, f"fig4_pipeline_latency_{q_tag}.png")
        pdf_path = os.path.join(save_dir, f"fig4_pipeline_latency_{q_tag}.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        saved_files["png"] = png_path
        saved_files["pdf"] = pdf_path

    return fig, saved_files


# ==========================================================
# Generate All Figures Suite
# ==========================================================

def generate_paper_figures_suite(
    query,
    retrieved_chunks,
    confidence,
    groundedness,
    evaluation,
    candidate_pool=None,
    query_vector=None,
    save_dir=FIGURES_DIR
):
    """
    Generates all 4 publication-ready figures for a given Q&A turn.
    Returns a dictionary of figure objects and saved file paths (PNG and PDF).
    """
    figures = {}
    files = {}

    # 1. Retrieval relevance distribution
    fig1, f1_files = plot_retrieval_relevance(retrieved_chunks, query=query, save_dir=save_dir)
    if fig1:
        figures["retrieval_relevance"] = fig1
        files["retrieval_relevance"] = f1_files

    # 2. 2D AGMR PCA clustering
    fig2, f2_files = plot_agmr_clusters_2d(
        retrieved_chunks,
        candidate_pool=candidate_pool,
        query_vector=query_vector,
        query_text=query,
        save_dir=save_dir
    )
    if fig2:
        figures["agmr_latent_space"] = fig2
        files["agmr_latent_space"] = f2_files

    # 3. Performance Radar profile
    fig3, f3_files = plot_performance_radar(
        confidence=confidence,
        groundedness=groundedness,
        evaluation=evaluation,
        query=query,
        save_dir=save_dir
    )
    if fig3:
        figures["performance_radar"] = fig3
        files["performance_radar"] = f3_files

    # 4. Latency breakdown
    retrieval_time = evaluation.get("retrieval_time", 0.5) if evaluation else 0.5
    llm_time = evaluation.get("llm_time", 1.5) if evaluation else 1.5
    fig4, f4_files = plot_latency_breakdown(retrieval_time, llm_time, query=query, save_dir=save_dir)
    if fig4:
        figures["latency_breakdown"] = fig4
        files["latency_breakdown"] = f4_files

    return figures, files
