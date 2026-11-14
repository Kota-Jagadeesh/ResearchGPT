import streamlit as st
import os
import sys
import time
import json
import glob


# ==========================================================
# Anchor Working Directory To Project Root
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

os.chdir(PROJECT_ROOT)


# ==========================================================
# Add src Folder
# ==========================================================

sys.path.append(
    os.path.join(
        PROJECT_ROOT,
        "src"
    )
)


from pipeline import run_pipeline
from agmr import agmr_rerank
from llm import generate_rag_answer
from summarizer import summarize_papers
from history import save_history, get_history
from web_fallback import web_search_fallback
from query_rewriter import rewrite_query

from recommender import recommend_similar_papers
from exporter import export_all
from report import generate_report
from evaluator import evaluate_system
from citation import format_sources
from visualizer import generate_paper_figures_suite
from dense_retriever import encode_query, reload_resources as reload_dense
from bm25_retriever import reload_resources as reload_bm25


# ==========================================================
# Folder Paths
# ==========================================================

RAW_FOLDER = "data/raw"
EMBEDDING_FOLDER = "embeddings"

os.makedirs(
    RAW_FOLDER,
    exist_ok=True
)

os.makedirs(
    EMBEDDING_FOLDER,
    exist_ok=True
)


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="ResearchGPT",
    page_icon="📚",
    layout="wide"
)


# ==========================================================
# Session State
# ==========================================================

if "answer" not in st.session_state:

    st.session_state.answer = ""


if "sources" not in st.session_state:

    st.session_state.sources = []


if "confidence" not in st.session_state:

    st.session_state.confidence = None


if "recommendations" not in st.session_state:

    st.session_state.recommendations = []


if "evaluation" not in st.session_state:

    st.session_state.evaluation = None


if "figures" not in st.session_state:

    st.session_state.figures = None


if "figure_files" not in st.session_state:

    st.session_state.figure_files = None


# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("📚 ResearchGPT")

st.sidebar.markdown("---")

st.sidebar.subheader(
    "Project Information"
)

st.sidebar.write(
    "LLM : openai/gpt-oss-20b (Groq)"
)

st.sidebar.write(
    "Embedding : Sentence-BERT"
)

st.sidebar.write(
    "Vector Database : FAISS"
)

st.sidebar.write(
    "Retriever : AGMR"
)

st.sidebar.markdown("---")

history = get_history()

st.sidebar.subheader(
    "Recent Questions"
)

if len(history) == 0:

    st.sidebar.info(
        "No History Available"
    )

else:

    for item in reversed(
        history[-10:]
    ):

        st.sidebar.write(
            "• " + item["question"]
        )


st.sidebar.markdown("---")

st.sidebar.subheader(
    "Project Features"
)

st.sidebar.success(
    "✓ Confidence Score"
)

st.sidebar.success(
    "✓ Similar Paper Recommendation"
)

st.sidebar.success(
    "✓ Research Report"
)

st.sidebar.success(
    "✓ Evaluation Dashboard"
)

st.sidebar.success(
    "✓ Download Answer"
)

st.sidebar.success(
    "✓ Search History"
)


# ==========================================================
# Title
# ==========================================================

st.title(
    "📚 ResearchGPT"
)

st.caption(
    "LLM-Based Research Paper Question Answering using Retrieval-Augmented Generation"
)

st.markdown("---")


# ==========================================================
# Statistics
# ==========================================================

paper_count = len(
    [
        file
        for file in os.listdir(RAW_FOLDER)
        if file.endswith(".pdf")
    ]
)


chunk_count = len(
    [
        file
        for file in os.listdir(EMBEDDING_FOLDER)
        if file.endswith("_chunks.txt")
    ]
)


c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        "Research Papers",
        paper_count
    )


with c2:

    st.metric(
        "Processed Papers",
        chunk_count
    )


with c3:

    st.metric(
        "Retriever",
        "AGMR"
    )


st.markdown("---")


# ==========================================================
# Upload Papers
# ==========================================================

uploaded_files = st.file_uploader(
    "Upload Research Papers",
    type=["pdf"],
    accept_multiple_files=True
)


new_uploads = []


if uploaded_files:

    for pdf in uploaded_files:

        save_path = os.path.join(
            RAW_FOLDER,
            pdf.name
        )

        if os.path.exists(
            save_path
        ):

            continue

        with open(
            save_path,
            "wb"
        ) as file:

            file.write(
                pdf.getbuffer()
            )

        new_uploads.append(
            pdf.name
        )


    if len(new_uploads):

        st.success(
            f"{len(new_uploads)} new paper(s) uploaded."
        )


raw_pdf_names = {f for f in os.listdir(RAW_FOLDER) if f.endswith(".pdf")}
chunk_pdf_names = {
    f.replace("_chunks.txt", "")
    for f in os.listdir(EMBEDDING_FOLDER)
    if f.endswith("_chunks.txt")
}
unprocessed_papers = sorted(list(raw_pdf_names - chunk_pdf_names))

if unprocessed_papers:

    st.warning(
        f"⚠️ **{len(unprocessed_papers)} uploaded paper(s) are not indexed yet:** "
        f"{', '.join(unprocessed_papers)}. "
        f"Please click **'⚙ Process Papers'** below to index them before asking questions!"
    )


# ==========================================================
# Buttons
# ==========================================================

left, right = st.columns(2)


with left:

    process = st.button(
        "⚙ Process Papers",
        use_container_width=True
    )


with right:

    summarize = st.button(
        "📝 Summarize Papers",
        use_container_width=True
    )


# ==========================================================
# Process Papers
# ==========================================================

if process:

    if paper_count == 0:

        st.error(
            "Upload at least one research paper."
        )

    else:

        progress = st.progress(
            0
        )

        status = st.empty()

        try:

            status.info(
                "Running ResearchGPT Pipeline..."
            )

            progress.progress(
                20
            )

            run_pipeline()
            reload_dense()
            reload_bm25()

            progress.progress(
                100
            )

            status.success(
                "Pipeline Completed Successfully."
            )

            time.sleep(
                1
            )

            st.rerun()

        except Exception as e:

            status.error(
                str(e)
            )


# ==========================================================
# Summarize Papers
# ==========================================================

if summarize:

    if chunk_count == 0:

        st.warning(
            "Please process papers first."
        )

    else:

        with st.spinner(
            "Generating Paper Summaries..."
        ):

            summaries = summarize_papers()

        st.success(
            "Summary Generated Successfully."
        )

        for paper in summaries:

            with st.expander(
                paper["paper_name"]
            ):

                st.write(
                    paper["summary"]
                )


st.markdown("---")


# ==========================================================
# Question Section
# ==========================================================

st.subheader(
    "Ask a Research Question"
)


# ==========================================================
# Metadata Filtering
# ==========================================================

available_papers = sorted(
    [
        file
        for file in os.listdir(RAW_FOLDER)
        if file.endswith(".pdf")
    ]
)


selected_papers = st.multiselect(
    "Filter by Paper (optional — leave empty to search all papers)",
    options=available_papers,
    default=[]
)


allow_general_knowledge = st.checkbox(
    "Allow general knowledge if the answer isn't in your papers "
    "(clearly labeled, not verified against your papers)",
    value=False
)


question = st.text_area(
    "Question",
    height=150,
    placeholder="Example: Explain Retrieval-Augmented Generation."
)


ask = st.button(
    "🚀 Get Answer",
    use_container_width=True
)


# ==========================================================
# Question Answering
# ==========================================================

if ask:

    if question.strip() == "":

        st.warning(
            "Please enter a question."
        )

    elif paper_count == 0:

        st.warning(
            "Please upload at least one research paper first."
        )

    else:

        try:

            # ------------------------------------------------
            # Auto-Sync Index for Newly Uploaded Papers
            # ------------------------------------------------
            raw_pdf_names = {f for f in os.listdir(RAW_FOLDER) if f.endswith(".pdf")}
            chunk_pdf_names = {
                f.replace("_chunks.txt", "")
                for f in os.listdir(EMBEDDING_FOLDER)
                if f.endswith("_chunks.txt")
            }
            unprocessed = sorted(list(raw_pdf_names - chunk_pdf_names))

            if unprocessed or chunk_count == 0:
                with st.spinner(
                    f"Indexing {len(unprocessed)} newly uploaded paper(s) before answering..."
                ):
                    run_pipeline()
                    reload_dense()
                    reload_bm25()

            # ------------------------------------------------
            # Query Rewriting
            # ------------------------------------------------

            recent_history = get_history()[-3:]

            target_papers = selected_papers if selected_papers else available_papers[:2]

            resolved_question, was_rewritten = rewrite_query(
                question,
                recent_history=recent_history,
                target_papers=target_papers
            )


            if was_rewritten:

                st.caption(
                    f"🔎 Interpreted as: *{resolved_question}*"
                )


            # ------------------------------------------------
            # Retrieval
            # ------------------------------------------------

            retrieval_start = time.time()


            with st.spinner(
                "Searching Research Papers..."
            ):

                retrieved_chunks, confidence = agmr_rerank(
                    resolved_question,
                    paper_filter=(
                        selected_papers
                        if selected_papers
                        else None
                    )
                )


            retrieval_time = round(
                time.time() - retrieval_start,
                3
            )


            # ------------------------------------------------
            # LLM Answer Generation
            # ------------------------------------------------

            llm_start = time.time()


            with st.spinner(
                "Generating Answer..."
            ):

                answer, sources, groundedness = generate_rag_answer(
                    resolved_question,
                    retrieved_chunks,
                    allow_general_knowledge=allow_general_knowledge
                )


            llm_time = round(
                time.time() - llm_start,
                3
            )


            # ------------------------------------------------
            # Recommendations
            # ------------------------------------------------

            recommendations = recommend_similar_papers(
                resolved_question
            )


            # ------------------------------------------------
            # Format Sources
            # ------------------------------------------------

            formatted_sources = format_sources(
                sources
            )


            # ------------------------------------------------
            # Save History
            # ------------------------------------------------

            save_history(
                question,
                answer,
                formatted_sources
            )


            # ------------------------------------------------
            # Evaluation
            # ------------------------------------------------

            evaluation = evaluate_system(
                retrieval_time=retrieval_time,
                llm_time=llm_time,
                retrieved_chunks=retrieved_chunks,
                recommended_papers=recommendations,
                confidence=confidence
            )


            # ==================================================
            # WEB SEARCH FALLBACK
            # ==================================================
            #
            # Confidence is calculated by confidence.py.
            #
            # Only if the calculated confidence is BELOW 40%,
            # the system performs a supplementary web search.
            #
            # High-confidence paper retrieval does NOT trigger
            # web fallback.
            #
            # ==================================================

            LOW_CONFIDENCE_THRESHOLD = 30


            web_fallback_result = None


            # Only trigger web fallback if confidence is truly very low AND no paper chunks were retrieved
            if confidence["confidence"] < LOW_CONFIDENCE_THRESHOLD and len(retrieved_chunks) == 0:

                with st.spinner(
                    "Checking the web for supplementary context..."
                ):

                    web_fallback_result = web_search_fallback(
                        question
                    )


            # ------------------------------------------------
            # Generate Figures for Research Paper
            # ------------------------------------------------

            paper_figures = {}
            paper_fig_files = {}

            with st.spinner(
                "Generating Publication-Ready Figures..."
            ):
                try:
                    q_vec = encode_query(resolved_question)
                    candidates_pool = confidence.get("candidates_pool") or retrieved_chunks
                    paper_figures, paper_fig_files = generate_paper_figures_suite(
                        query=resolved_question,
                        retrieved_chunks=retrieved_chunks,
                        confidence=confidence,
                        groundedness=groundedness,
                        evaluation=evaluation,
                        candidate_pool=candidates_pool,
                        query_vector=q_vec
                    )
                except Exception as fig_err:
                    print(f"Error generating figures: {fig_err}")


            # ------------------------------------------------
            # Session State
            # ------------------------------------------------

            st.session_state.answer = answer

            st.session_state.sources = formatted_sources

            st.session_state.confidence = confidence

            st.session_state.recommendations = recommendations

            st.session_state.evaluation = evaluation

            st.session_state.figures = paper_figures

            st.session_state.figure_files = paper_fig_files


            # ------------------------------------------------
            # Answer
            # ------------------------------------------------

            st.markdown("---")

            st.subheader(
                "📖 Answer"
            )


            if "[GENERAL KNOWLEDGE" in answer.upper():

                st.warning(
                    "⚠ Part or all of this answer comes from general knowledge, "
                    "NOT your uploaded papers — it has not been verified against "
                    "your papers and may be incomplete or inaccurate."
                )


            st.success(
                answer
            )


            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            c1, c2, c3 = st.columns(3)


            with c1:

                st.metric(
                    "Confidence",
                    f"{confidence['confidence']} %"
                )


            with c2:

                st.metric(
                    "Level",
                    confidence["label"]
                )


            with c3:

                st.metric(
                    "Groundedness",
                    f"{groundedness['groundedness']} %",
                    help=groundedness["label"]
                )


            # ==================================================
            # Web Fallback Results
            # ==================================================

            if web_fallback_result is not None:

                st.markdown("---")

                with st.expander(
                    "🌐 Supplementary Web Search Results",
                    expanded=False
                ):

                    st.caption(
                        "Supplementary results from the web (not verified against your papers):"
                    )

                    if web_fallback_result.get("error"):

                        st.info(
                            "Web search unavailable: "
                            + str(web_fallback_result["error"])
                        )

                    elif len(
                        web_fallback_result.get("results", [])
                    ) == 0:

                        st.info(
                            "No web results found."
                        )

                    else:

                        for item in web_fallback_result["results"]:

                            with st.container():

                                st.write(
                                    f"**[{item['title']}]({item['url']})**"
                                )

                                st.caption(
                                    item["snippet"]
                                )


            # ==================================================
            # Sources
            # ==================================================

            st.markdown("---")

            st.subheader(
                "📚 Sources"
            )


            if len(formatted_sources) == 0:

                st.info(
                    "No Sources Found."
                )


            else:

                for source in formatted_sources:

                    st.write(
                        "• " + source
                    )


            # ==================================================
            # Similar Paper Recommendations
            # ==================================================

            st.markdown("---")

            st.subheader(
                "📄 Similar Paper Recommendations"
            )


            if len(recommendations) == 0:

                st.info(
                    "No similar papers found."
                )


            else:

                for i, paper in enumerate(
                    recommendations[:10],
                    start=1
                ):

                    with st.container():

                        col1, col2 = st.columns(
                            [4, 1]
                        )


                        with col1:

                            st.write(
                                f"**{i}. {paper['paper']}**"
                            )


                        with col2:

                            st.write(
                                f"{paper['score']:.4f}"
                            )


            # ==================================================
            # Evaluation
            # ==================================================

            st.markdown("---")

            st.subheader(
                "📊 Evaluation"
            )


            e1, e2, e3 = st.columns(3)


            with e1:

                st.metric(
                    "Retrieval Time",
                    f"{evaluation['retrieval_time']} sec"
                )


            with e2:

                st.metric(
                    "LLM Time",
                    f"{evaluation['llm_time']} sec"
                )


            with e3:

                st.metric(
                    "Total Time",
                    f"{evaluation['total_time']} sec"
                )


            e4, e5, e6 = st.columns(3)


            with e4:

                st.metric(
                    "Similarity",
                    evaluation["average_similarity"]
                )


            with e5:

                st.metric(
                    "Retrieved Chunks",
                    evaluation["retrieved_chunks"]
                )


            with e6:

                st.metric(
                    "Overall Rating",
                    evaluation["overall_rating"]
                )


            # ==================================================
            # Research Paper Visualizations & Metrics
            # ==================================================

            st.markdown("---")

            st.subheader(
                "📊 Research Paper Visualizations & Comparative Metrics"
            )

            st.caption(
                "Publication-ready figures (300 DPI PNG & vector PDF) with LaTeX-ready formatting and descriptive captions for direct inclusion in academic publications."
            )

            if st.session_state.figures:

                tab1, tab2, tab3, tab4 = st.tabs([
                    "📈 Fig 1: Retrieval Relevance",
                    "🔬 Fig 2: AGMR Latent Space",
                    "🎯 Fig 3: Verification Profile",
                    "⏱ Fig 4: Latency Breakdown"
                ])

                # --- Tab 1: Retrieval Relevance ---
                with tab1:
                    if "retrieval_relevance" in st.session_state.figures and st.session_state.figures["retrieval_relevance"]:
                        st.pyplot(st.session_state.figures["retrieval_relevance"], use_container_width=True)
                        st.markdown(
                            "**Figure 1:** *Multi-stage retrieval and reranking scores for top candidate chunks.* "
                            "Shows calibrated Cross-Encoder relevance logits, Sentence-BERT dense semantic similarity, "
                            "normalized BM25 keyword matching, and the final AGMR blended score across the retrieved hierarchy."
                        )
                        f1_files = st.session_state.figure_files.get("retrieval_relevance", {}) if st.session_state.figure_files else {}
                        d1_col1, d1_col2 = st.columns(2)
                        with d1_col1:
                            if "png" in f1_files and os.path.exists(f1_files["png"]):
                                with open(f1_files["png"], "rb") as f:
                                    st.download_button(
                                        "⬇ Download Fig 1 (300 DPI PNG)",
                                        data=f.read(),
                                        file_name=os.path.basename(f1_files["png"]),
                                        mime="image/png",
                                        key="dl_fig1_png",
                                        use_container_width=True
                                    )
                        with d1_col2:
                            if "pdf" in f1_files and os.path.exists(f1_files["pdf"]):
                                with open(f1_files["pdf"], "rb") as f:
                                    st.download_button(
                                        "⬇ Download Fig 1 (Vector PDF - LaTeX)",
                                        data=f.read(),
                                        file_name=os.path.basename(f1_files["pdf"]),
                                        mime="application/pdf",
                                        key="dl_fig1_pdf",
                                        use_container_width=True
                                    )
                        with st.expander("📋 LaTeX Snippet for Figure 1"):
                            pdf_name = os.path.basename(f1_files.get("pdf", "fig1_retrieval_relevance.pdf"))
                            st.code(
                                "\\begin{figure}[htbp]\n"
                                "  \\centering\n"
                                f"  \\includegraphics[width=0.9\\linewidth]{{{pdf_name}}}\n"
                                "  \\caption{Multi-stage retrieval and reranking scores for candidate chunks, comparing Cross-Encoder, Dense Semantic, BM25, and AGMR Blended scores.}\n"
                                "  \\label{fig:retrieval_relevance}\n"
                                "\\end{figure}",
                                language="latex"
                            )

                # --- Tab 2: AGMR Latent Space (PCA) ---
                with tab2:
                    if "agmr_latent_space" in st.session_state.figures and st.session_state.figures["agmr_latent_space"]:
                        st.pyplot(st.session_state.figures["agmr_latent_space"], use_container_width=True)
                        st.markdown(
                            "**Figure 2:** *Two-dimensional PCA projection of candidate passage embeddings under AGMR.* "
                            "Visualizes Gaussian Mixture Model (GMM) semantic clustering, query vector embedding position (red star), "
                            "and final diverse passages selected by AGMR (#1 through #k) spanning the candidate manifold."
                        )
                        f2_files = st.session_state.figure_files.get("agmr_latent_space", {}) if st.session_state.figure_files else {}
                        d2_col1, d2_col2 = st.columns(2)
                        with d2_col1:
                            if "png" in f2_files and os.path.exists(f2_files["png"]):
                                with open(f2_files["png"], "rb") as f:
                                    st.download_button(
                                        "⬇ Download Fig 2 (300 DPI PNG)",
                                        data=f.read(),
                                        file_name=os.path.basename(f2_files["png"]),
                                        mime="image/png",
                                        key="dl_fig2_png",
                                        use_container_width=True
                                    )
                        with d2_col2:
                            if "pdf" in f2_files and os.path.exists(f2_files["pdf"]):
                                with open(f2_files["pdf"], "rb") as f:
                                    st.download_button(
                                        "⬇ Download Fig 2 (Vector PDF - LaTeX)",
                                        data=f.read(),
                                        file_name=os.path.basename(f2_files["pdf"]),
                                        mime="application/pdf",
                                        key="dl_fig2_pdf",
                                        use_container_width=True
                                    )
                        with st.expander("📋 LaTeX Snippet for Figure 2"):
                            pdf_name = os.path.basename(f2_files.get("pdf", "fig2_agmr_latent_space.pdf"))
                            st.code(
                                "\\begin{figure}[htbp]\n"
                                "  \\centering\n"
                                f"  \\includegraphics[width=0.85\\linewidth]{{{pdf_name}}}\n"
                                "  \\caption{2D PCA projection of candidate passage embeddings under AGMR clustering, illustrating semantic diversity and query alignment.}\n"
                                "  \\label{fig:agmr_latent_space}\n"
                                "\\end{figure}",
                                language="latex"
                            )

                # --- Tab 3: RAG Verification Profile ---
                with tab3:
                    if "performance_radar" in st.session_state.figures and st.session_state.figures["performance_radar"]:
                        st.pyplot(st.session_state.figures["performance_radar"], use_container_width=True)
                        st.markdown(
                            "**Figure 3:** *Multi-metric polar evaluation profile of the generated response.* "
                            "Assesses Retrieval Confidence, Answer Groundedness, Semantic Alignment, System Speed Efficiency, "
                            "and Recommendation Alignment against the 40% Web Fallback Decision Threshold."
                        )
                        f3_files = st.session_state.figure_files.get("performance_radar", {}) if st.session_state.figure_files else {}
                        d3_col1, d3_col2 = st.columns(2)
                        with d3_col1:
                            if "png" in f3_files and os.path.exists(f3_files["png"]):
                                with open(f3_files["png"], "rb") as f:
                                    st.download_button(
                                        "⬇ Download Fig 3 (300 DPI PNG)",
                                        data=f.read(),
                                        file_name=os.path.basename(f3_files["png"]),
                                        mime="image/png",
                                        key="dl_fig3_png",
                                        use_container_width=True
                                    )
                        with d3_col2:
                            if "pdf" in f3_files and os.path.exists(f3_files["pdf"]):
                                with open(f3_files["pdf"], "rb") as f:
                                    st.download_button(
                                        "⬇ Download Fig 3 (Vector PDF - LaTeX)",
                                        data=f.read(),
                                        file_name=os.path.basename(f3_files["pdf"]),
                                        mime="application/pdf",
                                        key="dl_fig3_pdf",
                                        use_container_width=True
                                    )
                        with st.expander("📋 LaTeX Snippet for Figure 3"):
                            pdf_name = os.path.basename(f3_files.get("pdf", "fig3_rag_verification_profile.pdf"))
                            st.code(
                                "\\begin{figure}[htbp]\n"
                                "  \\centering\n"
                                f"  \\includegraphics[width=0.85\\linewidth]{{{pdf_name}}}\n"
                                "  \\caption{Polar radar profile of RAG verification metrics against the 40\\% fallback threshold.}\n"
                                "  \\label{fig:performance_radar}\n"
                                "\\end{figure}",
                                language="latex"
                            )

                # --- Tab 4: Pipeline Latency Breakdown ---
                with tab4:
                    if "latency_breakdown" in st.session_state.figures and st.session_state.figures["latency_breakdown"]:
                        st.pyplot(st.session_state.figures["latency_breakdown"], use_container_width=True)
                        st.markdown(
                            "**Figure 4:** *Horizontal latency composition of the end-to-end RAG pipeline.* "
                            "Breaks down dense retrieval, hybrid fusion, and AGMR reranking versus LLM generation time."
                        )
                        f4_files = st.session_state.figure_files.get("latency_breakdown", {}) if st.session_state.figure_files else {}
                        d4_col1, d4_col2 = st.columns(2)
                        with d4_col1:
                            if "png" in f4_files and os.path.exists(f4_files["png"]):
                                with open(f4_files["png"], "rb") as f:
                                    st.download_button(
                                        "⬇ Download Fig 4 (300 DPI PNG)",
                                        data=f.read(),
                                        file_name=os.path.basename(f4_files["png"]),
                                        mime="image/png",
                                        key="dl_fig4_png",
                                        use_container_width=True
                                    )
                        with d4_col2:
                            if "pdf" in f4_files and os.path.exists(f4_files["pdf"]):
                                with open(f4_files["pdf"], "rb") as f:
                                    st.download_button(
                                        "⬇ Download Fig 4 (Vector PDF - LaTeX)",
                                        data=f.read(),
                                        file_name=os.path.basename(f4_files["pdf"]),
                                        mime="application/pdf",
                                        key="dl_fig4_pdf",
                                        use_container_width=True
                                    )
                        with st.expander("📋 LaTeX Snippet for Figure 4"):
                            pdf_name = os.path.basename(f4_files.get("pdf", "fig4_pipeline_latency.pdf"))
                            st.code(
                                "\\begin{figure}[htbp]\n"
                                "  \\centering\n"
                                f"  \\includegraphics[width=0.9\\linewidth]{{{pdf_name}}}\n"
                                "  \\caption{End-to-end RAG latency breakdown comparing retrieval \\& AGMR reranking with LLM generation.}\n"
                                "  \\label{fig:latency_breakdown}\n"
                                "\\end{figure}",
                                language="latex"
                            )

                st.info(
                    "📁 **All figures are automatically exported to `results/figures/`** "
                    "in both publication-grade 300 DPI PNG and vector PDF format for direct inclusion in LaTeX or Word documents."
                )


            # ==================================================
            # Download Answer
            # ==================================================

            st.markdown("---")

            st.subheader(
                "📄 Download Answer"
            )


            exported_files = export_all(
                question,
                answer,
                formatted_sources,
                confidence
            )


            download_col1, download_col2, download_col3 = st.columns(3)


            with download_col1:

                with open(
                    exported_files["pdf"],
                    "rb"
                ) as file:

                    st.download_button(
                        "⬇ Download PDF",
                        data=file,
                        file_name=os.path.basename(
                            exported_files["pdf"]
                        ),
                        mime="application/pdf",
                        use_container_width=True
                    )


            with download_col2:

                with open(
                    exported_files["docx"],
                    "rb"
                ) as file:

                    st.download_button(
                        "⬇ Download DOCX",
                        data=file,
                        file_name=os.path.basename(
                            exported_files["docx"]
                        ),
                        mime=(
                            "application/"
                            "vnd.openxmlformats-officedocument."
                            "wordprocessingml.document"
                        ),
                        use_container_width=True
                    )


            with download_col3:

                with open(
                    exported_files["txt"],
                    "rb"
                ) as file:

                    st.download_button(
                        "⬇ Download TXT",
                        data=file,
                        file_name=os.path.basename(
                            exported_files["txt"]
                        ),
                        mime="text/plain",
                        use_container_width=True
                    )


            # ==================================================
            # Research Report
            # ==================================================

            st.markdown("---")

            st.subheader(
                "📝 Generate Research Report"
            )


            report_path = generate_report(
                question=question,
                answer=answer,
                confidence=confidence["confidence"],
                confidence_label=confidence["label"],
                sources=formatted_sources,
                recommended_papers=recommendations
            )


            with open(
                report_path,
                "rb"
            ) as report_file:

                st.download_button(
                    "⬇ Download Research Report",
                    data=report_file,
                    file_name=os.path.basename(
                        report_path
                    ),
                    mime="text/plain",
                    use_container_width=True
                )


            # ==================================================
            # Top Retrieved Chunks
            # ==================================================

            st.markdown("---")

            st.subheader(
                "🔍 Top Retrieved Chunks"
            )


            if len(retrieved_chunks) == 0:

                st.info(
                    "No retrieved chunks."
                )


            else:

                for i, chunk in enumerate(
                    retrieved_chunks,
                    start=1
                ):

                    with st.expander(
                        f"Rank {i} | "
                        f"{chunk['paper_name']} | "
                        f"Chunk {chunk['chunk_id']}"
                    ):

                        st.write(
                            f"**Similarity Score:** "
                            f"{chunk['distance']:.4f}"
                        )

                        st.write("")

                        st.write(
                            chunk["chunk_text"]
                        )


        # ======================================================
        # Error Handling
        # ======================================================

        except Exception as e:

            st.error(
                "An error occurred while processing your request."
            )

            st.exception(
                e
            )


# ==========================================================
# Search History
# ==========================================================

st.markdown("---")

st.subheader(
    "🕘 Search History"
)


history = get_history()


if len(history) == 0:

    st.info(
        "No search history available."
    )


else:

    for item in reversed(
        history[-10:]
    ):

        with st.expander(
            f"{item['time']} | {item['question']}"
        ):

            st.markdown(
                "### Question"
            )

            st.write(
                item["question"]
            )


            st.markdown(
                "### Answer"
            )

            st.write(
                item["answer"]
            )


            st.markdown(
                "### Sources"
            )


            if len(item["sources"]) == 0:

                st.write(
                    "No Sources"
                )


            else:

                for source in item["sources"]:

                    st.write(
                        "• " + source
                    )


# ==========================================================
# Retrieval Quality Dashboard
# ==========================================================

st.markdown("---")

st.subheader(
    "📊 Retrieval Quality Dashboard"
)


RESULTS_FOLDER = "results"


eval_reports = sorted(
    glob.glob(
        os.path.join(
            RESULTS_FOLDER,
            "eval_report_*.json"
        )
    ),
    reverse=True
)


if len(eval_reports) == 0:

    st.info(
        "No eval reports found yet. Populate "
        "`data/eval/golden_test_cases.json` "
        "with real questions from your papers, then run "
        "`python src/eval_harness.py` to generate one."
    )


else:

    report_labels = [
        os.path.basename(path)
        for path in eval_reports
    ]


    selected_label = st.selectbox(
        "Eval Report",
        report_labels
    )


    selected_path = eval_reports[
        report_labels.index(
            selected_label
        )
    ]


    try:

        with open(
            selected_path,
            "r",
            encoding="utf-8"
        ) as file:

            report_data = json.load(
                file
            )


        table_rows = []


        for strategy_result in report_data:

            hit_rate_key = next(
                (
                    k
                    for k in strategy_result
                    if k.startswith("hit_rate_at_")
                ),
                None
            )


            ndcg_key = next(
                (
                    k
                    for k in strategy_result
                    if k.startswith("ndcg_at_")
                ),
                None
            )


            table_rows.append(
                {
                    "Strategy":
                        strategy_result["strategy"],

                    "Hit Rate (%)":
                        strategy_result.get(
                            hit_rate_key
                        ),

                    "MRR":
                        strategy_result.get(
                            "mrr"
                        ),

                    "NDCG":
                        strategy_result.get(
                            ndcg_key
                        ),

                    "Adversarial Accuracy (%)":
                        strategy_result.get(
                            "adversarial_accuracy"
                        )
                }
            )


        st.dataframe(
            table_rows,
            use_container_width=True
        )


        chart_data = {

            row["Strategy"]:
                row["Hit Rate (%)"]

            for row in table_rows

            if row["Hit Rate (%)"] is not None

        }


        if len(chart_data) > 0:

            st.bar_chart(
                chart_data
            )


    except Exception as e:

        st.error(
            "Could not load this eval report."
        )

        st.exception(
            e
        )


# ==========================================================
# Sidebar Statistics
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "Statistics"
)


st.sidebar.metric(
    "Research Papers",
    paper_count
)


st.sidebar.metric(
    "Processed Papers",
    chunk_count
)


st.sidebar.metric(
    "History",
    len(history)
)


# ==========================================================
# About
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "About"
)


st.sidebar.info(
    """
ResearchGPT is an LLM-Based Research Paper
Question Answering System using

• Retrieval-Augmented Generation (RAG)

• Sentence-BERT

• FAISS

• Adaptive Gaussian Mixture Retrieval (AGMR)

• openai/gpt-oss-20b (Groq)

Developed for academic research assistance.
"""
)


# ==========================================================
# Footer
# ==========================================================

st.markdown("---")


st.caption(
    "ResearchGPT | LLM-Based Research Paper Question Answering "
    "using Retrieval-Augmented Generation (RAG), "
    "Sentence-BERT, FAISS, AGMR and openai/gpt-oss-20b"
)