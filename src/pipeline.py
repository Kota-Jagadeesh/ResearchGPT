from pdf_loader import process_all_pdfs
from preprocessing import preprocess_all_files
from chunking import chunk_all_files
from embedding import generate_embeddings
from faiss_db import build_faiss_index
from dense_retriever import reload_resources as reload_dense
from bm25_retriever import reload_resources as reload_bm25


def run_pipeline():

    print("\n")
    print("=" * 70)
    print("ResearchGPT Automatic Pipeline")
    print("=" * 70)

    # ----------------------------------------------------
    print("\nStep 1 : Extracting Text from PDFs\n")
    process_all_pdfs()

    # ----------------------------------------------------
    print("\nStep 2 : Preprocessing Text\n")
    preprocess_all_files()

    # ----------------------------------------------------
    print("\nStep 3 : Creating Text Chunks\n")
    chunk_all_files()

    # ----------------------------------------------------
    print("\nStep 4 : Generating Sentence-BERT Embeddings\n")
    generate_embeddings()

    # ----------------------------------------------------
    print("\nStep 5 : Building FAISS Index\n")
    build_faiss_index()

    # ----------------------------------------------------
    print("\nStep 6 : Refreshing In-Memory Search Caches\n")
    try:
        reload_dense()
    except Exception as e:
        print(f"Notice: Dense cache refresh: {e}")
    try:
        reload_bm25()
    except Exception as e:
        print(f"Notice: BM25 cache refresh: {e}")

    print("\n")
    print("=" * 70)
    print("ResearchGPT Pipeline Completed Successfully")
    print("=" * 70)


if __name__ == "__main__":

    run_pipeline()