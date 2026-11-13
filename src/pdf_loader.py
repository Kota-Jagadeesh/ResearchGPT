import os
import fitz  # PyMuPDF

# ==========================================================
# Folder Paths
# ==========================================================

RAW_FOLDER = "data/raw"
OUTPUT_FOLDER = "data/processed"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# Extract Text with Page Numbers
# ==========================================================

def extract_text(pdf_path):

    extracted_text = []

    with fitz.open(pdf_path) as document:

        page_count = document.page_count

        paper_name = os.path.basename(pdf_path)

        for page_number, page in enumerate(document, start=1):

            page_text = page.get_text("text").strip()

            if not page_text:
                continue

            extracted_text.append(

                f"Paper : {paper_name}\n"
                f"Page : {page_number}\n\n"
                f"{page_text}\n\n"
                + "=" * 80
                + "\n"

            )

    full_text = "".join(extracted_text)

    return full_text, page_count


# ==========================================================
# Process PDFs
# ==========================================================

def process_all_pdfs(pdf_files=None):

    if pdf_files is None:

        pdf_files = sorted(
            [
                file
                for file in os.listdir(RAW_FOLDER)
                if file.lower().endswith(".pdf")
            ]
        )

    else:

        pdf_files = sorted(pdf_files)

    if len(pdf_files) == 0:

        print("No PDF files found.")

        return

    print("=" * 60)
    print("ResearchGPT PDF Loader")
    print("=" * 60)

    total_pages = 0
    total_words = 0
    total_characters = 0

    for pdf in pdf_files:

        pdf_path = os.path.join(
            RAW_FOLDER,
            pdf
        )

        if not os.path.exists(pdf_path):

            print(f"Skipped : {pdf} (Not Found)")
            continue

        print(f"\nProcessing : {pdf}")

        try:

            text, pages = extract_text(pdf_path)

            output_file = os.path.join(
                OUTPUT_FOLDER,
                os.path.splitext(pdf)[0] + ".txt"
            )

            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(text)

            words = len(text.split())
            characters = len(text)

            total_pages += pages
            total_words += words
            total_characters += characters

            print(f"Pages              : {pages}")
            print(f"Characters         : {characters}")
            print(f"Words              : {words}")
            print(f"Saved              : {output_file}")

        except Exception as e:

            print(f"Failed : {pdf}")
            print(f"Reason : {e}")

    print("\n" + "=" * 60)
    print("PDF Extraction Summary")
    print("=" * 60)
    print(f"Processed PDFs      : {len(pdf_files)}")
    print(f"Total Pages         : {total_pages}")
    print(f"Total Words         : {total_words}")
    print(f"Total Characters    : {total_characters}")
    print("=" * 60)
    print("PDF Extraction Completed Successfully.")


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    process_all_pdfs()