import os
import re

INPUT_FOLDER = "data/processed"
OUTPUT_FOLDER = "data/chunks"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def clean_text(text):
    """
    Clean extracted PDF text.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove tabs
    text = text.replace("\t", " ")

    # Remove multiple spaces
    text = re.sub(r"[ ]+", " ", text)

    # Remove spaces before new lines
    text = re.sub(r" +\n", "\n", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading and trailing whitespace
    text = text.strip()

    return text


def preprocess_all_files():

    txt_files = sorted(
        [
            file
            for file in os.listdir(INPUT_FOLDER)
            if file.lower().endswith(".txt")
        ]
    )

    if not txt_files:
        print("No text files found in data/processed/")
        return

    print("=" * 60)
    print("ResearchGPT Text Preprocessing")
    print("=" * 60)

    total_original = 0
    total_cleaned = 0

    for file in txt_files:

        input_path = os.path.join(INPUT_FOLDER, file)

        try:

            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()

            cleaned_text = clean_text(text)

            output_path = os.path.join(OUTPUT_FOLDER, file)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            original_chars = len(text)
            cleaned_chars = len(cleaned_text)

            total_original += original_chars
            total_cleaned += cleaned_chars

            print(f"\nProcessed            : {file}")
            print(f"Original Characters  : {original_chars}")
            print(f"Cleaned Characters   : {cleaned_chars}")

        except Exception as e:

            print(f"\nFailed : {file}")
            print(f"Reason : {e}")

    print("\n" + "=" * 60)
    print("Preprocessing Summary")
    print("=" * 60)
    print(f"Files Processed      : {len(txt_files)}")
    print(f"Original Characters  : {total_original}")
    print(f"Cleaned Characters   : {total_cleaned}")
    print("=" * 60)
    print("All text files cleaned successfully.")


if __name__ == "__main__":
    preprocess_all_files()