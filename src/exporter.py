import os
import re
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from docx import Document


# ==========================================================
# Folder
# ==========================================================

RESULT_FOLDER = "results"

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)


# ==========================================================
# Generate Timestamp
# ==========================================================

def get_timestamp():

    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


# ==========================================================
# Clean Text For ReportLab PDF
# ==========================================================
# LLM answers can contain Markdown such as:
#
# **bold**
# *italic*
# <br>
# <br>\
# | table | values |
#
# ReportLab Paragraph does not understand normal Markdown
# and can interpret HTML-like tags as ReportLab markup.
#
# This function converts the answer into safe ReportLab text.

def clean_text_for_pdf(text):

    if text is None:

        return ""

    text = str(text)

    # Normalize Windows line endings
    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # ------------------------------------------------------
    # Remove malformed HTML line breaks
    # ------------------------------------------------------

    text = re.sub(
        r"<br\s*/?\s*>",
        "\n",
        text,
        flags=re.IGNORECASE
    )

    # ------------------------------------------------------
    # Remove any remaining HTML/XML tags
    # ------------------------------------------------------

    text = re.sub(
        r"<[^>]*>",
        "",
        text
    )

    # ------------------------------------------------------
    # Remove backslash characters left by malformed
    # <br>\ sequences
    # ------------------------------------------------------

    text = text.replace(
        "\\",
        ""
    )

    # ------------------------------------------------------
    # Convert Markdown bold/italic markers
    # ------------------------------------------------------

    text = text.replace(
        "**",
        ""
    )

    text = text.replace(
        "__",
        ""
    )

    text = text.replace(
        "*",
        ""
    )

    # ------------------------------------------------------
    # Escape ReportLab special characters
    # ------------------------------------------------------

    text = escape(
        text
    )

    # ------------------------------------------------------
    # Convert newlines to safe ReportLab line breaks
    # ------------------------------------------------------

    text = text.replace(
        "\n",
        "<br/>"
    )

    return text


# ==========================================================
# Clean Question For PDF
# ==========================================================

def clean_question_for_pdf(text):

    return clean_text_for_pdf(
        text
    )


# ==========================================================
# Clean Source For PDF
# ==========================================================

def clean_source_for_pdf(text):

    return clean_text_for_pdf(
        text
    )


# ==========================================================
# Export PDF
# ==========================================================

def export_pdf(
    question,
    answer,
    sources,
    confidence=None
):

    file_path = os.path.join(

        RESULT_FOLDER,

        f"ResearchGPT_Answer_{get_timestamp()}.pdf"

    )

    document = SimpleDocTemplate(
        file_path
    )

    styles = getSampleStyleSheet()

    story = []


    # ------------------------------------------------------
    # Title
    # ------------------------------------------------------

    story.append(

        Paragraph(

            "<b>ResearchGPT Answer</b>",

            styles["Heading1"]

        )

    )

    story.append(
        Spacer(1, 12)
    )


    # ------------------------------------------------------
    # Question
    # ------------------------------------------------------

    story.append(

        Paragraph(

            "<b>Question</b>",

            styles["Heading2"]

        )

    )

    story.append(

        Paragraph(

            clean_question_for_pdf(
                question
            ),

            styles["BodyText"]

        )

    )

    story.append(
        Spacer(1, 10)
    )


    # ------------------------------------------------------
    # Answer
    # ------------------------------------------------------

    story.append(

        Paragraph(

            "<b>Answer</b>",

            styles["Heading2"]

        )

    )

    safe_answer = clean_text_for_pdf(
        answer
    )

    if safe_answer:

        story.append(

            Paragraph(

                safe_answer,

                styles["BodyText"]

            )

        )

    else:

        story.append(

            Paragraph(

                "No answer available.",

                styles["BodyText"]

            )

        )


    story.append(
        Spacer(1, 10)
    )


    # ------------------------------------------------------
    # Confidence
    # ------------------------------------------------------

    if confidence is not None:

        story.append(

            Paragraph(

                "<b>Confidence Score</b>",

                styles["Heading2"]

            )

        )

        confidence_text = (

            f"{confidence.get('confidence', 0)} % "

            f"({confidence.get('label', 'Unknown')})"

        )

        story.append(

            Paragraph(

                escape(
                    confidence_text
                ),

                styles["BodyText"]

            )

        )

        story.append(
            Spacer(1, 10)
        )


    # ------------------------------------------------------
    # Sources
    # ------------------------------------------------------

    story.append(

        Paragraph(

            "<b>Sources</b>",

            styles["Heading2"]

        )

    )


    if sources:

        for source in sources:

            safe_source = clean_source_for_pdf(
                source
            )

            if safe_source:

                story.append(

                    Paragraph(

                        safe_source,

                        styles["BodyText"]

                    )

                )

    else:

        story.append(

            Paragraph(

                "No sources available.",

                styles["BodyText"]

            )

        )


    # ------------------------------------------------------
    # Build PDF
    # ------------------------------------------------------

    document.build(
        story
    )

    return file_path


# ==========================================================
# Export DOCX
# ==========================================================

def export_docx(
    question,
    answer,
    sources,
    confidence=None
):

    file_path = os.path.join(

        RESULT_FOLDER,

        f"ResearchGPT_Answer_{get_timestamp()}.docx"

    )

    doc = Document()


    doc.add_heading(
        "ResearchGPT Answer",
        level=1
    )


    doc.add_heading(
        "Question",
        level=2
    )

    doc.add_paragraph(
        question
    )


    doc.add_heading(
        "Answer",
        level=2
    )

    doc.add_paragraph(
        answer
    )


    if confidence is not None:

        doc.add_heading(
            "Confidence Score",
            level=2
        )

        doc.add_paragraph(

            f"{confidence.get('confidence', 0)} % "

            f"({confidence.get('label', 'Unknown')})"

        )


    doc.add_heading(
        "Sources",
        level=2
    )


    if sources:

        for source in sources:

            doc.add_paragraph(
                source
            )

    else:

        doc.add_paragraph(
            "No sources available."
        )


    doc.save(
        file_path
    )

    return file_path


# ==========================================================
# Export TXT
# ==========================================================

def export_txt(
    question,
    answer,
    sources,
    confidence=None
):

    file_path = os.path.join(

        RESULT_FOLDER,

        f"ResearchGPT_Answer_{get_timestamp()}.txt"

    )

    with open(

        file_path,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(
            "ResearchGPT Answer\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )


        file.write(
            "Question\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            question + "\n\n"
        )


        file.write(
            "Answer\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            answer + "\n\n"
        )


        if confidence is not None:

            file.write(
                "Confidence Score\n"
            )

            file.write(
                "-" * 60 + "\n"
            )

            file.write(

                f"{confidence.get('confidence', 0)} % "

                f"({confidence.get('label', 'Unknown')})\n\n"

            )


        file.write(
            "Sources\n"
        )

        file.write(
            "-" * 60 + "\n"
        )


        if sources:

            for source in sources:

                file.write(
                    source + "\n"
                )

        else:

            file.write(
                "No sources available.\n"
            )


    return file_path


# ==========================================================
# Export All Formats
# ==========================================================

def export_all(
    question,
    answer,
    sources,
    confidence=None
):

    return {

        "pdf": export_pdf(

            question,

            answer,

            sources,

            confidence

        ),

        "docx": export_docx(

            question,

            answer,

            sources,

            confidence

        ),

        "txt": export_txt(

            question,

            answer,

            sources,

            confidence

        )

    }


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    question = (
        "What are the limitations of "
        "Retrieval-Augmented Generation?"
    )


    answer = """

**Drawbacks / limitations of the RAG model**

| Limitation | Evidence from the papers |
|------------|--------------------------|
| Dependence on retrieval quality | RAG performance depends on the quality and relevance of the retrieved passages. |
| Sensitivity to retrieved documents | Increasing the number of retrieved documents does not always improve performance. |

<br>Testing PDF export.<br>\\

"""


    sources = [

        "RAG.pdf (Chunk 12)",

        "RePlug.pdf (Chunk 5)"

    ]


    confidence = {

        "confidence": 39.42,

        "label": "Very Low"

    }


    files = export_all(

        question,

        answer,

        sources,

        confidence

    )


    print(
        "\nGenerated Files\n"
    )


    for file_type, path in files.items():

        print(
            f"{file_type.upper()} : {path}"
        )