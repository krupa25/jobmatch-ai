"""
resume_parser.py
This files - Takes a PDF resume and 
transfroms it into cleaner text and simpler sections organized.
"""
# Cleaning text with patterns, re - Regex module
import re
# Type hint for better readability
from typing import Dict
import pdfplumber #Better PDF Extraction
from pypdf import PdfReader # Fall back PDF extractor

def extract_text_with_pdfplumber(file_path: str) -> str:
    """
    Extract text from a PDF using pdfplumber.
    This often gives better text for resumes than basic PDF readers.
    """

    all_pages_text = [] # Store text page by page

    # Open PDF file
    with pdfplumber.open(file_path) as pdf:
        # Loop through each page
        for page in pdf.pages:
        # Extract text from page
        # Sometimes extract_text() retuens None, so we default to ""

            page_text = page.extract_text() or ""
            all_pages_text.append(page_text)

    # Join all pages into one big string
    return "\n".join(all_pages_text)

def extract_text_with_pypdf(file_path: str) -> str:
    """
    Extract text from a PDF using pypdf.
    This is a fallback method in case pdfplumber returns little or no text
    """
    reader = PdfReader(file_path)
    all_pages_text = []

    for page in reader.pages:
        all_pages_text.append(page.extract_text() or "")

    return "\n".join(all_pages_text)

def clean_resume_text(text: str) -> str:
    """
    Make extracted resume text cleaner and more consistent.
    this improves:
    - section detection
    - retrieval later
    - LLM prompt quality
    """

    # Replace tabs with spaces (tabs mess up formatting)
    text = text.replace("\t", " ")

    # Replace multiple spaces with a single space
    text = re.sub(r"[ ]{2,}", " ", text)


    # Reduce too many blank lines to just two
    # Example: "\n\n\n\n" -> "\n\n"
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove extra whitespace at start/end
    return text.strip()

# Section header words we try to detect in the resume
SECTION_HEADERS = {
    "skills": ["skills", "technical skills", "core skills"],
    "experience": ["experience", "work experience", "professional experience"],
    "projects": ["projects", "academic projects", "personal projects"],
    "education": ["education", "academic background"],
}


def split_into_sections(text: str) -> Dict[str, str]:
    """
    Split resume text into sections using simple heuristics.

    How it works:
    - We scan each line
    - If a line looks like a header (e.g., 'SKILLS'), we mark it
    - Everything until the next header becomes that section

    If no headers are found:
    - return {"raw": full_text}
    """

    # Split into lines and trim spaces
    lines = [line.strip() for line in text.splitlines()]

    # Find header positions: list of (line_index, section_name)
    found = []

    for i, line in enumerate(lines):
        if not line:
            # Skip empty lines
            continue

        low = line.lower()

        # Check every known section and its possible header names
        for section_name, header_variants in SECTION_HEADERS.items():
            # We use "contains" matching to be more flexible:
            # Example: "TECHNICAL SKILLS & TOOLS" should match "skills"
            for header in header_variants:
                if header in low:
                    found.append((i, section_name))
                    break

    # If no headers detected, return everything as raw
    if not found:
        return {"raw": text}

    # Sort headers by where they occur in the document
    found.sort(key=lambda x: x[0])

    sections: Dict[str, str] = {}

    # Slice text between headers
    for idx, (start_idx, section_name) in enumerate(found):
        # The end of this section is the start of the next section header
        end_idx = found[idx + 1][0] if idx + 1 < len(found) else len(lines)

        # Take lines after the header line itself
        content_lines = lines[start_idx + 1:end_idx]

        # Join them back into a single string
        section_text = "\n".join(content_lines).strip()

        # Store result
        sections[section_name] = section_text

    return sections


def parse_resume_pdf(file_path: str) -> Dict:
    """
    Main function for Day 2.
    Returns:
    - raw_text (cleaned)
    - sections (dict)
    """

    # Step 1: try best extractor first
    raw_text = extract_text_with_pdfplumber(file_path)

    # Step 2: if too little text, fallback extractor
    if len(raw_text.strip()) < 30:
        raw_text = extract_text_with_pypdf(file_path)

    # Step 3: clean text
    cleaned = clean_resume_text(raw_text)

    # Step 4: split into sections
    sections = split_into_sections(cleaned)

    # Step 5: return structured output
    return {
        "raw_text": cleaned,
        "sections": sections,
    }