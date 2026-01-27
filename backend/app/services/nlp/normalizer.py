import re

def normalize_text(text: str) -> str:
    """
    Cleans OCR text:
    - Removes extra whitespace
    - Removes duplicate lines
    - Strips emails & phone numbers (PII)
    """
    # Remove duplicate lines
    lines = list(dict.fromkeys(text.splitlines()))
    text = "\n".join(lines)

    # Strip emails
    text = re.sub(r'\S+@\S+', '[email]', text)
    # Strip phone numbers
    text = re.sub(r'\+?\d[\d -]{7,}\d', '[phone]', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()
