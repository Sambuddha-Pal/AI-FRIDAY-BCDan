"""
Lightweight PII scrubbing + text preprocessing for customer complaint text.

This is a regex/heuristic based anonymizer intended for a demo/prototype.
For production use, swap in a proper PII-detection library (e.g. Microsoft
Presidio) or a cloud DLP service.
"""
import re

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}")
ORDER_ID_RE = re.compile(r"\b(order|invoice|ref|serial|sr|so)[\s#:-]*[a-zA-Z0-9-]{4,}\b", re.IGNORECASE)
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
ADDRESS_HINT_RE = re.compile(r"\b\d{1,5}\s+\w+\s+(street|st|avenue|ave|road|rd|lane|ln|block|sector)\b", re.IGNORECASE)

REPLACEMENTS = [
    (EMAIL_RE, "[EMAIL_REDACTED]"),
    (CREDIT_CARD_RE, "[CARD_REDACTED]"),
    (PHONE_RE, "[PHONE_REDACTED]"),
    (ORDER_ID_RE, "[ORDER_ID_REDACTED]"),
    (ADDRESS_HINT_RE, "[ADDRESS_REDACTED]"),
]


def anonymize_text(text: str) -> str:
    """Redact common PII patterns from a complaint text."""
    if not text:
        return text
    cleaned = text
    for pattern, replacement in REPLACEMENTS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def clean_whitespace(text: str) -> str:
    """Normalize whitespace/newlines from messy pasted or CRM-exported text."""
    if not text:
        return text
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def preprocess(text: str) -> str:
    """Full preprocessing pipeline: clean whitespace then anonymize."""
    return anonymize_text(clean_whitespace(text))
