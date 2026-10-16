import re

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"system prompt", r"\bDAN\b", r"reveal .*(key|password|secret)", r"api key",
    r"התעלם מ?(כל )?ה?הוראות", r"הנחיות המערכת", r"סיסמ[הא]",
]
_RX = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def is_injection(text: str) -> bool:
    return bool(_RX.search(text))


def sanitize_context(text: str) -> str:
    """Drop sentences inside retrieved docs that try to give the model instructions."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(p for p in parts if not is_injection(p))
