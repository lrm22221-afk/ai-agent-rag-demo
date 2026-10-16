from .embeddings import tokenize


def grounding_score(answer: str, context: list[str]) -> float:
    """Share of answer tokens that appear in the retrieved context (1.0 = fully grounded)."""
    a = tokenize(answer)
    if not a:
        return 1.0
    ctx = set(tokenize(" ".join(context)))
    return sum(t in ctx for t in a) / len(a)
