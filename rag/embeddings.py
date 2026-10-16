"""Pluggable embeddings. Default: deterministic, offline hashed char n-grams (works for Hebrew + English)."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

_HE_PREFIX = re.compile(r"^[והבלמשכ](?=[\u0590-\u05FF]{3,})")


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[\w\u0590-\u05FF:]+", text.lower())
    return [_HE_PREFIX.sub("", t) for t in toks]


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    def __init__(self, dim: int = 2048, n: int = 3):
        self.dim, self.n = dim, n

    def embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in tokenize(text):
            if tok in STOPWORDS:
                continue
            padded = f"#{tok}#"
            grams = [padded[i:i + self.n] for i in range(max(1, len(padded) - self.n + 1))]
            for g in grams + [tok]:
                h = int(hashlib.md5(g.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0 if g != tok else 2.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


STOPWORDS = {"a", "an", "the", "is", "are", "do", "does", "i", "you", "your", "our", "we", "of", "to", "in", "on", "for", "and", "or", "what", "how", "many", "long", "much", "can", "with", "by", "be", "it", "my", "me", "this", "that", "at", "from", "any", "which", "who", "when", "where", "מה", "איך", "כמה", "של", "את", "על", "עם", "זה", "זו", "הם", "הן", "אני", "אתה", "אפשר", "יש", "לא", "תוך", "מי"}
