from __future__ import annotations

import re
from dataclasses import dataclass, field

from .embeddings import Embedder, HashingEmbedder
from .guard import is_injection, sanitize_context
from .llm import LLM, MockLLM
from .store import VectorStore
from .tools import TOOLS

REFUSAL = {"en": "I don't know based on our policy documents.", "he": "אין לי מידע על כך במסמכי המדיניות שלנו."}
BLOCKED = {"en": "I can't help with that request.", "he": "אני לא יכול לעזור עם הבקשה הזאת."}


@dataclass
class Result:
    answer: str
    language: str
    sources: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    refused: bool = False
    blocked: bool = False


def detect_language(text: str) -> str:
    return "he" if re.search(r"[\u0590-\u05FF]", text) else "en"


class RagService:
    def __init__(self, store: VectorStore, llm: LLM, min_score: float = 0.2):
        self.store, self.llm, self.min_score = store, llm, min_score

    @classmethod
    def from_docs_dir(cls, path: str, llm: LLM | None = None, embedder: Embedder | None = None):
        store = VectorStore(embedder or HashingEmbedder())
        store.load_dir(path)
        return cls(store, llm or MockLLM())

    def ask(self, question: str) -> Result:
        lang = detect_language(question)
        if is_injection(question):
            return Result(BLOCKED[lang], lang, blocked=True)
        call = self.llm.choose_tool(question, TOOLS)
        if call:
            out = TOOLS[call["name"]]["fn"](**call["args"])
            ans = f"Order {call['args']['order_id']}: {out}" if lang == "en" else f"הזמנה {call['args']['order_id']}: {out}"
            return Result(ans, lang, tool_calls=[call])
        hits = [(s, c) for s, c in self.store.search(question, k=3) if s >= self.min_score]
        if not hits:
            return Result(REFUSAL[lang], lang, refused=True)
        # prefer chunks in the user's language
        same = [h for h in hits if (detect_language(h[1].text) == lang)] or hits
        context = [sanitize_context(c.text) for _, c in same]
        return Result(self.llm.answer(question, context, lang), lang,
                      sources=[c.id for _, c in same], context=context)
