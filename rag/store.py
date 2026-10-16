from __future__ import annotations

import pathlib
from dataclasses import dataclass

from .embeddings import Embedder


@dataclass
class Chunk:
    id: str
    text: str
    vector: list[float]


class VectorStore:
    """Minimal in-memory vector store (swap for pgvector/Qdrant/Pinecone in production)."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []

    def add(self, cid: str, text: str) -> None:
        self.chunks.append(Chunk(cid, text, self.embedder.embed(text)))

    def search(self, query: str, k: int = 3) -> list[tuple[float, Chunk]]:
        q = self.embedder.embed(query)
        scored = [(sum(a * b for a, b in zip(q, c.vector)), c) for c in self.chunks]
        return sorted(scored, key=lambda x: -x[0])[:k]

    def load_dir(self, path: str) -> None:
        for f in sorted(pathlib.Path(path).glob("*.md")):
            section = ""
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
                line = line.strip()
                if line.startswith("#"):
                    section = line.lstrip("# ")
                elif line:
                    self.add(f"{f.stem}#L{i + 1}", f"{section}: {line}" if section else line)
