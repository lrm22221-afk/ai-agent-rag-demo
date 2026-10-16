"""Usage: python -m rag "your question" """
import sys

from .service import RagService

r = RagService.from_docs_dir("docs").ask(" ".join(sys.argv[1:]) or "What are your support hours?")
print(f"Q: {' '.join(sys.argv[1:])}\nA: {r.answer}\nlang={r.language} sources={r.sources} tools={r.tool_calls} refused={r.refused} blocked={r.blocked}")
