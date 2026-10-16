import json
import pathlib

WF = json.loads(pathlib.Path("n8n/rag_support_agent.json").read_text())

def test_has_agent_and_vector_store():
    types = {n["type"] for n in WF["nodes"]}
    assert "@n8n/n8n-nodes-langchain.agent" in types
    assert any("vectorStore" in t for t in types)
    assert any("embeddings" in t for t in types)

def test_connections_reference_existing_nodes():
    names = {n["name"] for n in WF["nodes"]}
    for src, outs in WF["connections"].items():
        assert src in names
        for kind in outs.values():
            for branch in kind:
                for c in branch:
                    assert c["node"] in names

def test_no_hardcoded_secrets():
    raw = json.dumps(WF)
    assert "sk-" not in raw and "apiKey\": \"" not in raw
