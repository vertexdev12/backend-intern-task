import pytest
from fastapi.testclient import TestClient
from app.chunking import chunk_text
from app.main import create_app

def c(): return TestClient(create_app())

def test_chunk_edges():
    assert chunk_text("") == [] and chunk_text("   \n ") == []
    assert chunk_text("a b c", size=5, overlap=2) == ["a b c"]
    assert chunk_text("a b c d e", size=5, overlap=2) == ["a b c d e"]
    assert chunk_text("a b c d e f", size=5, overlap=2) == ["a b c d e", "d e f"]
    for s, o in [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6)]:
        with pytest.raises(ValueError): chunk_text("a b", s, o)

def test_validation_and_flow():
    cl = c()
    assert cl.post("/ask", json={"question": "x"}).status_code == 404
    for body in [{}, {"documents": []}, {"documents": [{"id": "", "text": "a"}]},
                 {"documents": [{"id": "a", "text": "  "}]}]:
        assert cl.post("/ingest", json=body).status_code == 422
    long = " ".join(f"w{i}" for i in range(120))
    r = cl.post("/ingest", json={"documents": [{"id": "b", "text": long}, {"id": "a", "text": "cats purr"}]})
    assert r.json() == {"ingested": 2, "chunks": 4}
    assert cl.get("/documents").json() == {"documents": [{"id": "a", "chunks": 1}, {"id": "b", "chunks": 3}]}
    r = cl.post("/ingest", json={"documents": [{"id": "b", "text": "dogs bark"}]})
    assert cl.get("/documents").json()["documents"][1] == {"id": "b", "chunks": 1}
    for k in (0, 11):
        assert cl.post("/ask", json={"question": "cats", "top_k": k}).status_code == 422
    assert cl.post("/ask", json={"question": " "}).status_code == 422
    r = cl.post("/ask", json={"question": "cats purr", "top_k": 10}).json()
    assert len(r["sources"]) == 2 and r["sources"][0]["chunk_id"] == "a:0"
    assert cl.delete("/documents/a").status_code == 204
    assert cl.delete("/documents/a").status_code == 404
    r = cl.post("/ask", json={"question": "cats purr"}).json()
    assert all(s["doc_id"] != "a" for s in r["sources"])

def test_isolation():
    a, b = c(), c()
    a.post("/ingest", json={"documents": [{"id": "x", "text": "hi"}]})
    assert b.get("/documents").json() == {"documents": []}
