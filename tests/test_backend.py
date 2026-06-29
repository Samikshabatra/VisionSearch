from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_reports_gallery():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["gallery_size"] == 1000


def test_search_returns_ranked_results():
    r = client.post("/search", json={"query": "a dog on the beach", "k": 5})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 5
    assert all("filename" in x and "url" in x for x in results)
    scores = [x["score"] for x in results]
    assert scores == sorted(scores, reverse=True)   # ranked


def test_empty_query_rejected():
    assert client.post("/search", json={"query": "  "}).status_code == 400
