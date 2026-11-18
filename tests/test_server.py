import time

from fastapi.testclient import TestClient

import server
from scientist import llm, tools
from tests.test_scientist import ARXIV
from scientist.tools import parse_arxiv


def _stub(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "RUNS", tmp_path)
    monkeypatch.setattr(llm, "api_key_set", lambda: True)
    monkeypatch.setitem(tools.SEARCHERS, "search_arxiv", lambda q, n: parse_arxiv(ARXIV))

    def agent(system, user, tools, dispatch, **kw):
        dispatch("search_arxiv", {"query": "q"})
        return "notes"

    def complete(system, user, **kw):
        if "academic writer" in system:
            return "# T\n\nAccuracy was 91% [@lovelace2021]."
        if "scientific editor" in system:
            return "# T\n\nWe saw 91% accuracy [@lovelace2021]."
        return "ok"

    monkeypatch.setattr(llm, "run_tool_agent", agent)
    monkeypatch.setattr(llm, "complete", complete)


def test_full_run_through_api(monkeypatch, tmp_path):
    _stub(monkeypatch, tmp_path)
    c = TestClient(server.app)
    assert c.get("/").status_code == 200
    run_id = c.post("/api/runs", json={"topic": "graph networks for drugs", "kind": "paper", "turns": 3}).json()["id"]
    for _ in range(50):
        d = c.get(f"/api/runs/{run_id}").json()
        if d["status"] not in ("running", "queued"):
            break
        time.sleep(0.1)
    assert d["status"] == "done", d
    assert "We saw 91% accuracy" in d["docs"]["final"] and d["library"][0]["key"] == "lovelace2021"
    assert c.get(f"/api/runs/{run_id}/download").status_code == 200
    assert [r["id"] for r in c.get("/api/runs").json()] == [run_id]


def test_validation_and_safety(monkeypatch, tmp_path):
    _stub(monkeypatch, tmp_path)
    c = TestClient(server.app)
    assert c.post("/api/runs", json={"topic": "short"}).status_code == 422
    assert c.post("/api/runs", json={"topic": "a long enough topic", "kind": "bogus"}).status_code == 400
    assert c.get("/api/runs/..%2f..%2fetc").status_code == 404
    monkeypatch.setattr(llm, "api_key_set", lambda: False)
    assert c.post("/api/runs", json={"topic": "a long enough topic"}).status_code == 503
