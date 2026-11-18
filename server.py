"""Web app: FastAPI backend + static frontend. Run with:  python server.py"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scientist import llm, pdf, pipeline
from scientist.library import Library

ROOT = Path(__file__).parent
RUNS = Path(os.environ.get("SCIENTIST_RUNS", ROOT / "runs"))
ID_RE = re.compile(r"^[a-z0-9\-]{1,80}$")
FILES = {
    "final": "final.md",
    "evidence": "1_evidence.md",
    "hypotheses": "2_hypotheses.md",
    "draft": "3_draft.md",
    "verification": "5_verification.md",
}

app = FastAPI(title="Scientific Collaborator")
pool = ThreadPoolExecutor(max_workers=int(os.environ.get("MAX_CONCURRENT_RUNS", 1)))  # runs share one token budget
_lock = threading.Lock()
_state: dict[str, dict] = {}  # run id -> live status (also mirrored to status.json)


class NewRun(BaseModel):
    topic: str = Field(min_length=8, max_length=400)
    kind: str = "review"
    turns: int = Field(default=12, ge=3, le=30)


def _persist(run_id: str) -> None:
    (RUNS / run_id / "status.json").write_text(json.dumps(_state[run_id]), encoding="utf-8")


def _work(run_id: str, req: NewRun) -> None:
    st = _state[run_id]
    st["status"] = "running"

    def log(msg: str) -> None:
        with _lock:
            st["log"].append(msg)
            m = re.match(r"\[(\d)/5\]", msg)
            if m:
                st["stage"] = int(m.group(1))

    try:
        res = pipeline.run(req.topic, RUNS / run_id, kind=req.kind, max_turns=req.turns, log=log)
        st["status"] = "done" if res.ok else "done_with_warnings"
    except llm.Refused as e:
        st["status"], st["error"] = "failed", f"The model declined this request: {e}"
    except Exception as e:  # surface anything to the UI instead of dying silently
        st["status"], st["error"] = "failed", f"{type(e).__name__}: {e}"
    st["finished"] = time.time()
    _persist(run_id)


def _valid(run_id: str) -> Path:
    if not ID_RE.match(run_id) or not (RUNS / run_id).is_dir():
        raise HTTPException(404, "run not found")
    return RUNS / run_id


@app.get("/api/health")
def health():
    return {"api_key_set": llm.api_key_set(), "model": llm.MODEL, "endpoint": llm.BASE_URL}


@app.post("/api/runs")
def create_run(req: NewRun):
    if req.kind not in pipeline.KINDS:
        raise HTTPException(400, f"kind must be one of {list(pipeline.KINDS)}")
    if not llm.api_key_set():
        raise HTTPException(503, "LLM_API_KEY is not set on the server (see .env).")
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{pipeline.slugify(req.topic)}"
    (RUNS / run_id).mkdir(parents=True, exist_ok=True)
    _state[run_id] = {"id": run_id, "topic": req.topic, "kind": req.kind, "status": "queued",
                      "stage": 0, "log": [], "error": None, "started": time.time()}
    _persist(run_id)
    pool.submit(_work, run_id, req)
    return {"id": run_id}


@app.get("/api/runs")
def list_runs():
    rows = []
    for d in sorted(RUNS.glob("*/"), reverse=True) if RUNS.exists() else []:
        s = _state.get(d.name)
        if not s and (d / "status.json").exists():
            s = json.loads((d / "status.json").read_text(encoding="utf-8"))
        if s:
            row = {k: s.get(k) for k in ("id", "topic", "kind", "status", "started")}
            if d.name not in _state and row["status"] in ("running", "queued"):
                row["status"] = "failed"  # left over from a previous server session; same rule as get_run
            rows.append(row)
    return rows


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, since: int = 0):
    d = _valid(run_id)
    st = _state.get(run_id)
    if st is None:  # run from a previous server session
        path = d / "status.json"
        if not path.exists():
            raise HTTPException(404, "run not found")
        st = json.loads(path.read_text(encoding="utf-8"))
        if st["status"] in ("running", "queued"):
            st["status"], st["error"] = "failed", "Server restarted while this run was in progress."
    with _lock:
        out = {**st, "log": st["log"][since:], "log_total": len(st["log"])}
    if out["status"].startswith("done"):
        out["docs"] = {k: (d / f).read_text(encoding="utf-8") for k, f in FILES.items() if (d / f).exists()}
        fl = d / "flags.json"
        out["flags"] = json.loads(fl.read_text(encoding="utf-8")) if fl.exists() else None  # None: run predates claim checks
        lib = d / "library.json"
        out["library"] = json.loads(lib.read_text(encoding="utf-8")) if lib.exists() else []
    return out


@app.get("/api/runs/{run_id}/download")
def download(run_id: str):
    d = _valid(run_id)
    f = d / "final.md"
    if not f.exists():
        raise HTTPException(404, "not finished")
    return PlainTextResponse(f.read_text(encoding="utf-8"), media_type="text/markdown",
                             headers={"Content-Disposition": f'attachment; filename="{run_id}.md"'})


class PdfRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    affiliation: str = Field(default="", max_length=200)
    email: str = Field(default="", max_length=120)
    orcid: str = Field(default="", max_length=40)
    disclosure: bool = True


@app.post("/api/runs/{run_id}/pdf")
def make_pdf(run_id: str, req: PdfRequest):
    """POST so the author's details travel in the body, not the URL. Nothing is stored."""
    d = _valid(run_id)
    final, lib = d / "final.md", d / "library.json"
    if not final.exists() or not lib.exists():
        raise HTTPException(404, "this run has not finished")
    data = pdf.build_pdf(final.read_text(encoding="utf-8"), Library.load(lib),
                         pdf.Author(req.name, req.affiliation, req.email, req.orcid), req.disclosure)
    return Response(data, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{run_id[:40]}.pdf"'})


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
