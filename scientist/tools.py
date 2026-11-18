"""Literature search tools (arXiv, PubMed, Semantic Scholar) and an opt-in Python runner."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from typing import Callable

import requests

from .library import Library, Paper

TIMEOUT = 30
HEADERS = {"User-Agent": "scientist-collaborator/0.1 (research tool)"}
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def _get(url: str, params: dict, headers: dict | None = None, retries: int = 3) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers={**HEADERS, **(headers or {})}, timeout=TIMEOUT)
            if r.status_code == 429 or r.status_code >= 500:
                last = RuntimeError(f"HTTP {r.status_code} (rate limited or server error)")
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"request to {url} failed: {last}")


def parse_arxiv(xml_text: str) -> list[Paper]:
    papers = []
    for entry in ET.fromstring(xml_text).findall("a:entry", ATOM):
        title = " ".join((entry.findtext("a:title", "", ATOM)).split())
        if not title:
            continue
        published = entry.findtext("a:published", "", ATOM)
        doi = entry.findtext("{http://arxiv.org/schemas/atom}doi", "")
        papers.append(Paper(
            title=title,
            authors=[" ".join(a.findtext("a:name", "", ATOM).split()) for a in entry.findall("a:author", ATOM)],
            year=int(published[:4]) if published[:4].isdigit() else None,
            abstract=" ".join(entry.findtext("a:summary", "", ATOM).split()),
            venue="arXiv",
            url=entry.findtext("a:id", "", ATOM),
            doi=doi or "",
            source="arxiv",
        ))
    return papers


def search_arxiv(query: str, max_results: int = 8) -> list[Paper]:
    r = _get("http://export.arxiv.org/api/query", {
        "search_query": f"all:{query}", "max_results": max_results,
        "sortBy": "relevance", "sortOrder": "descending",
    })
    return parse_arxiv(r.text)


def parse_pubmed_articles(xml_text: str) -> list[Paper]:
    papers = []
    for art in ET.fromstring(xml_text).iter("PubmedArticle"):
        pmid = art.findtext(".//MedlineCitation/PMID", "")
        title = "".join(art.find(".//ArticleTitle").itertext()).strip() if art.find(".//ArticleTitle") is not None else ""
        if not title:
            continue
        abstract = " ".join(
            ("".join(t.itertext())).strip() for t in art.findall(".//Abstract/AbstractText")
        )
        authors = []
        for au in art.findall(".//AuthorList/Author"):
            last, fore = au.findtext("LastName", ""), au.findtext("ForeName", "")
            if last:
                authors.append(f"{fore} {last}".strip())
        year = art.findtext(".//JournalIssue/PubDate/Year") or (art.findtext(".//JournalIssue/PubDate/MedlineDate", "")[:4])
        doi = ""
        # Only the article's own IDs: a bare .//ArticleIdList also matches every cited reference's IDs.
        for aid in art.findall("PubmedData/ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text or ""
        papers.append(Paper(
            title=title.rstrip("."), authors=authors,
            year=int(year) if year and year.isdigit() else None,
            abstract=abstract, venue=art.findtext(".//Journal/Title", ""),
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", doi=doi, source="pubmed",
        ))
    return papers


def search_pubmed(query: str, max_results: int = 8) -> list[Paper]:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    common = {"db": "pubmed"}
    if os.environ.get("NCBI_API_KEY"):
        common["api_key"] = os.environ["NCBI_API_KEY"]
    ids = _get(base + "esearch.fcgi", {**common, "term": query, "retmax": max_results,
                                        "retmode": "json", "sort": "relevance"}).json()["esearchresult"]["idlist"]
    if not ids:
        return []
    r = _get(base + "efetch.fcgi", {**common, "id": ",".join(ids), "retmode": "xml"})
    return parse_pubmed_articles(r.text)


def parse_semantic_scholar(payload: dict) -> list[Paper]:
    papers = []
    for row in payload.get("data") or []:
        if not row.get("title"):
            continue
        ext = row.get("externalIds") or {}
        papers.append(Paper(
            title=row["title"],
            authors=[a.get("name", "") for a in row.get("authors") or []],
            year=row.get("year"),
            abstract=row.get("abstract") or "",
            venue=row.get("venue") or "",
            url=row.get("url") or "",
            doi=ext.get("DOI", "") or "",
            source="semantic_scholar",
            citations=row.get("citationCount"),
        ))
    return papers


def search_semantic_scholar(query: str, max_results: int = 8) -> list[Paper]:
    headers = {"x-api-key": os.environ["S2_API_KEY"]} if os.environ.get("S2_API_KEY") else None
    r = _get("https://api.semanticscholar.org/graph/v1/paper/search", {
        "query": query, "limit": max_results,
        "fields": "title,authors,year,abstract,venue,url,externalIds,citationCount",
    }, headers=headers)
    return parse_semantic_scholar(r.json())


SEARCHERS: dict[str, Callable[[str, int], list[Paper]]] = {
    "search_arxiv": search_arxiv,
    "search_pubmed": search_pubmed,
    "search_semantic_scholar": search_semantic_scholar,
}

_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query. Keywords work better than full sentences."},
        "max_results": {"type": "integer", "description": "1-10, default 6."},
    },
    "required": ["query"],
}

TOOL_DEFS: list[dict] = [
    {"name": "search_arxiv", "description": "Search arXiv preprints (physics, maths, CS, quantitative biology, stats). Adds results to the evidence library and returns citation keys.", "input_schema": _QUERY_SCHEMA},
    {"name": "search_pubmed", "description": "Search PubMed (biomedical and life-science literature). Adds results to the evidence library and returns citation keys.", "input_schema": _QUERY_SCHEMA},
    {"name": "search_semantic_scholar", "description": "Search Semantic Scholar (all fields, includes citation counts). Adds results to the evidence library and returns citation keys.", "input_schema": _QUERY_SCHEMA},
    {"name": "get_paper", "description": "Return the full stored record (complete abstract, DOI, URL) for a citation key already in the library.",
     "input_schema": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}},
]

PYTHON_TOOL_DEF = {
    "name": "run_python",
    "description": "Run a short Python 3 script for analysis (statistics, simulation, unit checks). 60 second limit. Print results to stdout.",
    "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
}


def run_python(code: str, timeout: int = 60) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "script.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            p = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True,
                               timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            return f"ERROR: timed out after {timeout}s"
    out = (p.stdout + (f"\n[stderr]\n{p.stderr}" if p.stderr else "")).strip()
    return out[:6000] or "(no output)"


class Toolbox:
    """Dispatches tool calls from the scout agent and records every paper it surfaces."""

    def __init__(self, library: Library, allow_code: bool = False) -> None:
        self.library = library
        self.allow_code = allow_code
        self.queries: list[str] = []

    @property
    def definitions(self) -> list[dict]:
        return TOOL_DEFS + ([PYTHON_TOOL_DEF] if self.allow_code else [])

    def call(self, name: str, args: dict) -> str:
        try:
            if name in SEARCHERS:
                n = max(1, min(int(args.get("max_results", 6)), 10))
                self.queries.append(f"{name}: {args['query']}")
                found = SEARCHERS[name](args["query"], n)
                if not found:
                    return "No results. Try different keywords or another database."
                return "\n".join(self.library.add(p).short(abstract_chars=260) for p in found)
            if name == "get_paper":
                p = self.library.get(args["key"])
                if not p:
                    return f"ERROR: no paper with key {args['key']!r} in the library."
                return f"{p.short(abstract_chars=2_000)}\n    DOI: {p.doi or 'n/a'}  URL: {p.url}"
            if name == "run_python" and self.allow_code:
                return run_python(args["code"])
            return f"ERROR: unknown tool {name!r}"
        except Exception as e:  # tool failures go back to the model, not up the stack
            return f"ERROR: {type(e).__name__}: {e}"
