import io

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

import server
from scientist import pdf
from scientist.library import Library, Paper

MD = """# Gut Flora and Fasting: A Review

## Abstract

We reviewed **three** studies. Non‑breaking hyphens and ≥ signs must not break the PDF [@lovelace2021].

## Findings

- First point cites two papers [@lovelace2021; @curie2019].
- Second point has *emphasis*.

1. Numbered item with & ampersand <tag>

## References

- **lovelace2021** — old reference text that must be replaced
"""


def _lib():
    lib = Library()
    lib.add(Paper("Graph Networks", ["Ada Lovelace", "Alan Turing"], 2021, venue="arXiv", doi="10.1/abc"))
    lib.add(Paper("A trial", ["Marie Curie", "A B", "C D"], 2019, venue="J Test"))
    return lib


def _text(data: bytes) -> str:
    raw = "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(data)).pages)
    return " ".join(raw.split())  # line wrapping must not matter


def test_pdf_has_author_details_and_author_year_citations():
    data = pdf.build_pdf(MD, _lib(), pdf.Author("Jane Q. Researcher", "Dept of Biology, Example University",
                                                 "jane@example.org", "0000-0001-2345-6789"))
    assert data.startswith(b"%PDF")
    text = _text(data)
    for expected in ["Gut Flora and Fasting", "Jane Q. Researcher", "Dept of Biology, Example University",
                     "jane@example.org", "ORCID: 0000-0001-2345-6789", "(Lovelace & Turing, 2021)",
                     "(Lovelace & Turing, 2021; Curie et al., 2019)", "References", "Drafting note"]:
        assert expected in text, expected
    assert "[@" not in text and "old reference text" not in text
    assert PdfReader(io.BytesIO(data)).metadata.author == "Jane Q. Researcher"


def test_headings_are_bold_and_placeholder_headings_dropped():
    reader = PdfReader(io.BytesIO(pdf.build_pdf(MD.replace("## Findings", "## Section\n## Findings"), _lib(),
                                                pdf.Author("Jane Doe"))))
    fonts = {f.get("/BaseFont", "") for p in reader.pages for f in
             (v.get_object() for v in p["/Resources"]["/Font"].values())}
    assert any("Bold" in str(f) or f.endswith("-B") or "timesbd" in str(f).lower() for f in fonts) or len(fonts) >= 2
    text = " ".join(" ".join(p.extract_text().split()) for p in reader.pages)
    assert "Section" not in text and "Findings" in text


def test_disclosure_can_be_turned_off_and_name_is_required():
    text = _text(pdf.build_pdf(MD, _lib(), pdf.Author("Jane Doe"), disclosure=False))
    assert "Drafting note" not in text
    with pytest.raises(ValueError):
        pdf.build_pdf(MD, _lib(), pdf.Author("   "))


def test_pdf_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "RUNS", tmp_path)
    run = tmp_path / "20260101-000000-demo"
    run.mkdir()
    (run / "final.md").write_text(MD, encoding="utf-8")
    _lib().save(run / "library.json")
    c = TestClient(server.app)
    r = c.post(f"/api/runs/{run.name}/pdf", json={"name": "Jane Doe", "affiliation": "Uni"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert "Jane Doe" in _text(r.content)
    assert c.post(f"/api/runs/{run.name}/pdf", json={"name": "J"}).status_code == 422
    assert c.post("/api/runs/nope/pdf", json={"name": "Jane Doe"}).status_code == 404
