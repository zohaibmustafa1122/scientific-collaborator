"""Render a finished manuscript (markdown + library) as a PDF with the author's details on it."""
from __future__ import annotations

import html
import io
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

from . import verify
from .library import Library

DISCLOSURE = (
    "Drafting note: this manuscript was produced with AI assistance from a literature search "
    "(arXiv, PubMed, Semantic Scholar) and was checked against paper abstracts only, not full texts. "
    "The author is responsible for verifying its content before relying on or submitting it."
)

FONT_SETS = [  # (regular, bold, italic, bold-italic)
    ("C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/timesbd.ttf", "C:/Windows/Fonts/timesi.ttf", "C:/Windows/Fonts/timesbi.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf"),
    ("/Library/Fonts/Times New Roman.ttf", "/Library/Fonts/Times New Roman Bold.ttf",
     "/Library/Fonts/Times New Roman Italic.ttf", "/Library/Fonts/Times New Roman Bold Italic.ttf"),
]


@dataclass
class Author:
    name: str
    affiliation: str = ""
    email: str = ""
    orcid: str = ""

    def clean(self) -> "Author":
        f = lambda s, n: re.sub(r"\s+", " ", (s or "")).strip()[:n]
        return Author(f(self.name, 120), f(self.affiliation, 200), f(self.email, 120), f(self.orcid, 40))


def _register_fonts() -> tuple[str, bool]:
    """Return (family name, unicode_ok). Falls back to built-in Times, which only covers Latin-1."""
    for i, paths in enumerate(FONT_SETS):
        if all(Path(p).exists() for p in paths):
            names = [f"Body{i}", f"Body{i}-B", f"Body{i}-I", f"Body{i}-BI"]
            for n, p in zip(names, paths):
                if n not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(n, p))
            pdfmetrics.registerFontFamily(names[0], normal=names[0], bold=names[1], italic=names[2], boldItalic=names[3])
            return names[0], True
    return "Times-Roman", False


_SUBS = {"‑": "-", "‐": "-", "−": "-", " ": " ", "≥": ">=", "≤": "<=", "×": "x"}


def _sanitize(text: str, unicode_ok: bool) -> str:
    text = "".join(_SUBS.get(c, c) for c in text)
    if unicode_ok:
        return text
    out = []
    for c in text:
        try:
            c.encode("cp1252")
            out.append(c)
        except UnicodeEncodeError:
            out.append(unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode() or "?")
    return "".join(out)


def _surname(name: str) -> str:
    return name.split()[-1] if name.split() else "Unknown"


def _author_year(lib: Library, key: str) -> str:
    p = lib.papers.get(key)
    if not p:
        return "?"
    names = [_surname(a) for a in p.authors]
    who = names[0] if len(names) == 1 else (f"{names[0]} & {names[1]}" if len(names) == 2 else f"{names[0]} et al.") if names else "Unknown"
    return f"{who}, {p.year or 'n.d.'}"


_RUN = re.compile(r"(?:\[@[^\]]+\])+")


def _cites_to_author_year(text: str, lib: Library) -> str:
    return _RUN.sub(lambda m: "(" + "; ".join(_author_year(lib, k) for k in verify.cited_keys(m.group(0))) + ")",
                    verify.split_groups(text))


def _inline(text: str, lib: Library, unicode_ok: bool) -> str:
    text = _sanitize(_cites_to_author_year(text, lib), unicode_ok)
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    return text


def build_pdf(markdown: str, library: Library, author: Author, disclosure: bool = True) -> bytes:
    author = author.clean()
    if not author.name:
        raise ValueError("an author name is required")
    family, uni = _register_fonts()
    body_md, _, _ = markdown.partition("\n## References")
    title_m = re.match(r"\s*#\s+(.+)", body_md)
    title = title_m.group(1).strip() if title_m else "Untitled"
    if title_m:
        body_md = body_md[title_m.end():]

    base = ParagraphStyle("base", fontName=family, fontSize=11, leading=15, alignment=TA_JUSTIFY, spaceAfter=7)
    st = {
        "base": base,
        "title": ParagraphStyle("t", parent=base, fontSize=19, leading=24, alignment=TA_CENTER, spaceAfter=10),
        "author": ParagraphStyle("a", parent=base, fontSize=12, leading=16, alignment=TA_CENTER, spaceAfter=2),
        "meta": ParagraphStyle("m", parent=base, fontSize=9.5, leading=13, alignment=TA_CENTER, spaceAfter=1, textColor="#444444"),
        "h2": ParagraphStyle("h2", parent=base, fontSize=14, leading=18, alignment=0, spaceBefore=12, spaceAfter=5, keepWithNext=1),
        "h3": ParagraphStyle("h3", parent=base, fontSize=12, leading=16, alignment=0, spaceBefore=8, spaceAfter=4, keepWithNext=1),
        "li": ParagraphStyle("li", parent=base, leftIndent=18, bulletIndent=6, spaceAfter=3),
        "ref": ParagraphStyle("r", parent=base, fontSize=9.5, leading=12.5, leftIndent=16, firstLineIndent=-16, alignment=0, spaceAfter=3),
        "note": ParagraphStyle("n", parent=base, fontSize=8.5, leading=11, textColor="#555555", alignment=0),
    }
    bold = f"{family}-B" if uni else "Times-Bold"
    for name in ("title", "h2", "h3"):
        st[name].fontName = bold
    S = lambda t, s: Paragraph(_inline(t, library, uni), st[s])

    flow = [S(title, "title"), S(f"**{author.name}**", "author")]
    for line in (author.affiliation, author.email, f"ORCID: {author.orcid}" if author.orcid else "", time.strftime("%d %B %Y")):
        if line:
            flow.append(Paragraph(_inline(line, library, uni), st["meta"]))
    flow += [Spacer(1, 8), HRFlowable(width="100%", thickness=0.6, color="#999999"), Spacer(1, 8)]

    para: list[str] = []

    def flush() -> None:
        if para:
            flow.append(S(" ".join(para), "base"))
            para.clear()

    for raw in body_md.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.strip() == "---" or re.fullmatch(r"#{1,3}\s*Section\s*", line.strip()):
            flush()  # (a bare "Section" heading is a placeholder some models emit)
        elif line.startswith("## "):
            flush(); flow.append(S(line[3:].strip(), "h2"))
        elif line.startswith("### ") or line.startswith("#### "):
            flush(); flow.append(S(line.lstrip("#").strip(), "h3"))
        elif m := re.match(r"^\s*(?:[-*]|\d+[.)])\s+(.*)", line):
            flush()
            bullet = "•" if re.match(r"^\s*[-*]\s", line) else (re.match(r"^\s*(\d+)", line).group(1) + ".")
            flow.append(Paragraph(_inline(m.group(1), library, uni), st["li"], bulletText=bullet))
        else:
            para.append(line.strip())
    flush()

    cited = verify.cited_keys(markdown.partition("\n## References")[0])
    refs = sorted((library.papers[k] for k in cited if k in library.papers),
                  key=lambda p: ((p.authors[0].split()[-1].lower() if p.authors else ""), p.year or 0))
    if refs:
        flow.append(S("References", "h2"))
        flow += [Paragraph(_inline(p.reference(), library, uni), st["ref"]) for p in refs]
    if disclosure:
        flow += [Spacer(1, 10), HRFlowable(width="100%", thickness=0.4, color="#bbbbbb"), Spacer(1, 4),
                 Paragraph(_inline(DISCLOSURE, library, uni), st["note"])]

    short = _sanitize(title, uni)[:70]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(family, 8.5)
        canvas.setFillGray(0.4)
        canvas.drawString(2.2 * cm, 1.2 * cm, f"{_sanitize(author.name, uni)}  ·  {short}")
        canvas.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.2 * cm,
                      bottomMargin=2.2 * cm, title=_sanitize(title, uni), author=_sanitize(author.name, uni),
                      subject="Research manuscript").build(flow, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
