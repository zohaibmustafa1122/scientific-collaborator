"""Evidence library: every paper the agents may cite, keyed so citations can be checked."""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Paper:
    title: str
    authors: list[str]
    year: int | None
    abstract: str = ""
    venue: str = ""
    url: str = ""
    doi: str = ""
    source: str = ""
    citations: int | None = None
    key: str = ""
    extra_ids: dict = field(default_factory=dict)

    def line(self) -> str:
        who = self.authors[0].split()[-1] if self.authors else "Unknown"
        who += " et al." if len(self.authors) > 1 else ""
        return f"[@{self.key}] {who} ({self.year or 'n.d.'}). {self.title[:140]}"

    def short(self, abstract_chars: int = 500) -> str:
        who = self.authors[0].split()[-1] if self.authors else "Unknown"
        if len(self.authors) > 1:
            who += " et al."
        abstract = self.abstract.strip().replace("\n", " ")
        if len(abstract) > abstract_chars:
            abstract = abstract[:abstract_chars].rstrip() + "..."
        cited = f", {self.citations} citations" if self.citations is not None else ""
        return (
            f"[@{self.key}] {who} ({self.year or 'n.d.'}). {self.title}. "
            f"{self.venue or self.source}{cited}\n    {abstract or '(no abstract available)'}"
        )

    def reference(self) -> str:
        names = self.authors[:6]
        authors = ", ".join(names) + (", et al." if len(self.authors) > 6 else "")
        parts = [f"{authors or 'Unknown'} ({self.year or 'n.d.'}). {self.title}."]
        if self.venue:
            parts.append(f"{self.venue}.")
        if self.doi:
            parts.append(f"https://doi.org/{self.doi}")
        elif self.url:
            parts.append(self.url)
        return " ".join(parts)


def _slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", text.lower()) or "anon"


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


class Library:
    def __init__(self) -> None:
        self.papers: dict[str, Paper] = {}
        self._by_title: dict[str, str] = {}
        self._by_doi: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.papers)

    def add(self, paper: Paper) -> Paper:
        """Add a paper, merging with an existing copy of the same work."""
        doi = paper.doi.lower().strip()
        existing_key = self._by_doi.get(doi) if doi else None
        existing_key = existing_key or self._by_title.get(_norm_title(paper.title))
        if existing_key:
            old = self.papers[existing_key]
            for attr in ("abstract", "venue", "url", "doi"):
                if not getattr(old, attr) and getattr(paper, attr):
                    setattr(old, attr, getattr(paper, attr))
            if paper.citations is not None:
                old.citations = max(old.citations or 0, paper.citations)
            return old
        paper.key = self._new_key(paper)
        self.papers[paper.key] = paper
        self._by_title[_norm_title(paper.title)] = paper.key
        if doi:
            self._by_doi[doi] = paper.key
        return paper

    def _new_key(self, paper: Paper) -> str:
        first = _slug(paper.authors[0].split()[-1]) if paper.authors else "anon"
        base = f"{first}{paper.year or 'nd'}"
        key, suffix = base, 97  # 'a'
        while key in self.papers:
            key = base + chr(suffix)
            suffix += 1
        return key

    def get(self, key: str) -> Paper | None:
        return self.papers.get(key.strip().lstrip("@"))

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps([asdict(p) for p in self.papers.values()], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Library":
        lib = cls()
        for row in json.loads(path.read_text(encoding="utf-8")):
            paper = Paper(**row)
            lib.papers[paper.key] = paper
            lib._by_title[_norm_title(paper.title)] = paper.key
            if paper.doi:
                lib._by_doi[paper.doi.lower()] = paper.key
        return lib

    def references(self, keys: list[str]) -> str:
        rows = [self.papers[k] for k in keys if k in self.papers]
        rows.sort(key=lambda p: (p.authors[0].split()[-1].lower() if p.authors else "", p.year or 0))
        return "\n".join(f"- **{p.key}** — {p.reference()}" for p in rows)
