"""Deterministic checks that don't depend on the model being honest."""
from __future__ import annotations

import re

from .library import Library

CITE_RE = re.compile(r"\[@([A-Za-z0-9_:\-]+)\]")
# Group citations like [@a2020; @b2021] are normalised by split_groups() before matching.
GROUP_RE = re.compile(r"\[(@[^\]]+)\]")
NUM_RE = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?")


def split_groups(text: str) -> str:
    """[@a; @b] -> [@a][@b] so every citation is one token."""
    def fix(m: re.Match) -> str:
        parts = [p.strip().lstrip("@") for p in re.split(r"[;,]", m.group(1)) if p.strip()]
        return "".join(f"[@{p}]" for p in parts)
    return GROUP_RE.sub(fix, text)


def cited_keys(text: str) -> list[str]:
    seen: list[str] = []
    for k in CITE_RE.findall(split_groups(text)):
        if k not in seen:
            seen.append(k)
    return seen


def unknown_keys(text: str, library: Library) -> list[str]:
    return [k for k in cited_keys(text) if k not in library.papers]


def numbers(text: str) -> list[str]:
    """Numbers outside citation keys and heading numbering."""
    body = CITE_RE.sub("", split_groups(text))
    body = re.sub(r"^#+\s.*$", "", body, flags=re.M)
    return sorted(n.rstrip(",").replace(",", "") for n in NUM_RE.findall(body))


def compare_rewrite(before: str, after: str, library: Library) -> list[str]:
    """Problems introduced by a rewrite. Empty list means it preserved citations and numbers."""
    problems = []
    b_keys, a_keys = set(cited_keys(before)), set(cited_keys(after))
    if b_keys - a_keys:
        problems.append(f"dropped citations: {sorted(b_keys - a_keys)}")
    if a_keys - b_keys:
        problems.append(f"added citations: {sorted(a_keys - b_keys)}")
    bad = unknown_keys(after, library)
    if bad:
        problems.append(f"unknown citation keys: {bad}")
    b_nums, a_nums = numbers(before), numbers(after)
    missing = _multiset_diff(b_nums, a_nums)
    extra = _multiset_diff(a_nums, b_nums)
    if missing:
        problems.append(f"numbers changed or removed: {missing}")
    if extra:
        problems.append(f"numbers added: {extra}")
    return problems


SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(\"“])")
YEAR_RE = re.compile(r"^(19|20)\d\d$")


def _checkable(num: str) -> str | None:
    """Normalised value of a number worth checking, or None for years and small counts (word/digit ambiguity)."""
    raw = num.rstrip(",").replace(",", "")
    pct = raw.endswith("%")
    val = raw.rstrip("%").lstrip("+")
    if YEAR_RE.match(val):
        return None
    try:
        f = float(val)
    except ValueError:
        return None
    if not pct and "." not in val and abs(f) <= 10:
        return None
    return val


def ungrounded_flags(text: str, library: Library) -> list[str]:
    """Flag lines for numbers in cited sentences that do not appear in any cited abstract."""
    flags = []
    for sent in SENT_RE.split(split_groups(text)):
        keys = [k for k in CITE_RE.findall(sent) if k in library.papers]
        if not keys:
            continue
        pool = " ".join(library.papers[k].abstract for k in keys).replace(",", "")
        pool_nums = {v for n in NUM_RE.findall(pool) if (v := _checkable(n))}
        pool_nums |= {n.lstrip("+").rstrip("%") for n in NUM_RE.findall(pool)}
        body = CITE_RE.sub("", sent)
        for n in NUM_RE.findall(body):
            v = _checkable(n)
            if v and v not in pool_nums:
                claim = " ".join(body.split())[:220]
                flags.append(f"`[@{keys[0]}]` — “{claim}” — UNSUPPORTED — the number {n.strip()} does not appear in the cited abstract")
                break
    return flags


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _norm_text(s))


def overlap(text: str, library: Library, n: int = 8) -> tuple[list[str], float]:
    """Find runs of n+ identical words shared with a source abstract.

    Returns (flag lines, share of the text's words that sit inside such runs).
    """
    owner: dict[tuple, str] = {}
    for p in library.papers.values():
        w = _words(p.abstract)
        for i in range(len(w) - n + 1):
            owner.setdefault(tuple(w[i:i + n]), p.key)
    flags, covered, total = [], 0, 0
    for sent in SENT_RE.split(split_groups(text)):
        body = CITE_RE.sub("", sent)
        w = _words(body)
        total += len(w)
        hits = [i for i in range(len(w) - n + 1) if tuple(w[i:i + n]) in owner]
        if not hits:
            continue
        mask = [False] * len(w)
        for i in hits:
            for j in range(i, i + n):
                mask[j] = True
        covered += sum(mask)
        key = owner[tuple(w[hits[0]:hits[0] + n])]
        claim = " ".join(body.split())[:220]
        flags.append(f"`[@{key}]` — “{claim}” — COPIED — {sum(mask)} words repeat that abstract word for word")
    return flags, (covered / total if total else 0.0)


def _norm_text(s: str) -> str:
    s = s.lower().replace("‑", "-").replace("–", "-").replace("—", "-")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    return re.sub(r"\s+", " ", s).strip()


def quote_in_abstract(quote: str, abstract: str) -> bool:
    q = _norm_text(quote).strip(" .…\"'")
    return len(q) >= 25 and q in _norm_text(abstract)


def parse_flag(line: str) -> dict | None:
    """`[@key] — “claim” — VERDICT — why`  ->  dict, or None if the line is not a flag."""
    clean = line.replace("*", "").replace("`", "").strip()
    parts = [p.strip() for p in clean.split(" — ")]
    m = re.match(r"^\[@([^\]]+)\]", parts[0]) if parts else None
    if not m or len(parts) < 3:
        return None
    verdict = next((v for v in ("UNSUPPORTED", "OVERSTATED", "COPIED") if v in clean), "UNSUPPORTED")
    claim = parts[1].strip("“”\"' …")
    why = parts[3] if len(parts) > 3 else ""
    return {"key": m.group(1), "claim": claim, "verdict": verdict, "why": why}


def _multiset_diff(a: list[str], b: list[str]) -> list[str]:
    remaining = list(b)
    out = []
    for x in a:
        if x in remaining:
            remaining.remove(x)
        else:
            out.append(x)
    return out
