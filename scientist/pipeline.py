"""The pipeline: scout -> discover -> write -> humanize -> verify.

Free-tier models have small per-request token limits, so everything after the draft works one
section at a time. That also gives finer fallbacks: a bad rewrite of one section only reverts
that section.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import llm, verify
from .library import Library
from .tools import Toolbox

SKILLS = Path(__file__).parent / "skills"
KINDS = {
    "paper": "a research paper (a position or proposal-style paper built on existing literature)",
    "review": "a narrative literature review",
    "proposal": "a grant proposal",
}


def skill(*names: str) -> str:
    return "\n\n".join((SKILLS / f"{n}.md").read_text(encoding="utf-8") for n in names)


def _titles(library: Library, max_chars: int = 3000) -> str:
    out, used = [], 0
    for p in library.papers.values():
        line = p.line()
        if used + len(line) > max_chars:
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)


def _sources(library: Library, keys: list[str], abstract_chars: int, max_chars: int) -> str:
    out, used = [], 0
    for k in keys:
        p = library.papers.get(k)
        if not p:
            continue
        row = p.short(abstract_chars)
        if used + len(row) > max_chars:
            break
        out.append(row)
        used += len(row) + 1
    return "\n".join(out)


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n].rsplit("\n", 1)[0] + "\n[...truncated]"


def split_sections(md: str) -> list[str]:
    """Split on level-2 headings; the title and any lead-in stay with the first chunk."""
    parts = re.split(r"(?m)^(?=## )", md.strip())
    return [p.strip("\n") for p in parts if p.strip()]


@dataclass
class Result:
    out_dir: Path
    final: str
    report: str
    ok: bool


def run(
    topic: str,
    out_dir: Path,
    kind: str = "review",
    max_turns: int = 20,
    allow_code: bool = False,
    log: Callable[[str], None] = print,
) -> Result:
    out_dir.mkdir(parents=True, exist_ok=True)
    library = Library()
    toolbox = Toolbox(library, allow_code=allow_code)
    llm.wait_hook = log

    # 1. Scout: autonomous literature search -----------------------------------------------
    log(f"[1/5] Scouting literature on: {topic}")
    evidence = llm.run_tool_agent(
        system=(
            "You are a careful research scout.\n\n" + skill("agent_skills")
            + "\n\nCite only keys returned by your tools. Keep the final notes under 700 words."
        ),
        user=f"Topic / question: {topic}\n\nGather the evidence base for writing {KINDS[kind]}. You have at most {max_turns} turns.",
        tools=toolbox.definitions,
        dispatch=toolbox.call,
        max_turns=max_turns,
        on_tool=lambda name, args: log(f"      {name}({args.get('query') or args.get('key') or '...'})"),
    )
    library.save(out_dir / "library.json")
    (out_dir / "1_evidence.md").write_text(evidence, encoding="utf-8")
    log(f"      library: {len(library)} papers")
    if not len(library):
        raise RuntimeError("The scout found no papers (network or API problem?). Nothing to write from.")
    evidence = _clip(evidence, 5000)

    # 2. Discover: cross-disciplinary hypotheses -------------------------------------------
    log("[2/5] Looking for cross-disciplinary connections and hypotheses")
    hypotheses = llm.complete(
        system="You are a cross-disciplinary research analyst.\n\n" + skill("superpowers"),
        user=(
            f"Topic: {topic}\n\n## Evidence notes\n{evidence}\n\n## Library (citable keys)\n{_titles(library)}\n\n"
            "Produce, in under 600 words: (1) 3-4 cross-cutting patterns, (2) 3 hypotheses in the required format, "
            "(3) the main gaps in the evidence. Markdown."
        ),
        max_tokens=2500,
    )
    (out_dir / "2_hypotheses.md").write_text(hypotheses, encoding="utf-8")

    # 3. Write: rigorous structured draft --------------------------------------------------
    log(f"[3/5] Drafting {kind}")
    facts = (
        f"- Databases queried: {', '.join(sorted({q.split(':')[0].replace('search_', '') for q in toolbox.queries}))}\n"
        f"- Queries run: {len(toolbox.queries)}\n- Papers in library: {len(library)} (abstracts only; full texts were not read)"
    )
    cards = _cards(library, log)
    (out_dir / "2b_cards.json").write_text(json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8")
    if cards:
        grounding = (
            "## Citable facts (each quote was checked word for word against its abstract)\n"
            + "\n".join(f"- [@{c['key']}] {c['claim']}  (quote: “{c['quote'][:170]}”)" for c in cards)
            + "\n\nRULE: every sentence that carries a [@key] must restate one of these facts and nothing more. "
              "Use no number that is not in the fact or its quote. Put anything else in uncited sentences, "
              "worded as your own inference (\"we infer\", \"this suggests\")."
        )
    else:  # extraction failed: fall back to the scout's notes
        log("      no verified fact cards; writing from scout notes")
        grounding = f"## Evidence notes\n{evidence}\n\n## Library (citable keys)\n{_titles(library, 2500)}"
    draft = verify.split_groups(llm.complete(
        system="You are an academic writer.\n\n" + skill("academic_paper"),
        user=(
            f"Write {KINDS[kind]} on: {topic}\n\nKind: {kind}. Target about 1000 words. "
            f"Use real topic headings (never a heading that just says 'Section').\n\n"
            f"## Search facts (the only facts you may state about the method)\n{facts}\n\n"
            f"{grounding}\n\n## Hypotheses (label as hypotheses; do not cite them as findings)\n{_clip(hypotheses, 1800)}\n\n"
            "Output markdown only, with a `# Title` line."
        ),
        max_tokens=3600,
    ))
    draft = re.sub(r"(?m)^#{1,3} Section\s*\n+(?=#)", "", draft)  # weaker models sometimes emit a placeholder heading
    (out_dir / "3_draft.md").write_text(draft, encoding="utf-8")

    # 4. Humanize, section by section, guarded by code -------------------------------------
    log("[4/5] Humanizing prose")
    sections, problems = [], []
    for sec in split_sections(draft):
        new, probs = _humanize(sec, library)
        sections.append(new)
        problems += probs
    final = "\n\n".join(sections)
    (out_dir / "4_humanized.md").write_text(final, encoding="utf-8")

    # 5. Verify against the abstracts, revise unsupported claims, re-check -----------------
    log("[5/5] Verifying claims against abstracts")
    audit, flagged, first_audit = "", [], ""
    revised_parts = []
    for sec in split_sections(final):
        a = _audit(sec, library)
        fl = _flags(a)
        if fl:
            first_audit += f"{a}\n"
            log(f"      {len(fl)} unsupported claim(s) in: {sec.splitlines()[0][:50]}; revising")
            sec2 = _revise(sec, a, library)
            if sec2 != sec:
                a2 = _audit(sec2, library)
                if len(_flags(a2)) <= len(fl):
                    sec, a, fl = sec2, a2, _flags(a2)
        audit += f"{a}\n" if fl else ""
        flagged += fl
        revised_parts.append(sec)
    final = "\n\n".join(revised_parts)
    audit = audit.strip() or "NO PROBLEMS"
    (out_dir / "4_humanized.md").write_text(final, encoding="utf-8")
    (out_dir / "flags.json").write_text(json.dumps({
        "remaining": [f for f in map(verify.parse_flag, flagged) if f],
        "fixed": max(len(_flags(first_audit)) - len(flagged), 0),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    keys = verify.cited_keys(final)
    unknown = verify.unknown_keys(final, library)
    report = _report(topic, library, keys, unknown, problems, audit, toolbox.queries, first_audit.strip(),
                     overlap_pct=100 * verify.overlap(final, library)[1])
    (out_dir / "5_verification.md").write_text(report, encoding="utf-8")
    final_doc = f"{final.rstrip()}\n\n## References\n\n{library.references(keys)}\n"
    (out_dir / "final.md").write_text(final_doc, encoding="utf-8")
    ok = not unknown and not problems and not flagged
    if flagged:
        log(f"      {len(flagged)} flagged claim(s) remain after revision")
    log(f"Done: {out_dir / 'final.md'}  (checks {'passed' if ok else 'raised warnings - see 5_verification.md'})")
    return Result(out_dir, final_doc, report, ok)


def _humanize(section: str, library: Library, attempts: int = 2) -> tuple[str, list[str]]:
    system = "You are an expert scientific editor.\n\n" + skill("humanize")
    feedback, problems = "", ["not attempted"]
    for _ in range(attempts):
        candidate = verify.split_groups(llm.complete(
            system=system,
            user=f"{feedback}Rewrite this section. Output the full markdown of the section only.\n\n{section}",
            max_tokens=2600,
        ))
        problems = verify.compare_rewrite(section, candidate, library)
        if not problems:
            return candidate, []
        feedback = "Your previous rewrite broke hard constraints: " + "; ".join(problems) + ". Fix these.\n\n"
    # Never ship a rewrite that lost citations or changed numbers: keep the checked original.
    head = section.splitlines()[0][:40]
    return section, [f"kept unedited section '{head}': {p}" for p in problems]


def _cards(library: Library, log, per_call: int = 3, max_papers: int = 14) -> list[dict]:
    """Findings pulled from abstracts. A card survives only if its quote appears verbatim in the abstract."""
    papers = [p for p in library.papers.values() if len(p.abstract) >= 200][:max_papers]
    cards: list[dict] = []
    log(f"      extracting verified facts from {len(papers)} abstracts")
    for i in range(0, len(papers), per_call):
        batch = papers[i:i + per_call]
        text = "\n\n".join(f"KEY: {p.key}\nABSTRACT: {p.abstract[:1100]}" for p in batch)
        raw = llm.complete(
            system=(
                "You extract findings from abstracts. Output ONLY a JSON list of objects "
                '{"key": ..., "quote": ..., "claim": ...}, one or two per paper. "quote" must be copied '
                'word for word from that paper\'s abstract (one sentence or clause, 40-200 characters). "claim" is a '
                "short paraphrase that says no more than the quote. Skip papers whose abstract states no concrete "
                "finding. No prose, no markdown fences."
            ),
            user=text,
            max_tokens=1500,
        )
        for c in _parse_json_list(raw):
            p = library.papers.get(str(c.get("key", "")).lstrip("@"))
            if p and isinstance(c.get("quote"), str) and isinstance(c.get("claim"), str) \
                    and verify.quote_in_abstract(c["quote"], p.abstract):
                cards.append({"key": p.key, "quote": c["quote"].strip(), "claim": c["claim"].strip()})
    log(f"      kept {len(cards)} verified facts")
    return cards


def _parse_json_list(raw: str) -> list[dict]:
    m = re.search(r"\[.*\]", raw, re.S)
    try:
        data = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)]


def _audit(section: str, library: Library) -> str:
    keys = verify.cited_keys(section)
    if not keys:
        return "NO PROBLEMS"
    det = "\n".join(verify.ungrounded_flags(section, library) + verify.overlap(section, library)[0])  # code-checked
    return (_model_audit(section, keys, library) + "\n" + det).strip()


def _model_audit(section: str, keys: list[str], library: Library) -> str:
    return llm.complete(
        system=(
            "You are a skeptical reviewer checking claims against sources. You only have the stored abstracts. "
            "For each citation use, decide: SUPPORTED, UNSUPPORTED (abstract does not say this) or OVERSTATED. "
            "List only the problems, one per line, each as: `[@key]` — quoted claim — UNSUPPORTED or OVERSTATED — why. "
            "If there are none, reply exactly: NO PROBLEMS"
        ),
        user=f"## Text\n{section}\n\n## Sources\n{_sources(library, keys, 1200, 6000)}",
        max_tokens=1200,
    )


def _flags(audit: str) -> list[str]:
    return [l for l in audit.splitlines() if re.search(r"\b(UNSUPPORTED|OVERSTATED|COPIED)\b", l.replace("*", ""))]


def _revise(section: str, audit: str, library: Library) -> str:
    keys = verify.cited_keys(section)
    revised = verify.split_groups(llm.complete(
        system=(
            "You are a careful scientific editor. Fix a section so every cited claim is directly stated by the "
            "cited abstract. For each flagged claim: delete it, or rewrite it to say only what the abstract says, "
            "or keep it as an uncited inference clearly labelled as such. For a COPIED flag, restate the idea in "
            "your own words and structure, keeping the citation and every number. Leave everything else unchanged. "
            "Output the full markdown of the section only."
        ),
        user=f"## Section\n{section}\n\n## Flagged claims\n{_clip(audit, 2000)}\n\n## Sources\n{_sources(library, keys, 700, 3500)}",
        max_tokens=2600,
    ))
    return section if verify.unknown_keys(revised, library) else revised


def _report(topic, library, keys, unknown, problems, audit, queries, first_audit="", overlap_pct=0.0) -> str:
    lines = [
        f"# Verification report: {topic}",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Programmatic checks",
        f"- Papers in library: {len(library)}; cited in final: {len(keys)}",
        f"- Unknown citation keys: {unknown or 'none'}",
        f"- Text copied word for word from source abstracts (runs of 8+ words): {overlap_pct:.1f}% of the words",
        f"- Humanizer integrity: {'; '.join(problems) if problems else 'citations and numbers preserved'}",
        "",
        "## Claim audit of the final text (model-assisted, abstracts only)",
        audit,
        "",
        *(["## Claims flagged in the first draft (revised where possible)", first_audit, ""] if first_audit else []),
        "## Search log",
        *[f"- {q}" for q in queries],
        "",
        "> Only abstracts were available to the system. Read the full papers before relying on any claim.",
    ]
    return "\n".join(lines)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50] or "run"
