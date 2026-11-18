from pathlib import Path

from scientist import llm, pipeline, verify
from scientist.library import Library, Paper
from scientist.tools import Toolbox, parse_arxiv, parse_pubmed_articles, parse_semantic_scholar

ARXIV = """<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/1</id><published>2021-03-01T00:00:00Z</published>
<title>Graph  Networks
 for Drugs</title><summary>We study X. Accuracy was 91%.</summary>
<author><name>Ada Lovelace</name></author><author><name>Alan Turing</name></author>
<arxiv:doi>10.1/abc</arxiv:doi></entry></feed>"""

PUBMED = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>42</PMID><Article>
<Journal><Title>J Test</Title><JournalIssue><PubDate><Year>2019</Year></PubDate></JournalIssue></Journal>
<ArticleTitle>A trial of <i>things</i>.</ArticleTitle>
<Abstract><AbstractText Label="A">Part one.</AbstractText><AbstractText>Part two.</AbstractText></Abstract>
<AuthorList><Author><LastName>Curie</LastName><ForeName>Marie</ForeName></Author></AuthorList>
</Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.2/xyz</ArticleId></ArticleIdList></PubmedData>
</PubmedArticle></PubmedArticleSet>"""


def test_pubmed_doi_ignores_reference_list_ids():
    xml = PUBMED.replace(
        "</PubmedData>",
        '<ReferenceList><Reference><ArticleIdList><ArticleId IdType="doi">10.9/WRONG</ArticleId></ArticleIdList>'
        "</Reference></ReferenceList></PubmedData>")
    assert parse_pubmed_articles(xml)[0].doi == "10.2/xyz"


def test_parsers():
    a = parse_arxiv(ARXIV)[0]
    assert a.title == "Graph Networks for Drugs" and a.year == 2021 and a.doi == "10.1/abc"
    p = parse_pubmed_articles(PUBMED)[0]
    assert p.title == "A trial of things" and p.abstract == "Part one. Part two." and p.doi == "10.2/xyz"
    s = parse_semantic_scholar({"data": [{"title": "T", "year": 2020, "authors": [{"name": "A B"}],
                                          "externalIds": {"DOI": "10.3/q"}, "citationCount": 7}]})[0]
    assert s.citations == 7 and s.doi == "10.3/q"


def test_library_dedupes_and_keys():
    lib = Library()
    one = lib.add(Paper("Same Title", ["Ada Lovelace"], 2021, doi="10.1/abc"))
    two = lib.add(Paper("same title!", ["A. Lovelace"], 2021, abstract="more", citations=5))
    other = lib.add(Paper("Another", ["Ada Lovelace"], 2021))
    assert one is two and one.abstract == "more" and one.citations == 5
    assert (one.key, other.key) == ("lovelace2021", "lovelace2021a")


def test_citation_guard():
    lib = Library()
    lib.add(Paper("T", ["A Smith"], 2020))
    assert verify.cited_keys("x [@smith2020; @jones1999] y") == ["smith2020", "jones1999"]
    assert verify.unknown_keys("x [@smith2020] [@jones1999]", lib) == ["jones1999"]
    before = "Smith found 91% accuracy in 120 patients [@smith2020]."
    assert verify.compare_rewrite(before, "Smith found 91% accuracy in 120 patients [@smith2020].", lib) == []
    bad = verify.compare_rewrite(before, "Smith found 95% accuracy in patients.", lib)
    assert any("citations" in p for p in bad) and any("numbers" in p for p in bad)


def test_toolbox_adds_to_library(monkeypatch):
    lib = Library()
    monkeypatch.setitem(__import__("scientist.tools", fromlist=["x"]).SEARCHERS, "search_arxiv",
                        lambda q, n: parse_arxiv(ARXIV))
    out = Toolbox(lib).call("search_arxiv", {"query": "x"})
    assert "[@lovelace2021]" in out and len(lib) == 1
    assert Toolbox(lib).call("run_python", {"code": "print(1)"}).startswith("ERROR: unknown tool")  # off by default
    assert "lovelace2021" in Toolbox(lib).call("get_paper", {"key": "@lovelace2021"})


def test_python_runner_opt_in():
    assert Toolbox(Library(), allow_code=True).call("run_python", {"code": "print(2+2)"}) == "4"


def test_unsupported_claims_trigger_revision(monkeypatch, tmp_path: Path):
    import scientist.tools as t
    monkeypatch.setitem(t.SEARCHERS, "search_arxiv", lambda q, n: parse_arxiv(ARXIV))
    monkeypatch.setattr(llm, "run_tool_agent", lambda *a, dispatch, **k: dispatch("search_arxiv", {"query": "q"}) and "n")
    audits = []

    def complete(system, user, **kw):
        if "skeptical reviewer" in system:
            audits.append(1)
            return "[@lovelace2021] — 'a claim' — UNSUPPORTED — not in abstract" if len(audits) == 1 else "NO PROBLEMS"
        if "careful scientific editor" in system:
            return "# T\n\nAccuracy was 91% [@lovelace2021]. (revised)"
        return "# T\n\nAccuracy was 91% [@lovelace2021]. A claim [@lovelace2021]."

    monkeypatch.setattr(llm, "complete", complete)
    res = pipeline.run("t", tmp_path, kind="paper", log=lambda *_: None)
    assert res.ok and "(revised)" in res.final and len(audits) == 2
    assert "first draft" in res.report


def _lib():
    lib = Library()
    lib.add(Paper("Deep Cas", ["Ada Lovelace"], 2021,
                  abstract="We trained a CNN on 4,500 guides. Accuracy reached 91% on held-out data and the model ran in 12 minutes."))
    return lib


def test_ungrounded_numbers_are_flagged():
    lib = _lib()
    ok = "The CNN reached 91% accuracy on held-out data [@lovelace2021]. Training used 4,500 guides [@lovelace2021]."
    assert verify.ungrounded_flags(ok, lib) == []
    bad = "The CNN reached 95% accuracy [@lovelace2021]. Uncited 77% numbers are not checked. It gave an AUROC of 0.89 [@lovelace2021]."
    flags = verify.ungrounded_flags(bad, lib)
    assert len(flags) == 2 and "95%" in flags[0] and "0.89" in flags[1]
    assert verify.ungrounded_flags("Lovelace (2021) ran 3 experiments [@lovelace2021].", lib) == []  # years/small counts ignored


def test_copied_runs_are_flagged_and_measured():
    lib = _lib()
    copied = "As reported, accuracy reached 91% on held-out data and the model ran in 12 minutes [@lovelace2021]."
    flags, share = verify.overlap(copied, lib)
    assert len(flags) == 1 and "COPIED" in flags[0] and share > 0.5
    own = "The network performed well on unseen examples, finishing quickly [@lovelace2021]."
    assert verify.overlap(own, lib) == ([], 0.0)
    assert verify.parse_flag(flags[0])["verdict"] == "COPIED"


def test_quote_must_appear_verbatim_and_flag_parsing():
    abstract = "We trained a CNN on 4,500 guides. Accuracy reached 91% on held-out data."
    assert verify.quote_in_abstract("Accuracy reached 91% on held-out data", abstract)
    assert not verify.quote_in_abstract("Accuracy reached 95% on held-out data", abstract)
    f = verify.parse_flag("`[@lovelace2021]` — “Accuracy hit 95%” — **UNSUPPORTED** — the number 95% does not appear")
    assert f == {"key": "lovelace2021", "claim": "Accuracy hit 95%", "verdict": "UNSUPPORTED",
                 "why": "the number 95% does not appear"}
    assert verify.parse_flag("NO PROBLEMS") is None


def test_cards_drop_fabricated_quotes(monkeypatch):
    lib = _lib()
    lib.papers["lovelace2021"].abstract += " " + "x" * 200
    reply = ('```json\n[{"key":"lovelace2021","quote":"Accuracy reached 91% on held-out data","claim":"91% accuracy"},'
             '{"key":"lovelace2021","quote":"The model beat every baseline by 12%","claim":"beats baselines"}]\n```')
    monkeypatch.setattr(llm, "complete", lambda *a, **k: reply)
    cards = pipeline._cards(lib, lambda *_: None)
    assert [c["claim"] for c in cards] == ["91% accuracy"]


def test_daily_quota_moves_to_next_model_then_fails_fast(monkeypatch):
    class R:
        def __init__(self, code, text="", data=None):
            self.status_code, self.text, self.headers, self._d = code, text, {}, data
        def json(self):
            return self._d

    used = []

    def post(url, headers, json, timeout):
        used.append(json["model"])
        if json["model"] == "a":
            return R(429, "Quota exceeded: Generate content requests per day limit exceeded on Free Tier")
        if json["model"] == "b":
            return R(429, "RPD limit reached")
        return R(200, data={"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]})

    monkeypatch.setattr(llm.requests, "post", post)
    monkeypatch.setattr(llm, "MODELS", ["a", "b", "c"])
    monkeypatch.setattr(llm, "_exhausted", set())
    monkeypatch.setattr(llm, "wait_hook", None)
    assert llm.complete("s", "u", max_tokens=100) == "hi"
    assert used == ["a", "b", "c"]
    # all models exhausted -> clear error right away instead of retrying for minutes
    monkeypatch.setattr(llm, "_exhausted", {"a", "b", "c"})
    import pytest
    with pytest.raises(llm.DailyQuotaExhausted):
        llm.complete("s", "u", max_tokens=100)


def test_sections_and_token_budget(monkeypatch):
    monkeypatch.setattr(llm, "TPM", 8000)  # independent of whatever .env configures
    md = "# T\n\nintro\n\n## A\n\nx\n\n## B\n\ny"
    assert [s.splitlines()[0] for s in pipeline.split_sections(md)] == ["# T", "## A", "## B"]
    assert llm.out_budget("x" * 3000, 1000) == 1000          # plenty of room: request honoured
    assert llm.out_budget("x" * 3000, 9000) == llm.TPM - llm.SAFETY - llm.est_tokens("x" * 3000)  # clipped to fit
    import pytest
    with pytest.raises(llm.TooLarge):
        llm.out_budget("x" * 30_000, 1000)


def test_pipeline_end_to_end_with_fake_llm(monkeypatch, tmp_path: Path):
    def fake_agent(system, user, tools, dispatch, **kw):
        dispatch("search_arxiv", {"query": "q"})
        return "### Theme\n- `[@lovelace2021]` Accuracy was 91%."

    drafts = []

    def fake_complete(system, user, **kw):
        if "cross-disciplinary research analyst" in system:
            return "H1 [@lovelace2021]"
        if "academic writer" in system:
            return "# Title\n\nAccuracy was 91% [@lovelace2021]."
        if "scientific editor" in system:
            drafts.append(1)
            # first rewrite is bad (drops citation), second is good
            return "Accuracy was 91%." if len(drafts) == 1 else "# Title\n\nWe saw 91% accuracy [@lovelace2021]."
        return "No problems found."

    import scientist.tools as t
    monkeypatch.setitem(t.SEARCHERS, "search_arxiv", lambda q, n: parse_arxiv(ARXIV))
    monkeypatch.setattr(llm, "run_tool_agent", fake_agent)
    monkeypatch.setattr(llm, "complete", fake_complete)

    res = pipeline.run("t", tmp_path, kind="paper", log=lambda *_: None)
    assert res.ok and len(drafts) == 2
    assert "We saw 91% accuracy [@lovelace2021]" in res.final
    assert "## References" in res.final and "Ada Lovelace" in res.final
    for f in ["library.json", "1_evidence.md", "2_hypotheses.md", "3_draft.md", "4_humanized.md", "5_verification.md"]:
        assert (tmp_path / f).exists()
