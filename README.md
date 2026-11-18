# Scientific Collaborator

An autonomous pipeline that takes a research question and produces a cited, humanized manuscript, built from four skill layers:

| Stage | Skill | What it does |
|---|---|---|
| 1. Scout | Scientific Agent Skills (`scientist/skills/agent_skills.md`) | Tool-using Claude agent searching arXiv, PubMed and Semantic Scholar; optional Python runner |
| 2. Discover | Science Superpowers (`superpowers.md`) | Cross-field patterns and falsifiable hypotheses, each rated for evidence strength |
| 3. Write | Academic Paper Skills (`academic_paper.md`) | Structured paper / review / proposal with `[@key]` citations |
| 4. Humanize | Humanized Academic Writing (`humanize.md`) | Clearer prose, with a code-enforced guard: if citations or numbers change, the rewrite is rejected, retried once, then discarded in favour of the unedited draft |
| 5. Verify | (built in) | Unknown citation keys are caught in code; a reviewer pass flags claims the abstracts don't support. The References list is generated from the library, never by the model |

## Setup (free)

Any OpenAI-compatible endpoint works. The current `.env` uses Groq's free tier (`openai/gpt-oss-120b`, with the smaller `gpt-oss-20b` as a fallback when the daily quota runs out). In testing it found more papers and wrote better-supported text than Gemini's free tier, which also allows only 20 requests per day per model. Gemini and local Ollama work too (Ollama needs no key); list several models in `LLM_MODEL` separated by commas. Set `LLM_TPM_LIMIT` / `LLM_RPM_LIMIT` to your provider's free-tier limits (Groq: 8000 tokens per minute; Gemini: a requests-per-minute cap).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your own key (`.env` is git-ignored, so it is never uploaded):

```
LLM_API_KEY=your-free-groq-key        # https://console.groq.com/keys
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
```

For a fully local setup instead: install Ollama, run `ollama pull qwen3:8b`, and set
`LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=qwen3:8b`, and no key.

Groq's free tier allows 8,000 tokens per minute, so every call is sized to fit and the pipeline pauses when it hits the limit. A full run takes roughly 8-12 minutes. Raise `LLM_TPM_LIMIT` if your plan allows more.

## Run as a website

```bash
python server.py
```

Open http://127.0.0.1:8000. Enter a question, pick the output type, and watch the five stages progress. Results appear in tabs (manuscript, hypotheses, evidence, verification, sources); citations are hover-able chips showing the paper title, and past runs are listed in the sidebar. The server binds to localhost only and does not expose the code-running option.

### PDF with your name on it

On a finished run, click **Download PDF**, enter your name (affiliation, email and ORCID are optional), and the paper is generated as a formatted A4 PDF: title, author block, date, author-year citations, a generated reference list, and page footers carrying your name. The details travel in the request body and are not stored on the server (the browser remembers them locally for next time). A short note that the draft was AI-assisted and checked against abstracts only is added by default; untick the box to leave it out.

Runs are processed one at a time because they share the model's per-minute token budget; extra runs wait in the queue (`MAX_CONCURRENT_RUNS` changes this).

## Use from the command line

```bash
python main.py "Do gut microbiome signatures predict immunotherapy response?" --kind review
```

`--kind` is `review`, `paper` or `proposal`. Output goes to `runs/<topic>/`:
`library.json`, `1_evidence.md`, `2_hypotheses.md`, `3_draft.md`, `4_humanized.md`, `5_verification.md`, and `final.md` (with references).

Options: `--turns N` caps scout tool-use turns; `--allow-code` lets the scout run model-written Python on your machine (off by default).

Environment: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_TPM_LIMIT`, `LLM_REASONING_EFFORT`, `NCBI_API_KEY` and `S2_API_KEY` (optional; Semantic Scholar rate-limits unauthenticated use heavily, and the scout falls back to the other databases when it does).

## Limits you should know about

- The system only reads **abstracts**, not full texts. The verification report says so; treat the output as a strong first draft, not a finished paper.
- Hypotheses are leads to test, not findings. Check novelty yourself.
- The verifier's claim audit is model-assisted. The citation-key and number checks are the deterministic part.

## Tests

```bash
python -m pytest
```

Tests run offline (parsers, citation guard, and the full pipeline with a stubbed model).
