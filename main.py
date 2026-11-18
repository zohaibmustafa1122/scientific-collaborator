"""CLI: python main.py "your research question" --kind review"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scientist import llm, pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="Autonomous scientific collaborator")
    ap.add_argument("topic", help="research question or topic")
    ap.add_argument("--kind", choices=list(pipeline.KINDS), default="review")
    ap.add_argument("--out", type=Path, help="output directory (default: runs/<topic-slug>)")
    ap.add_argument("--turns", type=int, default=20, help="max tool-use turns for the literature scout")
    ap.add_argument("--allow-code", action="store_true",
                    help="let the scout run model-written Python on this machine (off by default)")
    args = ap.parse_args()

    out = args.out or Path("runs") / pipeline.slugify(args.topic)
    try:
        result = pipeline.run(args.topic, out, kind=args.kind, max_turns=args.turns, allow_code=args.allow_code)
    except llm.Refused as e:
        print(f"Refused: {e}", file=sys.stderr)
        return 2
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
