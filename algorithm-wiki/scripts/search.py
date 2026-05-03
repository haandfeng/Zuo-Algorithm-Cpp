#!/usr/bin/env python3
"""
search.py — naive keyword search over the algorithm wiki.

Designed for two use cases:
  1. CLI:    python search.py "单调栈"
  2. From an LLM agent: same CLI, but it'll parse the structured output.

Returns a ranked list of {file, score, snippet} hits. Scoring is intentionally
naive — title hit > heading hit > body hit, with TF weighting.

For more sophisticated semantic search, embed each article and do cosine
similarity. This implementation deliberately stays in stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
WIKI = ROOT / "wiki"
RAW = ROOT / "raw"

CJK = re.compile(r"[一-鿿]+")
WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")


def tokenize(text: str) -> list[str]:
    """Tokenize CJK character-by-character + ascii word-by-word."""
    tokens: list[str] = []
    for m in CJK.finditer(text):
        tokens.extend(list(m.group()))
    for m in WORD.finditer(text):
        tokens.append(m.group().lower())
    return tokens


def first_heading(text: str) -> str:
    for line in text.splitlines():
        m = re.match(r"^#\s+(.+)$", line.strip())
        if m:
            return m.group(1).strip()
    return ""


def gather_files(include_raw: bool) -> list[Path]:
    files = list(WIKI.rglob("*.md"))
    if include_raw:
        files.extend(RAW.rglob("*.md"))
    return [f for f in files if f.is_file()]


def score_file(text: str, query_tokens: list[str], weights: dict[str, float]) -> float:
    """Score a file based on title, headings, body."""
    score = 0.0
    title = first_heading(text)
    title_tokens = tokenize(title)
    body_tokens = tokenize(text)

    title_set = set(title_tokens)
    body_set = set(body_tokens)
    body_freq: dict[str, int] = {}
    for t in body_tokens:
        body_freq[t] = body_freq.get(t, 0) + 1

    headings = [m.group(1) for m in re.finditer(r"^#{2,6}\s+(.+)$", text, re.MULTILINE)]
    heading_tokens = set()
    for h in headings:
        heading_tokens.update(tokenize(h))

    for q in query_tokens:
        if q in title_set:
            score += weights["title"]
        if q in heading_tokens:
            score += weights["heading"]
        if q in body_set:
            score += weights["body"] * (1.0 + 0.1 * (body_freq.get(q, 0) - 1))
    return score


def make_snippet(text: str, query_tokens: list[str], width: int = 80) -> str:
    """Find the densest window around a query hit and return a single-line snippet."""
    lower = text.lower()
    best_idx = -1
    for q in query_tokens:
        idx = lower.find(q.lower())
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx = idx
    if best_idx == -1:
        return text[:width].replace("\n", " ")
    start = max(0, best_idx - width // 2)
    end = min(len(text), start + width)
    snippet = text[start:end].replace("\n", " ").strip()
    return ("…" if start > 0 else "") + snippet + ("…" if end < len(text) else "")


def search(query: str, top_k: int, include_raw: bool) -> list[dict[str, Any]]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    weights = {"title": 10.0, "heading": 3.0, "body": 1.0}
    results: list[dict[str, Any]] = []
    for f in gather_files(include_raw):
        text = f.read_text(encoding="utf-8", errors="replace")
        s = score_file(text, query_tokens, weights)
        if s > 0:
            results.append({
                "file": f.relative_to(ROOT).as_posix(),
                "title": first_heading(text),
                "score": round(s, 2),
                "snippet": make_snippet(text, query_tokens),
            })
    results.sort(key=lambda r: -r["score"])
    return results[:top_k]


def main() -> int:
    ap = argparse.ArgumentParser(description="Search the algorithm wiki.")
    ap.add_argument("query", help="Keyword(s) to search.")
    ap.add_argument("-k", "--top-k", type=int, default=10, help="How many results to show.")
    ap.add_argument("--include-raw", action="store_true", help="Also search raw/ files.")
    ap.add_argument("--json", action="store_true", help="Emit JSON for LLM consumption.")
    args = ap.parse_args()

    hits = search(args.query, args.top_k, args.include_raw)

    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return 0

    if not hits:
        print(f"(no results for '{args.query}')")
        return 0

    print(f"=== {len(hits)} hit(s) for '{args.query}' ===\n")
    for i, h in enumerate(hits, 1):
        print(f"{i:>2}. [{h['score']:>5}] {h['title']}")
        print(f"     {h['file']}")
        print(f"     {h['snippet']}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
