#!/usr/bin/env python3
"""Reproduce and verify the frozen contract-evaluation packet offline.

This script reads only the retained local HTML snapshots and metadata.  It
does not fetch URLs, call a model, or alter the packet unless --output is
explicitly supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from html.parser import HTMLParser
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "results" / "contract-eval-packet-v1"
sys.path.insert(0, str(REPO))
from generation.sourcepack import build_from_file, validate_source_pack


class VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip: str | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if self.skip:
            return
        if tag in {"script", "style", "svg"}:
            self.skip = tag
        elif tag in {"h1", "h2", "h3", "h4", "h5", "p", "div", "section", "li", "pre"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.skip:
            if tag == self.skip:
                self.skip = None
        elif tag in {"h1", "h2", "h3", "h4", "h5", "p", "div", "section", "li", "pre"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)


def normalize_html(path: Path) -> str:
    parser = VisibleText()
    parser.feed(path.read_text(encoding="utf-8"))
    return "\n".join(" ".join(line.split()) for line in "".join(parser.parts).splitlines() if line.strip()) + "\n"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_text(full: str, ranges: list[list[int]]) -> str:
    lines = full.splitlines()
    pieces = []
    for start, end in ranges:
        if start < 1 or end < start or end > len(lines):
            raise ValueError(f"line range outside normalized source: {start}-{end}")
        pieces.append(f"[normalized visible source lines {start}-{end}]\n" + "\n".join(lines[start - 1:end]))
    return "\n\n".join(pieces) + "\n\n[OMITTED: all normalized source lines outside the listed inspected ranges; this is not a complete paper.]\n"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--output", type=Path, help="write a freshly rebuilt pack here")
    args = ap.parse_args()
    root = args.packet_root.resolve()
    manifest = load(root / "manifest.json")
    receipts = {r["id"]: r for r in load(root / "receipts.json")["records"]}
    scopes = load(root / "scopes.json")
    if set(scopes) != {s["id"] for s in manifest["sources"]}:
        raise ValueError("operation scopes must be a direct map for every manifest source")
    for source in manifest["sources"]:
        sid = source["id"]
        receipt = receipts[sid]
        html = root / source["source_html"]
        selected = root / source["path"]
        if sha(html) != source["source_html_sha256"] or sha(html) != receipt["source_html_sha256"]:
            raise ValueError(f"source HTML hash mismatch: {sid}")
        rebuilt = selected_text(normalize_html(html), source["line_ranges"])
        if rebuilt.encode("utf-8") != selected.read_bytes():
            raise ValueError(f"normalized line-range excerpt mismatch: {sid}")
        if sha(selected) != source["selected_sha256"] or sha(selected) != receipt["selected_sha256"]:
            raise ValueError(f"selected excerpt hash mismatch: {sid}")
        if len(rebuilt) != source["selected_chars"]:
            raise ValueError(f"selected excerpt character count mismatch: {sid}")
    pack = build_from_file(root / "manifest.json", root, max_total_chars=36_000, chunk_chars=4_000, max_source_bytes=1_000_000)
    validate_source_pack(pack)
    if pack["limits"]["truncated_source_count"]:
        raise ValueError("source pack would truncate a selected excerpt")
    expected = load(root / "pack.json")
    if pack["digest"] != expected["digest"]:
        raise ValueError(f"pack digest mismatch: rebuilt {pack['digest']} != recorded {expected['digest']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ready", "sources": len(pack["sources"]), "chunks": len(pack["chunks"]), "included_chars": pack["limits"]["included_chars"], "digest": pack["digest"], "output": str(args.output) if args.output else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
