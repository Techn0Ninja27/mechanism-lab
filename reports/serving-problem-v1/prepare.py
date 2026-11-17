"""Freeze a bounded offline source pack for disaggregated-serving research."""
from __future__ import annotations

import hashlib
import json
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from generation.sourcepack import build_from_file, validate_source_pack


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []; self.skip = None

    def handle_starttag(self, tag, attrs):
        if self.skip:
            return
        if tag in {"script", "style", "svg"}:
            self.skip = tag
        elif tag in {"p", "div", "h1", "h2", "h3", "h4", "h5", "section", "li", "tr", "pre"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.skip:
            if tag == self.skip:
                self.skip = None
        elif tag in {"p", "div", "h1", "h2", "h3", "h4", "h5", "section", "li", "tr", "pre"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


SOURCES = [
    ("2610.15579v1", "DMRS dual-memory request scheduling", "https://arxiv.org/html/2610.15579v1",
     [(65, 81), (130, 291), (292, 350), (638, 758), (1731, 1763)]),
    ("2505.11916v2", "Arrow adaptive P/D scheduling", "https://arxiv.org/html/2505.11916v2",
     [(139, 286)]),

]


def normalize(html: str) -> str:
    parser = VisibleText(); parser.feed(html)
    return "\n".join(" ".join(line.split()) for line in "".join(parser.parts).splitlines() if line.strip()) + "\n"


def select_sections(text: str, ranges, per_source=None):
    lines = text.splitlines()
    # Exact full line ranges, inspected against normalized snapshot. Never
    # silently discard a later limitations section to meet a character cap.
    return "\n\n".join("\n".join(lines[start-1:end]) for start,end in ranges)+"\n"


def main():
    for sid, _, _, _ in SOURCES:
        src = HERE / "sources" / f"{sid}.html"
        full = normalize(src.read_text(encoding="utf-8"))
        (HERE / "sources" / f"{sid}-full.txt").write_text(full, encoding="utf-8")
        out = HERE / "sources" / f"{sid}.txt"
        ranges = next(item[3] for item in SOURCES if item[0] == sid)
        out.write_text(select_sections(full, ranges), encoding="utf-8")
    manifest = {"sources": [
        {"id": sid, "title": title, "url": url, "path": f"sources/{sid}.txt", "evidence_level": "full_text"}
        for sid, title, url, _ in SOURCES
    ]}
    manifest_path = HERE / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    pack = build_from_file(manifest_path, HERE, max_total_chars=60_000, chunk_chars=4_000, max_source_bytes=2_000_000)
    validate_source_pack(pack)
    if pack["limits"]["truncated_source_count"]:
        raise ValueError("Review selections rather than silently truncating sources")
    (HERE / "pack.json").write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n")
    receipts = [{"id": sid, "html_bytes": (HERE / "sources" / f"{sid}.html").stat().st_size,
                 "html_sha256": hashlib.sha256((HERE / "sources" / f"{sid}.html").read_bytes()).hexdigest(),
                 "full_text_bytes": (HERE / "sources" / f"{sid}-full.txt").stat().st_size,
                 "selected_text_bytes": (HERE / "sources" / f"{sid}.txt").stat().st_size,
                 "selected_text_sha256": hashlib.sha256((HERE / "sources" / f"{sid}.txt").read_bytes()).hexdigest()}
                for sid, _, _, _ in SOURCES]
    (HERE / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(json.dumps({"included_chars": pack["limits"]["included_chars"], "chunks": len(pack["chunks"]), "digest": pack["digest"]}))


if __name__ == "__main__":
    main()
