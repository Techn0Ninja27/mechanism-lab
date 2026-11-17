"""Extract inspected line ranges from downloaded arXiv HTML snapshots."""
from html.parser import HTMLParser
from pathlib import Path
import hashlib, json

ROOT = Path(__file__).parent
SPECS = [
    {
        "id": "2601.22996v1", "title": "Competitive Non-Clairvoyant KV-Cache Scheduling for LLM Inference",
        "url": "https://arxiv.org/html/2601.22996v1",
        "ranges": [(245, 248), (419, 454), (491, 493), (824, 827)],
        "purpose": "problem assumptions, kill-and-restart model, GSA, GBA, conclusion",
    },
    {
        "id": "2607.09248v1", "title": "General Non-Clairvoyant KV-Cache Scheduling via Regime-Aware Routing",
        "url": "https://arxiv.org/html/2607.09248v1",
        "ranges": [(423, 424), (665, 705), (768, 776), (904, 905)],
        "purpose": "clairvoyant lower-bound obstruction, rectangle-strip, response slicing, routing, conclusion",
    },
    {
        "id": "2508.14544v2", "title": "Adaptively Robust LLM Inference Optimization under Prediction Uncertainty",
        "url": "https://arxiv.org/html/2508.14544v2",
        "ranges": [(85, 89), (142, 170)],
        "purpose": "memory/cancellation model and A_min mechanism; heterogeneous extension and experiments omitted",
    },
]

class Visible(HTMLParser):
    def __init__(self):
        super().__init__(); self.skip = False; self.parts = []
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "svg"}: self.skip = True
        if not self.skip and tag in {"h1", "h2", "h3", "h4", "h5", "p", "div", "section", "li", "pre"}: self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag in {"script", "style", "svg"}: self.skip = False
        if not self.skip and tag in {"h1", "h2", "h3", "h4", "h5", "p", "div", "section", "li", "pre"}: self.parts.append("\n")
    def handle_data(self, data):
        if not self.skip: self.parts.append(data)

def normalized(path):
    parser = Visible(); parser.feed(path.read_text(encoding="utf-8"))
    return [" ".join(line.split()) for line in "".join(parser.parts).splitlines() if line.strip()]

def main():
    records = []
    for spec in SPECS:
        source = ROOT / "sources" / f"{spec['id']}.html"
        lines = normalized(source)
        if not lines: raise ValueError(f"empty source: {source}")
        selected = []
        for start, end in spec["ranges"]:
            if start < 1 or end > len(lines) or start > end: raise ValueError(f"range outside source: {spec['id']} {start}-{end}")
            selected.append(f"[source lines {start}-{end}]\n" + "\n".join(lines[start-1:end]))
        text = "\n\n".join(selected) + "\n\n[Omitted: all source lines outside the inspected ranges above; this excerpt is not a complete paper.]\n"
        if len(text) > 8000: raise ValueError(f"selected excerpt exceeds 8000 chars: {spec['id']} ({len(text)})")
        out = ROOT / "sources" / f"{spec['id']}.selected.txt"
        out.write_text(text, encoding="utf-8")
        records.append({"id": spec["id"], "title": spec["title"], "url": spec["url"], "path": f"sources/{spec['id']}.selected.txt", "evidence_level": "full_text", "source_html": f"sources/{spec['id']}.html", "source_html_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "selected_sha256": hashlib.sha256(text.encode()).hexdigest(), "selected_chars": len(text), "line_ranges": spec["ranges"], "purpose": spec["purpose"], "omitted": "All normalized HTML lines outside listed ranges; no completeness claim."})
    (ROOT / "manifest.json").write_text(json.dumps({"schema": "adjacent-primary-evidence.v1", "sources": records}, indent=2) + "\n")
    (ROOT / "selection-receipt.json").write_text(json.dumps({"schema": "adjacent-selection-receipt.v1", "ranges_are_normalized_visible_html_lines": True, "source_count": len(records), "total_downloaded_bytes": sum((ROOT / r["source_html"]).stat().st_size for r in records), "records": records}, indent=2) + "\n")
    print(json.dumps({"sources": len(records), "chars": {r["id"]: r["selected_chars"] for r in records}}))

if __name__ == "__main__": main()
