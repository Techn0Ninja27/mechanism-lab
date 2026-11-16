#!/usr/bin/env python3
"""Build a bounded offline source pack from a JSON manifest."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation.sourcepack import build_from_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-total-chars", type=int, default=100_000)
    parser.add_argument("--chunk-chars", type=int, default=4_000)
    parser.add_argument("--max-source-bytes", type=int, default=5_000_000)
    args = parser.parse_args()
    pack = build_from_file(args.manifest, args.root, max_total_chars=args.max_total_chars,
                           chunk_chars=args.chunk_chars, max_source_bytes=args.max_source_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "digest": pack["digest"],
                      "sources": len(pack["sources"]), "chunks": len(pack["chunks"])}, indent=2))


if __name__ == "__main__":
    main()
