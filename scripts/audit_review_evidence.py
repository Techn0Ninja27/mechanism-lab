"""Redacted diagnostics for source-review evidence.

This tool is deliberately non-semantic: it reports evidence-shape and wording
signals for human inspection.  It never changes a verdict and cannot create a
campaign gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

HEADER = "Selected Section IV only; figures omitted; equations from HTML alttext."
QUALIFIERS = ("fixed", "always", "every", "exactly", "guaranteed")
NUMBER_RE = re.compile(r"(?<![A-Za-z])(?:\d+(?:\.\d+)?%?|\d+(?:\.\d+)?[×x])")


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _items(review: Mapping[str, Any]) -> Iterable[tuple[str, str, Mapping[str, Any]]]:
    for record in review.get("records", []):
        rid = str(record.get("record_id", record.get("id", "")))
        fields = record.get("field_items", record.get("fields", []))
        if isinstance(fields, list):
            for item in fields:
                if isinstance(item, Mapping):
                    yield rid, str(item.get("field", "")), item


def _tokens(text: str) -> set[str]:
    return {x.lower() for x in re.findall(r"[A-Za-z]+", text)}


def diagnose(review: Mapping[str, Any]) -> dict[str, Any]:
    reasons = Counter()
    lengths: list[int] = []
    flags: list[dict[str, Any]] = []
    header_count = 0
    supported = 0
    for rid, field, item in _items(review):
        reason = str(item.get("reason", ""))
        if reason:
            reasons[reason] += 1
        quote = item.get("support_quote")
        if not isinstance(quote, str):
            continue
        supported += item.get("verdict") == "supported"
        qlen = len(quote)
        lengths.append(qlen)
        has_header = HEADER in quote
        header_count += int(has_header)
        value = str(item.get("value", ""))
        value_words = _tokens(value)
        quote_words = _tokens(quote)
        qualifiers = [q for q in QUALIFIERS if re.search(rf"\b{re.escape(q)}\b", value, re.I)
                      and not re.search(rf"\b{re.escape(q)}\b", quote, re.I)]
        numbers = [n for n in NUMBER_RE.findall(value) if n not in NUMBER_RE.findall(quote)]
        # This is intentionally a warning signal, not an entailment decision:
        # paraphrases and symbols can defeat lexical overlap.
        no_content_overlap = bool(value_words) and not (value_words & quote_words)
        if qualifiers or numbers or no_content_overlap or has_header:
            flags.append({
                "record_id": rid,
                "field": field,
                "verdict": item.get("verdict"),
                "quote_sha256": _sha_text(quote),
                "quote_chars": qlen,
                "quote_sentences": max(1, len(re.findall(r"[.!?](?:\s|$)", quote))),
                "header_present": has_header,
                "qualifier_words_absent_from_quote": qualifiers,
                "numeric_tokens_absent_from_quote": numbers,
                "lexical_overlap_warning": no_content_overlap,
                "diagnostic": "diagnostic_not_entailment",
            })
    lengths_sorted = sorted(lengths)
    histogram = {"0-300": 0, "301-800": 0, "801-1500": 0, "1501+": 0}
    for length in lengths:
        if length <= 300:
            histogram["0-300"] += 1
        elif length <= 800:
            histogram["301-800"] += 1
        elif length <= 1500:
            histogram["801-1500"] += 1
        else:
            histogram["1501+"] += 1
    return {
        "schema": "review-evidence-diagnostics.v1",
        "reviewer_id": review.get("reviewer_id"),
        "record_count": len({rid for rid, _, _ in _items(review)}),
        "field_item_count": sum(1 for _ in _items(review)),
        "supported_quote_count": supported,
        "quote_chars": {"count": len(lengths), "min": min(lengths) if lengths else None,
                         "median": lengths_sorted[len(lengths_sorted) // 2] if lengths else None,
                         "max": max(lengths) if lengths else None,
                         "histogram": histogram},
        "injected_header_quote_count": header_count,
        "reason_frequencies": [{"reason": reason, "count": count} for reason, count in reasons.most_common()],
        "flagged_fields": flags,
        "scope": "Shape and lexical diagnostics only; no semantic verdict, gate, or faithfulness claim.",
    }


def audit_paths(paths: list[Path], output: Path | None = None) -> dict[str, Any]:
    reviews = []
    for path in paths:
        review = json.loads(path.read_text(encoding="utf-8"))
        result = diagnose(review)
        result["input_path"] = str(path)
        result["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        reviews.append(result)
    result = {"schema": "review-instrument-diagnostics.v1", "diagnostic_not_entailment": True,
              "reviews": reviews}
    if output is not None:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reviews", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_paths(args.reviews, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
