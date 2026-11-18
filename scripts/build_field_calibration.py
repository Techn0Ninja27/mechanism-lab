"""Build an offline field-review calibration packet from the closed v5 draw."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "results/generation-comparison-v5-plan"
OUT = ROOT / "reports/chunk-field-calibration-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    pack_path = V5 / "sourcepack.json"
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    atoms = json.loads((V5 / "representations/atoms.json").read_text(encoding="utf-8"))["atoms"]
    distillations = json.loads((V5 / "representations/distillations.json").read_text(encoding="utf-8"))["distillations"]
    records = [{"representation": "atom", "record": copy.deepcopy(row)} for row in atoms]
    records += [{"representation": "distillation", "record": copy.deepcopy(row)} for row in distillations]

    planted = []
    row = copy.deepcopy(next(x for x in atoms if x["id"] == "sni-1"))
    row["id"] = "cal-contradiction-atom-gpu"
    row["conditions"] = ["each node has no GPU"]
    records.append({"representation": "atom", "record": row}); planted.append(row["id"])

    row = copy.deepcopy(next(x for x in atoms if x["id"] == "fresh-2"))
    row["id"] = "cal-unsupported-atom-speed"
    row["outputs"] = ["exactly 999x speedup on every dataset"]
    records.append({"representation": "atom", "record": row}); planted.append(row["id"])

    row = copy.deepcopy(next(x for x in distillations if x["id"] == "sni-1-overview"))
    row["id"] = "cal-contradiction-distill-dpu"
    row["summary"] += " No DPU is used by the system."
    records.append({"representation": "distillation", "record": row}); planted.append(row["id"])

    row = copy.deepcopy(next(x for x in distillations if x["id"] == "fresh-2-results"))
    row["id"] = "cal-unsupported-distill-speed"
    row["summary"] += " It achieves exactly 999x speedup on every dataset."
    records.append({"representation": "distillation", "record": row}); planted.append(row["id"])

    source_hashes = {
        "sourcepack.json": sha(pack_path),
        "atoms.json": sha(V5 / "representations/atoms.json"),
        "distillations.json": sha(V5 / "representations/distillations.json"),
    }
    inputs = {
        "schema": "chunk-field-calibration-inputs.v1",
        "purpose": "Offline reviewer calibration only; never use these records for generation.",
        "source_pack_path": str(pack_path),
        "source_pack_digest": pack["digest"],
        "source_hashes": source_hashes,
        "records": records,
        "review_criterion": "cited_chunk_support.v1",
        "field_rule": "Review every substantive field against the complete cited chunk text; exact evidence quotes are pointers.",
    }
    answer = {
        "schema": "chunk-field-calibration-answer-key.v1",
        "source_hashes": source_hashes,
        "planted_bad_records": {
            planted[0]: "contradicted",
            planted[1]: "unsupported",
            planted[2]: "contradicted",
            planted[3]: "unsupported",
        },
        "required_positive_checks": {
            "fresh-2": "supported",
            "fresh-2-results": "supported",
        },
        "required_pass_rule": "A calibration review must flag all four planted records and must not flag either named positive check. Verdicts on the other original records are descriptive only.",
        "limitations": [
            "The planted negatives test the rubric, not reviewer population error rates.",
            "The original v5 records are closed and cannot be used for generation.",
        ],
    }
    write_once(OUT / "inputs.json", inputs)
    write_once(OUT / "answer-key.json", answer)
    print(json.dumps({"records": len(records), "source_hashes": source_hashes, "planted": planted}, indent=2))


if __name__ == "__main__":
    main()
