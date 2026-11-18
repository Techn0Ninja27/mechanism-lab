# Fixed-artifact evaluation handoff

Run from `experiments/alien-science/upstream` with `../.venv/bin/python`.
These are local evaluator steps after `python -m scripts.fixed_artifact_campaign export`; they do not alter frozen inputs, the paid ledger, or API transport.

## Exported inventories

Export writes `results/fixed-artifact-comparison-v1/evaluation/<decoder>/inventory.json`.
Use these files directly. They are schema `causal-candidate-inventory.v1` with 124 rows in frozen schedule order. `inventory_sha256` is the SHA-256 of file bytes, while `candidate_inventory_sha256` is the separate `canonical_hash(sorted(candidates, key=id))` field used inside selection validation. Do not rewrite or reorder the exported file.

```sh
DECODER=opus-5.5 ../.venv/bin/python -c 'import hashlib,os; from pathlib import Path; p=Path("results/fixed-artifact-comparison-v1/evaluation")/os.environ["DECODER"]/"inventory.json"; print(hashlib.sha256(p.read_bytes()).hexdigest())'
DECODER=fable-5.2 ../.venv/bin/python -c 'import hashlib,os; from pathlib import Path; p=Path("results/fixed-artifact-comparison-v1/evaluation")/os.environ["DECODER"]/"inventory.json"; print(hashlib.sha256(p.read_bytes()).hexdigest())'
```

The printed SHA is the required `source_inventory_sha256` and must equal every dev/final `inventory_sha256`. The candidate-list hash remains computed by the selector from sorted IDs; it is not a replacement for the file SHA.

## Development and frozen selection

```sh
../.venv/bin/python -m scripts.run_causal_campaign --inventory results/fixed-artifact-comparison-v1/evaluation/opus-5.5/inventory.json --output results/fixed-artifact-comparison-v1/evaluation/opus-5.5/dev --stage dev --execute
../.venv/bin/python -m scripts.run_causal_campaign --inventory results/fixed-artifact-comparison-v1/evaluation/fable-5.2/inventory.json --output results/fixed-artifact-comparison-v1/evaluation/fable-5.2/dev --stage dev --execute
```

Development uses seeds 100–115. The CLI has no selection-writing command; use the actual selector, binding the exported *file* SHA:

```sh
DECODER=opus-5.5 ../.venv/bin/python -c '
import json,hashlib,os; from pathlib import Path; from generation.campaign_selection import select_development
d=os.environ["DECODER"]; r=Path("results/fixed-artifact-comparison-v1/evaluation")/d; ip=r/"inventory.json"; inv=json.loads(ip.read_text()); dev=json.loads((r/"dev/result.json").read_text()); h=hashlib.sha256(ip.read_bytes()).hexdigest(); assert dev["stage"]=="dev" and dev["inventory_sha256"]==h; s=select_development(inv["candidates"],dev["runs"],expected_seeds=tuple(range(100,116)),inventory_sha256=h); assert s["status"]=="complete",s; p=r/"selection.json"; assert not p.exists(), "selection already exists"; p.open("x").write(json.dumps(s,ensure_ascii=False,indent=2)+"\n")'
DECODER=fable-5.2 ../.venv/bin/python -c 'import json,hashlib,os; from pathlib import Path; from generation.campaign_selection import select_development; d=os.environ["DECODER"]; r=Path("results/fixed-artifact-comparison-v1/evaluation")/d; ip=r/"inventory.json"; inv=json.loads(ip.read_text()); dev=json.loads((r/"dev/result.json").read_text()); h=hashlib.sha256(ip.read_bytes()).hexdigest(); assert dev["stage"]=="dev" and dev["inventory_sha256"]==h; s=select_development(inv["candidates"],dev["runs"],expected_seeds=tuple(range(100,116)),inventory_sha256=h); assert s["status"]=="complete",s; p=r/"selection.json"; assert not p.exists(), "selection already exists"; p.open("x").write(json.dumps(s,ensure_ascii=False,indent=2)+"\n")'
```

Do not edit `selection.json`; final validation re-derives it from the dev aggregate.

## Final execution and report

The required stronger comparator is `reports/generation-comparison-v2/control-selection.json` (`linear-control-03`, `final_results_seen: false`).

```sh
../.venv/bin/python -m scripts.run_causal_campaign --inventory results/fixed-artifact-comparison-v1/evaluation/opus-5.5/inventory.json --output results/fixed-artifact-comparison-v1/evaluation/opus-5.5/final --stage final --selection results/fixed-artifact-comparison-v1/evaluation/opus-5.5/selection.json --dev-result results/fixed-artifact-comparison-v1/evaluation/opus-5.5/dev/result.json --control-selection reports/generation-comparison-v2/control-selection.json --execute
../.venv/bin/python -m scripts.run_causal_campaign --inventory results/fixed-artifact-comparison-v1/evaluation/fable-5.2/inventory.json --output results/fixed-artifact-comparison-v1/evaluation/fable-5.2/final --stage final --selection results/fixed-artifact-comparison-v1/evaluation/fable-5.2/selection.json --dev-result results/fixed-artifact-comparison-v1/evaluation/fable-5.2/dev/result.json --control-selection reports/generation-comparison-v2/control-selection.json --execute
```

For each decoder, filter `final/result.json`'s `runs` exactly into `final/hold-runs.json` (`id == "__hold__"`) and `final/control-runs.json` (`id == "linear-control-03"`), preserving list order. Then run:

```sh
../.venv/bin/python -m scripts.run_causal_campaign --inventory results/fixed-artifact-comparison-v1/evaluation/opus-5.5/inventory.json --selection results/fixed-artifact-comparison-v1/evaluation/opus-5.5/selection.json --dev-result results/fixed-artifact-comparison-v1/evaluation/opus-5.5/dev/result.json --runs results/fixed-artifact-comparison-v1/evaluation/opus-5.5/final/result.json --hold-runs results/fixed-artifact-comparison-v1/evaluation/opus-5.5/final/hold-runs.json --control-selection reports/generation-comparison-v2/control-selection.json --control-runs results/fixed-artifact-comparison-v1/evaluation/opus-5.5/final/control-runs.json --output results/fixed-artifact-comparison-v1/evaluation/opus-5.5/report --report
../.venv/bin/python -m scripts.run_causal_campaign --inventory results/fixed-artifact-comparison-v1/evaluation/fable-5.2/inventory.json --selection results/fixed-artifact-comparison-v1/evaluation/fable-5.2/selection.json --dev-result results/fixed-artifact-comparison-v1/evaluation/fable-5.2/dev/result.json --runs results/fixed-artifact-comparison-v1/evaluation/fable-5.2/final/result.json --hold-runs results/fixed-artifact-comparison-v1/evaluation/fable-5.2/final/hold-runs.json --control-selection reports/generation-comparison-v2/control-selection.json --control-runs results/fixed-artifact-comparison-v1/evaluation/fable-5.2/final/control-runs.json --output results/fixed-artifact-comparison-v1/evaluation/fable-5.2/report --report
```

## Summary

Call `summarize_fixed_artifact(reports, selections=..., inventories=..., fidelity=..., expected_decoders=("opus-5.5","fable-5.2"))`. `reports` maps each decoder to its `report/report.json`; selections and inventories map to the frozen `selection.json` and exported inventory. Fidelity must be `{ "atoms": json.load(atoms-fidelity.json), "distill": json.load(prepared/prose-fidelity.json) }`; standalone fidelity objects are not accepted for positive prepared-arm rows. Write the returned object to `reports/fixed-artifact-comparison-v1/final-summary.json`.

## Blockers/invariants

* No CLI writes development selections; the explicit `select_development` call is required.
* File SHA and sorted-candidate hash are distinct contracts; preserve the exported file and bind both through the existing selector/validator.
* Production `--report` requires the frozen stronger control manifest.
* Final report rejects missing/foreign/duplicate hold or control records.
