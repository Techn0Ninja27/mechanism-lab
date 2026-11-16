# Bounded local source packs

`generation/sourcepack.py` and `scripts/build_sourcepack.py` turn an explicit JSON manifest into a deterministic, offline evidence pack. The packer accepts UTF-8 Markdown, text, Python, and JSON files below a supplied root. It never executes code, follows a symlink outside the root, fetches a URL, or treats a source as novel because it was imported.

Each source retains its ID, title, URL, declared evidence level, relative path, byte SHA-256, character count, and truncation counts. The `chunks` array contains exact numbered line spans with the source content hash. `digest` hashes the canonical pack body, so a prompt consumer can bind its request to one evidence snapshot. The top-level `limits` record makes omitted material explicit.

Manifest example:

```json
{"sources":[{"id":"paper-1","title":"A paper","url":"https://doi.org/example","path":"papers/paper.md","evidence_level":"full_text","license":"CC-BY-4.0"}]}
```

Run:

```sh
python scripts/build_sourcepack.py --manifest manifest.json --root ./sources \
  --output results/sourcepack.json --max-total-chars 100000 --chunk-chars 4000 \
  --max-source-bytes 5000000
```

`full_text` and `official_code` are declarations supplied by the manifest owner; they are not independently verified by this utility. `abstract` and `user_document` remain weaker evidence levels. Imported documents are untrusted prompt data and must not be interpreted as instructions. The packer is a provenance boundary, not a semantic entailment or novelty checker.

Each input is bounded to 5 MB by default (`--max-source-bytes`); empty files are rejected. The output character limit does not replace the input-size limit.

Generation callers should run `validate_source_pack(pack)` immediately before constructing a paid prompt. It checks schema, digest, internal source/chunk hashes, line spans, limits, and counts. This is snapshot integrity only: without the original files it cannot prove source authenticity, completeness, semantic entailment, licensing, or novelty.
