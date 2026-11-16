# Real SmolVLM attention capture and independent replay

Recorded 2026-11-16. SmolVLM-256M ran on the Mac GPU through PyTorch MPS,
using two already-exposed image prompts. This is a development trace and
extraction check, not a task-quality, latency, usefulness or novelty result.

The corrected run has 64 genuine visual feature tokens per image, 30 text
layers, nine query heads and three key/value heads. The model configuration's
image token ID identifies visual positions; adjacent image delimiters are
excluded. We save complete prefill attention, prompt-to-visual attention,
per-step decode attention, RoPE-transformed queries and keys, values, and
aligned input embeddings. Four decoder calls follow prefill. Each call consumes
one generated token, so these query traces start after the prompt's final query.

Independent CPU NumPy reconstruction from the captured Q/K tensors agrees with
the model's eager attention to at most 2.23e-6 absolute error in prefill and
5.69e-7 in decode. All captured probability rows normalize within 3.58e-7;
causal future-token mass is zero. The verifier also checks exact visual slices,
trace hashes, image hashes, and the frozen plan/dependency hashes. These are
extraction-consistency checks, not an intervention test.

## Provenance and limitations

Model: [HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct),
revision `d039ab0d4b9311b617e8ffec1d06c8464364aeb3`. Every local model file
matches the pinned Hugging Face API's Git blob or LFS hash. The safetensors
weights occupy 513,028,808 bytes. No remote model Python code is trusted.

The initial agent run successfully exercised MPS but had invalid extraction:
a broad token-ID range included two nonvisual markers and decode means averaged
all keys, reducing to approximately 1/key_count. Its code and invalidity record
are preserved; v2 traces must not be used for the ablation. `capture.py` produces
the corrected v3 run. Its plan was written before inference. The prompts/images
had already been exposed in v2 and are explicitly development inputs.

The original two images came from the model-card examples: Hugging Face's candy
image and Britannica's Statue of Liberty image. The first runner overwrote raw
downloads with processed JPEGs, so v3 identifies its inputs by their exact
processed-byte hashes, not original source hashes. `prepare.py` now preserves
both and refuses changed inputs. These are two still images, not long videos,
held-out clips or an independent model transfer set. Optional image splitting
is disabled. Hooks duplicate projections to capture Q/K/V; runtime measurements
would be misleading. This does not reproduce WindowQuant or FIER.

## Reproduce

Install the exact `dependencies.lock` into an isolated Python 3.12 environment.
The existing local environment is `../.venv-vlm`. Then, from the engine root:

```sh
sh operator-studies/smolvlm-trace/run.sh results/smolvlm-reproduction-fresh
```

The runner fetches only the pinned model files (under a 1 GB metadata size cap),
recreates and checks the frozen processed images, captures on MPS, and runs
independent CPU NumPy attention replay. It refuses existing outputs. Set
`TRACE_PYTHON` to another compatible environment if needed. On a machine without
MPS, use `capture.py --device cpu` directly; it does not silently fall back.

The committed `manifest.json`, `plan.json`, `dependencies.lock` and `replay.json`
describe the corrected local run at `results/smolvlm-trace-v3/`. The compressed
traces and images remain local; the trace directory is about 28 MB. The next
step is a separately frozen equal-budget ranking/quantization ablation with
strong constituent controls, followed by unseen visual inputs and model/task
validation if the development result warrants it.

The full `run.sh` command was executed successfully into
`results/smolvlm-reproduce-v1`. Both compressed trace files are byte-identical
to the corrected run; see `reproduction.json`. Fresh original image receipts
are now retained in `image-receipts.json`. This is same-machine reproduction,
not independent external replication.
