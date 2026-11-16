# Mechanism Lab

**Turn research concepts into mechanisms you can actually test.**

Mechanism Lab explores a concrete question: can we break existing methods into
reusable parts, compose those parts into candidate implementations, and find
combinations that improve a measurable outcome?

The workbench connects source-grounded proposals, typed **atoms and molecules**,
prior-art retrieval, executable experiments, and evidence-aware ranking. A
molecule records how operators connect—not just which concepts occur together.
A promising proposal still has to survive a baseline and a reproducible test.

This is an experimental fork of [The Alien Space of Science](https://github.com/alejandrohdez00/alien-space-of-science).
The upstream authors created the sampling method and operate
[Lacuna](https://lacuna.tiptreesystems.com). Their work is credited below; these
extensions and results are specific to this fork.

## Results, including what failed

Results recorded **2026-11-16**. These are engineering pilots, not established
scientific discoveries.

| Experiment | What happened | Evidence |
|---|---|---|
| **Compose executable compression operators** | Searching 115 compatible chains selected frame-of-reference packing + LZMA. On three shifted synthetic streams it used **24.25% fewer bytes** than the control frozen during validation, with exact round trips. | [Protocol, results, and limitations](operator-studies/integer-composition/README.md) |
| **Recover atom combinations with graphs** | Adamic–Adar recovered **23/100** withheld pairs, versus **21/100** for degree product and **10/100** for text similarity. This measures co-occurrence recovery, not novelty or usefulness. | [Graph baseline](reports/atom-graph/README.md) |
| **Learn which implementation to choose** | A pairwise utility ranker improved validation selection, then **lost to the fixed baseline on all three unseen families**. Held-out regret: **0.09412 vs 0.02238**, lower is better. Not promoted. | [Held-out report](reports/utility-ranker/README.md) |
| **Reconstruct implementations from literature** | Six implementations generated from abstracts, atoms, or molecules produced **no gain against strong controls**. | [Reconstruction studies](reconstruction-studies/README.md) |
| **Ground proposals in code and prior work** | Three audited proposals exposed a normalization failure, overlap with known methods, and an incorrect fixed point. No demonstrated novelty. | [Grounded research reports](reports/improvement-2026-11-16/README.md) |
| **Repair scorer source consistency** | A small coherence model trained on original assignments beat its unigram baseline: test NLL **3.9868 vs 4.3186**. An archived-source repeat reproduced the metrics exactly. Availability alignment remains unresolved. | [Scorer report and provenance limits](reports/source-coherence/README.md) |

The compression result is a useful combination of **known** techniques on a
small synthetic benchmark. It is not a new codec, a comparison against optimized
production codecs, or a result from highest-beta Alien sampling. The learned
utility ranker's failure shows that successful search does not automatically
teach a model to choose well on new tasks.

## Research corpus

The local search corpus now contains **206,917 records**, combining an ML-first
arXiv selection with OpenAlex metadata. All previous records remain traceable.
Citation counts are available for 92,876 records and normalized influence
metrics for 60,237; these are metadata, not usefulness labels.
[Coverage, verification, and reproduction](reports/corpus-207k/README.md).

A first 24-paper packet produced **60 source-linked candidate atoms** for
$0.00835. Independent agent review flagged 12 semantic issues and separately
tagged 9 transfer-oriented results. [Extraction and review](reports/ml-graph-atoms/README.md).
The full corpus has not been atomized or used to retrain the scorers.

## What works today

- **Source-grounded generation:** bounded evidence packets, quoted source spans,
  prior-art retrieval, method-lineage checks, and cost-accounted API calls.
- **Atoms → molecules → executable chains:** typed operator interfaces,
  implementation hashes, bounded enumeration, and exact round-trip checks.
- **Separate ranking criteria:** novelty evidence, feasibility, and
  task-conditioned usefulness; incomparable tasks stay incomparable and
  exploration is explicit. This is a ranking scaffold, not a validated general
  novelty or usefulness predictor.
- **Reproducible studies:** frozen protocols, seeded inputs, strong controls,
  preserved negative results, and linked result artifacts.
- **Local webapp:** document workspace, study ledger, operator library, and
  bounded local checks. Generated reviews remain model judgments, not proof.

## Try the workbench

```sh
./workbench/start.sh --open
```

The document workspace works with Python alone. Engine features have additional
dependencies; see [engine setup](workbench/docs/ENGINE_SETUP.md).
The local integer round-trip checker requires no model key or paid API call.
Paid generation requires a configured provider key and explicit bounded execution.

[App guide](workbench/README.md) · [Where files go](workbench/docs/FOLDERS.md)

## Read or reproduce the experiments

- [Compression composition: protocol, frozen results, rerun commands](operator-studies/integer-composition/README.md)
- [Utility ranking: validation success and held-out failure](reports/utility-ranker/README.md)
- [Typed operator engine and primary sources](docs/operator-composition/ENGINE.md)
- [Dual-ranker contract](docs/ranking/DUAL_RANKER.md)
- [Source-grounded generation](docs/GROUNDED_RESEARCH.md)
- [World-search pilot](reports/world-search-2026-11-16/README.md) and [candidate studies](candidate-studies/README.md)
- [Shared evidence ledger](docs/RESEARCH_LEDGER.md)
- [Fork changes](FORK.md) and [released-artifact provenance audit](PROVENANCE.md)

Next research questions are whether composition gains survive real datasets,
stronger codec baselines, and independent replication; whether utility ranking
can generalize to fresh task families; and whether adding evidence-backed
primitives improves search. None is established by the current results.

## Upstream project and usage

The following describes the original Alien workflow. Released scorer lineage
limitations are documented in [PROVENANCE.md](PROVENANCE.md); a high beta score
is not evidence of novelty, feasibility, or utility.

Code and data for **[The Alien Space of Science: Sampling Coherent but Cognitively Unavailable Research Directions](https://arxiv.org/abs/2603.01092)**.

**This project was developed during my internship at [Tiptree Systems](https://tiptreesystems.com). A production version mapping 2m+ papers into 25k+ research hypotheses is available at [Lacuna](https://lacuna.tiptreesystems.com). You can also use our [Lacuna MCP](https://github.com/tiptreesystems/lacuna-research-mcp)!**

Alien represents papers as idea atoms and ranks combinations using a coherence transformer and an author–atom dual encoder. An LLM can then turn selected combinations into research proposals.

<p align="center"><img src="figures/figure1_page-0001.jpg" alt="Alien pipeline" width="95%"></p>

## 1. Install

Use Python 3.11 or newer. Run commands from this repository's root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Inference works on CPU or CUDA. Retraining the full datasets requires substantial GPU compute; the availability model was trained on four A100 GPUs. LLM keys are only needed for reconstruction and atomization.

## 2. Download the artifacts

The [Alien Space of Science collection](https://huggingface.co/collections/alejandrohdez/alien-space-of-science-6ad8b17d0399975c463c6e1e) is the release landing page.

Download the camera-ready bundles from the public [model repository](https://huggingface.co/alejandrohdez/alien-models) and [data repository](https://huggingface.co/datasets/alejandrohdez/alien-data).

```bash
MODEL_REPO="alejandrohdez/alien-models"
DATA_REPO="alejandrohdez/alien-data"
hf download "$MODEL_REPO" --include "*_model/*" --local-dir models
hf download "$DATA_REPO" papers.tar.gz data.tar.gz --repo-type dataset --local-dir downloads
tar -xzf downloads/papers.tar.gz
tar -xzf downloads/data.tar.gz
```

This provides `papers/clusters_80.json`, all 16,068 papers' methodological summaries and atom files, `data/llm-papers.db`, the coherence training data, and both model directories. See [DATA.md](DATA.md) for contents and optional downloads.

Fixed atom combinations and proposal texts from Alien, Random, Claude, and Gemini are available separately:

```bash
hf download "$DATA_REPO" research-assets.tar.gz --repo-type dataset --local-dir downloads
tar -xzf downloads/research-assets.tar.gz
```

## 3. Sample atom combinations

```bash
python alien.py sample --top-k 20 --output-dir results/demo
```

The default draws 1,000 candidate sequences, keeps distinct triplets, scores them with both models, and selects the best combinations at β = 0.7. It produces `results/demo/k3/beta_0.7/results.json`. Sampling makes no LLM calls. The selected count is capped by the number of distinct candidates.

Change the tradeoff or candidate count:

```bash
python alien.py sample --n-gen 10000 --betas 0.4 0.7 --top-k 300 --output-dir results/sweep
```

Higher β favors less available combinations. The score is `(1 − β) × z(coherence) + β × z(unavailability)`, where unavailability is negative availability. Normalization uses the entire candidate pool before selecting results. JSON includes atom IDs, component scores, and rankings.

The paper's exhaustive candidate pool is more expensive; use a GPU:

```bash
python alien.py sample --ks 3 --exhaustive-ks 3 --betas 0.7 --top-k 300 \
  --device cuda --output-dir results/exhaustive
```

Different candidate pools and numerical backends can change rankings.

## 4. Reconstruct proposals

Reconstruction uses LiteLLM. Set the key for your chosen provider, for example `GEMINI_API_KEY` or `ANTHROPIC_API_KEY`, then run:

```bash
python alien.py reconstruct --method inference \
  --results results/demo/k3/beta_0.7/results.json --top-k 3 \
  --reconstruction-model gemini/gemini-3.1-pro-preview
```

This makes paid API calls and writes Markdown under `reconstructions/inference/`. Rerunning resumes completed proposals. Decoder, prompt, and atom-catalogue changes use separate output/cache directories. Use `--cache-dir reconstructions/cache` to reuse proposals across selections.

## 5. Retrain and use your checkpoints

Coherence data is included in `data.tar.gz`. These options match the released coherence model's training settings:

```bash
python alien.py train-coherence \
  --train data/coherence_dataset/coherence_train.jsonl \
  --val data/coherence_dataset/coherence_val.jsonl \
  --token-mapping data/coherence_dataset/token_mapping.json \
  --output-dir models/coherence_model_retrained \
  --epochs 30 --batch-size 64 --gradient-accumulation-steps 2 \
  --learning-rate 0.0001 --warmup-steps 1000 --max-seq-len 32
```

Download the compact availability dataset (about 1.8 GB compressed, 4.3 GB extracted):

```bash
hf download "$DATA_REPO" availability-compact.tar.gz --repo-type dataset --local-dir downloads
tar -xzf downloads/availability-compact.tar.gz
python alien.py train-availability --dataset data/availability_dataset \
  --output-dir models/availability_model_retrained --batch-size 4096 --epochs 100
```

Training exports the model with its matching author bank. Reduce `--batch-size` for a smaller GPU. For four GPUs, use `torchrun --standalone --nproc_per_node=4 -m generation.train_availability --compile` with the same arguments. `--resume` continues an interrupted run from its last completed epoch.

```bash
python alien.py sample \
  --coherence-model models/coherence_model_retrained \
  --availability-model models/availability_model_retrained \
  --top-k 20 --output-dir results/retrained
```

## Atomization and rebuilding data

The original atomization, clustering, and dataset-building tools are included. See [ADVANCED.md](ADVANCED.md) for their commands and API-key requirements. Prompts and source code are intended to be adapted to your corpus.

Run `python alien.py --help` or `python alien.py <command> --help` for options.

## Fork: local search improvements

See [FORK.md](FORK.md) for opt-in batched sampling, explicit Apple MPS support, and saving/reusing complete candidate pools. Original serial generation remains the default.
