# Corpus expansion plan for mechanism and molecule discovery

Date: 2026-11-16

The current pilot was under-informed: it generated from two toy problem statements and reviewed against two lexical searches per proposal. The next corpus should expose the generator to mechanisms, failed approaches, benchmarks, and executable artifacts while preserving publication dates, source hashes, and paper-family splits. A paper is evidence for a mechanism or an evaluation protocol; it is not evidence that a generated idea is new.

## Priority primary sources

1. **FrontierScience** (OpenAI, 2025), [paper](https://arxiv.org/abs/2512.03343), [lab release](https://openai.com/index/frontierscience/). Expert-written physics, chemistry and biology tasks; the Research track is a useful evaluation target. Limitation: constrained expert questions do not measure open-ended discovery.
2. **Scientific computing in the age of agentic AI** (OpenAI, 2026), [report](https://openai.com/index/scientific-computing-agentic-ai/). Eight case studies emphasize external acceptance tests and human validation. Limitation: field report, not a controlled causal comparison.
3. **Distributed biological reasoning** (OpenAI, 2026), [report](https://openai.com/index/scaling-ai-research-for-complex-biological-data/). Thousands of agents analyzed a whole-brain perturbation atlas and linked effects to literature. Mechanism to extract: partitioned investigation plus a provenance-linked reasoning atlas. Limitation: novelty and value assessments still required blinded expert review.
4. **LifeSciBench** (OpenAI, 2026), [release](https://openai.com/index/introducing-life-sci-bench/). Expert life-science research tasks organized around evidence, analysis, design, validation, translation, and communication. Limitation: domain-specific and benchmark-style.
5. **ScienceAgentBench** (Chen et al., 2024), [arXiv:2410.05080](https://arxiv.org/abs/2410.05080). 102 executable tasks extracted from 44 papers and expert-validated; best agents solved only a minority independently. Mechanism: executable artifact evaluation and contamination controls. Limitation: task completion is not discovery.
6. **Auto-Bench** (Chen et al., 2025), [arXiv:2502.15224](https://arxiv.org/abs/2502.15224). Interactive causal-graph discovery with oracle interventions. Mechanism: hypothesis–intervention–update loops. Limitation: synthetic hidden structures and complexity-sensitive degradation.
7. **SCI-AGENT-EXAM** (Zhong et al., 2026), [arXiv:2610.32441](https://arxiv.org/abs/2610.32441), [code](https://github.com/SKYLENAGE-AI/Sci-Agent-Exam). Expert-curated workflow tasks with deterministic executable checks plus calibrated text judging. Limitation: the benchmark is recent and its reported model scores are not a measure of external research novelty.
8. **Are Self-Evolving Agents Continual Learners?** (Chen et al., 2026), [arXiv:2610.25858](https://arxiv.org/abs/2610.25858). Finds a stability–plasticity tradeoff in prompt, memory, and skill rewriting; retrieval can collide on similar tasks. Mechanism: evaluate retention, adaptation, and transfer separately.
9. **Memory-Centric Continual Learning for AI Agents** (Li et al., 2026), [arXiv:2610.29357](https://arxiv.org/abs/2610.29357). Defines substrate/asset/policy axes and evaluates behavioral improvement rather than retrieval alone. Limitation: proposed framework; requires independent task streams.
10. **When Continual Learning Moves to Memory** (Hu et al., 2026), [arXiv:2604.27003](https://arxiv.org/abs/2604.27003). Shows that memory representation and organization relocate, rather than remove, forgetting; abstract procedural memories can transfer better than detailed trajectories. Limitation: ALFWorld/BabyAI do not cover research domains.
11. **Modular Memory is the Key to Continual Learning Agents** (Dorovatas et al., 2026), [arXiv:2603.01761](https://arxiv.org/abs/2603.01761). Position paper combining in-context and in-weight learning. Limitation: roadmap rather than a decisive experimental result.
12. **DreamCoder** (Ellis et al., 2020/2021), [arXiv:2006.08381](https://arxiv.org/abs/2006.08381). Wake–sleep library learning promotes recurring program structure into reusable abstractions. Limitation: executable DSLs make compression and transfer measurable; prose research ideas need an equivalent representation.
13. **Stitch / Top-Down Synthesis for Library Learning** (Bowers et al., 2022), [arXiv:2211.16605](https://arxiv.org/abs/2211.16605). Corpus-guided compression of program fragments. Use as the molecule-promotion analogue. Limitation: syntactic compression can produce broad or accidental macros.
14. **RASP / Thinking Like Transformers** (Weiss, Goldberg, Yahav, 2021), [arXiv:2106.06981](https://arxiv.org/abs/2106.06981). A symbolic abstraction language for transformer computation. Mechanism: move from neural implementation to typed information-flow primitives. Limitation: abstraction is hand-designed and task-family-specific.
15. **Novelty Search** (Lehman and Stanley, 2011), [DOI](https://doi.org/10.1162/ARTL_a_00057). Preserves behavioral novelty instead of optimizing one scalar objective. Limitation: novelty requires a meaningful behavior descriptor.
16. **MAP-Elites** (Mouret and Clune, 2015), [arXiv:1504.04909](https://arxiv.org/abs/1504.04909). Quality-diversity archive retaining high-quality solutions across behavior niches. Limitation: the archive can be diverse in a bad descriptor while missing scientifically important distinctions.
17. **Toward Artificial Open-Ended Evolution within Lenia** (2024), [arXiv:2406.04235](https://arxiv.org/abs/2406.04235). Shows how quality-diversity can maintain evolving behavior niches. Limitation: artificial-life behavior does not establish research usefulness.
18. **The Ontology of the Alien** (Westerberg, 2026), [repository](https://github.com/emergent-wisdom/ontology-of-the-alien). World-diversity generation plus a persistent typed solution taxonomy. The release itself states that world coverage, diversity-preserving selection, and real-world validation remain open.
19. **Meaning Model alien search** (Westerberg, 2026), [recorded search](https://github.com/emergent-wisdom/meaning-model/tree/main/examples/alien-search-loneliness). Executable invented-world simulations, component removal, and transfer. Limitation: one loneliness search; no idea passed all four internal verification axes.
20. **Sema** (Westerberg, 2026), [repository](https://github.com/emergent-wisdom/sema). Content-addressed identity for agent concepts. Use for exact definition/version identity; do not use hashes as semantic novelty or correctness.

## Chinese and multilingual coverage

Include Chinese-language and China-based sources rather than relying only on English abstracts. Start with **Auto-Bench** (authors include Chinese and international institutions), **SCI-AGENT-EXAM** (Chinese-led author list), **DeepSeek LLM** [arXiv:2401.02954](https://arxiv.org/abs/2401.02954), and Chinese evaluation corpora such as **C-Eval** [arXiv:2305.08322](https://arxiv.org/abs/2305.08322) and **CMMLU** [arXiv:2306.09212](https://arxiv.org/abs/2306.09212). These sources provide multilingual and culturally situated task distributions, but benchmark performance must not be mistaken for discovery evidence. Search Chinese titles/abstracts and English translations, then deduplicate by DOI, arXiv ID, and paper hash.

## Retrieval query bank

Run each query against arXiv/OpenAlex/Semantic Scholar and the authors' repositories. Keep query text, date, API response, paper ID, abstract, PDF hash when downloaded, and whether the record is preprint, peer-reviewed, benchmark, code, or position paper.

1. `scientific discovery agent hypothesis intervention oracle causal graph`
2. `LLM research idea generation novelty prior art benchmark`
3. `AI scientist executable hypothesis testing benchmark`
4. `research agent literature synthesis experiment design reproducible evaluation`
5. `world model counterfactual mechanism discovery language model`
6. `invented worlds divergent ideation mechanism transfer evaluation`
7. `quality diversity novelty search scientific design behavior descriptor`
8. `MAP-Elites scientific discovery hypothesis generation`
9. `open-ended evolution abstraction library learning compression`
10. `DreamCoder Stitch reusable abstraction held-out generalization`
11. `program synthesis learned library recurring substructure compression`
12. `typed causal graph mechanism composition intervention prediction`
13. `concept discovery abstraction induction neural symbolic`
14. `compositional generalization reusable mechanisms transfer tasks`
15. `continual learning agent memory retention interference forward transfer`
16. `self-evolving agent prompt memory skill stability plasticity`
17. `memory organization retrieval collision negative transfer agent`
18. `model editing weight update locality generalization sequential edits`
19. `in-context learning in-weight learning modular memory`
20. `AI-generated research ideas external novelty human evaluation`
21. `scientific novelty evaluator retrieval grounded citation verification`
22. `LLM novelty judgment instability prior art retrieval`
23. `Chinese language scientific agent benchmark research workflow`
24. `中国 科研 智能体 科学发现 基准 大语言模型`
25. `中文 科研想法 新颖性 评估 大模型`
26. `AI for science protein design autonomous laboratory benchmark`
27. `AI materials discovery closed loop experiment validation`
28. `agentic coding Kaggle competition novel algorithm ablation held-out`

## Dataset construction

Store records at three levels: paper mechanism, executable artifact, and evaluation result. A mechanism record should include the intervention, causal interface, assumptions, predicted observable, nearest prior art, failure mode, and exact source spans. A candidate molecule should retain its atom definitions, typed edges, world/solver/compiler trace, and rejected translations.

Use family- and time-held-out splits. Do not train a generator or reward model on an automated novelty label before human adjudication. Keep negative examples: known techniques presented with unusual language, plausible-looking collages, retrieval misses, failed transfers, and proposals whose benchmark gain disappears under component removal.

The first expansion should be metadata and abstracts, not “all of arXiv.” Download full text only for records selected by mechanism queries or cited by a candidate. That gives broader coverage without burying the generator in redundant prose or creating an unbounded token bill.

## Next experiment

Use the expanded corpus to generate 12 molecules on two research problems, with three arms: direct prior-art-grounded generation, random seed framing, and target-blind world detour. Retrieve 8–12 records per candidate using exact mechanism queries plus citation chasing. Extract typed molecules before judging novelty. Test the best two molecules on a second task with a pre-registered component-removal ablation. Promote nothing unless recurrence, held-out transfer, ablation, prior-art review, and artifact hashes all pass.
