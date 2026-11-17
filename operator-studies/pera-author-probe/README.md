# PERA published-layer probe

2026-11-17. Question: does the published polynomial layer initialize the state
that its own forward pass consumes? This is an isolated code-path check, not
an installation of the authors’ complete training environment.

Inspected [hira/tuners/lora.py](https://github.com/zhangwenhao6/PERA/blob/87751531305bb4097e4ebed52fc0043cc1506b47/hira/tuners/lora.py)
at commit `87751531305bb4097e4ebed52fc0043cc1506b47`. The source declares a
coefficient ParameterDict, but creation of its adapter entry and initialization
are commented out (lines 425, 460, 481); the encoder indexes it at line 546.
The source snapshot stays local under `results/pera-author-code-v1` and is
hash-pinned by [protocol.json](protocol.json). A [byte/AST check](snapshot-verification.json)
verifies the upstream Git blob: local file creation added one trailing newline,
with identical Python AST. No upstream code is redistributed.

The runner evaluates only the inspected `LoraLayer` and `Linear` class ASTs,
supplying PyTorch, math and the usual transpose helper. It does not execute
package imports, model downloads or trainer code. The fixture uses a 12-to-16
linear layer, rank 2, CPU float32, seed 17, A zero/B Kaiming. The initializer
supports Kaiming/zero; this is deliberately not described as the paper’s
Gaussian initialization or a faithful training reproduction.

Observed [result](result.json):

- The unmodified bias-free layer raises an AttributeError for its missing
  coefficient entry.
- Explicitly injecting a zero coefficient in a diagnostic arm allows the
  bias-free forward pass and gives exactly the frozen base output at A=0.
- With that same injection and a non-null bias, the first reduced-dimension
  linear operation raises a shape error (expanded width 5 versus bias width 16).
  This fixture does not imply failure in the bias-free LLaMA layers used in
  typical reported experiments.

The code also leaves merge/unmerge unimplemented. That is a code observation,
not a timing result: the mathematical update can be merged into a dense weight,
but this published class does not implement that path. The missing coefficient
observation applies to this pinned isolated construction; an uninspected
external checkpoint or wrapper could supply state. We did not run a full
training pipeline and make no claim about the validity of the paper’s results.

Run with the retained source snapshot and the pinned environment in
`requirements.txt` (Python 3.12.14):

```sh
../.venv/bin/python operator-studies/pera-author-probe/run.py \
  --output /tmp/pera-author-probe-fresh.json
```

A changed source hash or existing output is refused. The recorded protocol
predates the probe. `all_expected_observations` means these code-path observations
were reproduced, not that PERA is ready for deployment or has scientific novelty.
