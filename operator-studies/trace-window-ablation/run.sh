#!/bin/sh
set -eu
if [ "$#" -ne 1 ]; then echo 'Usage: run.sh FRESH_OUTPUT_DIRECTORY' >&2; exit 2; fi
study_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
engine_root=$(CDPATH= cd -- "$study_dir/../.." && pwd)
ablation_python=${ABLATION_PYTHON:-"$engine_root/../.venv/bin/python"}
"$ablation_python" "$study_dir/experiment.py" --traces "$engine_root/results/smolvlm-trace-v3" --output "$1" --freeze
"$ablation_python" "$study_dir/experiment.py" --traces "$engine_root/results/smolvlm-trace-v3" --output "$1"
