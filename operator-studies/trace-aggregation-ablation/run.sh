#!/bin/sh
set -eu
if [ "$#" -ne 1 ]; then echo 'Usage: run.sh FRESH_OUTPUT_DIRECTORY' >&2; exit 2; fi
study_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
aggregation_python=${AGGREGATION_PYTHON:-"$study_dir/../../../.venv/bin/python"}
"$aggregation_python" "$study_dir/experiment.py" --output "$1" --freeze
"$aggregation_python" "$study_dir/experiment.py" --output "$1"
