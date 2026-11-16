#!/bin/sh
set -eu
if [ "$#" -ne 1 ]; then echo 'Usage: run.sh FRESH_OUTPUT_DIRECTORY' >&2; exit 2; fi
study_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
trace_python=${TRACE_PYTHON:-"$study_dir/../../../.venv-vlm/bin/python"}
trace_output=$1
if [ -e "$trace_output" ]; then echo 'Output exists; preserving it.' >&2; exit 1; fi
mkdir -p "$trace_output"
"$trace_python" "$study_dir/prepare.py" --output "$trace_output/assets"
"$trace_python" "$study_dir/capture.py" --model "$trace_output/assets/model" \
  --model-metadata "$trace_output/assets/model-api.json" --images-from "$trace_output/assets/images" \
  --output "$trace_output/capture" --device mps
"$trace_python" "$study_dir/replay.py" --run "$trace_output/capture" --output "$trace_output/replay.json"
