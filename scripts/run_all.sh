#!/usr/bin/env bash
# Reproduces the headline results end to end. Should finish well under 15
# minutes on a laptop with --llm mock (no API key / no internet needed).
# For the real, judged numbers reported in report.md, export
# ANTHROPIC_API_KEY and pass --llm claude (slower, costs API credits, needs
# internet).
set -euo pipefail
cd "$(dirname "$0")/.."

LLM_MODE="${1:-mock}"   # mock | claude | gemini
SAMPLE_SIZE="${2:-100000}"  # rows of raw twcs data to subsample before pairing

echo "== 1/4 Data prep (subsampling raw data to ${SAMPLE_SIZE} rows) =="
python3 src/data_prep.py --raw data/raw/AmazonHelp_sample.csv --sample "${SAMPLE_SIZE}" \
  --out data/processed/pairs.csv

echo "== 2/4 Building golden evaluation set (200 examples) =="
python3 eval/build_golden_set.py --pairs data/processed/pairs.csv --n 200 --out eval/golden_set.csv

echo "== 3/4 Running pipeline demo on 5 ad-hoc examples (llm=${LLM_MODE}) =="
python3 src/pipeline.py --pairs data/processed/pairs.csv --llm "${LLM_MODE}" --n_demo 5

echo "== 4/4 Running full evaluation harness (system + 2 baselines, llm=${LLM_MODE}) =="
python3 src/evaluate.py --pairs data/processed/pairs.csv --golden eval/golden_set.csv \
  --llm "${LLM_MODE}" --out_results outputs/eval_results.csv --out_metrics outputs/metrics.json

echo ""
echo "Done. See outputs/metrics.json for the summary and outputs/eval_results.csv for per-example detail."
