# AmazonHelp AI Support Agent — Hiver SDE Intern Take-Home

An AI support agent for **@AmazonHelp** (Twitter), built from the Kaggle
"Customer Support on Twitter" dataset. Given an incoming customer message,
it:

1. **Classifies intent** into a 7-way taxonomy derived from the data
   (`src/intents.py`).
2. **Drafts a reply grounded in how AmazonHelp has historically resolved
   similar issues**, via TF-IDF retrieval over past (customer → agent reply)
   pairs + LLM generation (`src/retrieval.py`, `src/reply_generator.py`).
3. **Decides auto-handle vs. escalate to a human, with a stated reason**,
   using hard safety/legal rules plus an LLM judgment call
   (`src/escalation.py`).

**Start with [`report.md`](report.md)** — especially Section 5, "What is
misleading about my headline number?" — before quoting any metric from this
repo. **[`decision_log.md`](decision_log.md)** has the non-obvious design
calls and why.

## Repo layout

```
data/raw/AmazonHelp_sample.csv   # provided AmazonHelp-filtered slice of the dataset
data/processed/pairs.csv         # generated: cleaned (customer, agent_reply) pairs
src/
  data_prep.py       # load raw CSV -> clean English (customer, agent_reply) pairs
  intents.py         # 7-intent taxonomy + rule-based classifier + LLM classifier
  retrieval.py        # TF-IDF index over historical resolved cases
  reply_generator.py  # drafts a reply grounded in retrieved historical cases
  escalation.py       # auto-handle vs escalate, with a stated reason
  llm_client.py        # Anthropic API wrapper + offline MockClient fallback
  pipeline.py          # ties the above together end-to-end
  evaluate.py          # full eval harness: system vs 2 baselines vs golden set
baselines/
  trivial_baseline.py  # majority-class intent, canned reply, never escalate
  simple_baseline.py   # keyword-rule intent, 1-NN reply copy, keyword escalation
eval/
  build_golden_set.py     # stratified sampling + first-pass heuristic labels
  labeling_guidelines.md  # rubric for a human to review/correct the golden set
  golden_set.csv           # generated: 200 labeled examples
  judge.py                 # LLM-as-judge rubric for reply quality
  judge_agreement.py       # judge-vs-human agreement statistics
scripts/run_all.sh   # one-command reproduction
outputs/             # generated: eval_results.csv, metrics.json, human ratings
report.md            # the required report (problem framing, results, failure analysis, etc.)
decision_log.md       # the required decision log
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ recommended. No GPU, no model downloads — retrieval is TF-IDF,
not embeddings (see `decision_log.md` #7).

## Reproduce the headline results (< 15 minutes, no API key needed)

```bash
bash scripts/run_all.sh mock 100000
```

This will (in order): subsample 100,000 raw rows and build clean
(customer, reply) pairs → build a 200-example stratified golden set → run a
5-example pipeline demo printed to stdout → run the full evaluation harness
(system + both baselines) and write `outputs/metrics.json` +
`outputs/eval_results.csv`. On a laptop this finishes in well under a
minute; 15 minutes is a very generous ceiling.

**Note:** in this default `mock` mode, every "LLM" call is answered by a
deterministic offline stand-in (`src/llm_client.py::MockClient`) — this
exists purely so the pipeline is reproducible without an API key. **It is
not evidence of reply quality.** See report.md Section 4 for a real,
measured judge-vs-human agreement check showing the mock judge has ~zero
agreement with a human rater, and Section 5 for why the mock-mode
intent/escalation numbers shouldn't be over-read either.

## Run with a real LLM (Claude)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pip install anthropic
bash scripts/run_all.sh claude 100000
```

This calls the real Anthropic API for intent classification, reply
generation, escalation decisions, and judging. It costs API credits and
needs internet; budget more time than the mock run since each of the 200
golden examples makes 4 API calls (intent, reply, escalation, judge) —
budget roughly 10-20 minutes depending on rate limits. This is the mode
whose output should actually be trusted for quality claims.

## Try one message ad-hoc

```bash
python3 src/pipeline.py --llm mock --text "My package says delivered but I never got it, this is the second time this month"
```

## Rebuild the golden set with different parameters

```bash
python3 eval/build_golden_set.py --pairs data/processed/pairs.csv --n 200 --out eval/golden_set.csv
```

**Before trusting the golden set as ground truth**, a human should review
`eval/golden_set.csv`'s `gold_intent_reviewed` / `gold_escalate_reviewed`
columns per `eval/labeling_guidelines.md` and flip `reviewed_by_human` to
`True` per row. The `_firstpass` columns are auto-generated starting points,
not verified labels — see `decision_log.md` #12.

## Check judge-vs-human agreement (real numbers, already run once)

```bash
# Step 1 (already done for this submission): sample N rows to rate by hand
python3 eval/judge_agreement.py --results outputs/eval_results.csv --n 30
# Step 2: open outputs/human_ratings_template.csv, fill in human_overall_score (1-5)
# Step 3: score it
python3 eval/judge_agreement.py --template outputs/human_ratings_template.csv --score
```

The already-completed run for this submission is in
`outputs/human_ratings_template.csv` (filled in) and
`outputs/judge_agreement_mock.txt` (results) — see report.md Section 4.

## What's cited / borrowed

- Dataset: Kaggle "Customer Support on Twitter"
  (`thoughtvector/customer-support-on-twitter`), AmazonHelp-filtered slice
  as provided in the assignment.
- Libraries: pandas, numpy, scikit-learn (`TfidfVectorizer`,
  `cosine_similarity`, metrics), scipy (`spearmanr`), langdetect, anthropic
  Python SDK. No borrowed code beyond standard library usage of these
  packages per their public docs.
- No fine-tuned or third-party models beyond the Anthropic API (optional,
  for `--llm claude` mode).
