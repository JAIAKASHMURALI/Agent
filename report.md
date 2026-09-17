# Report — AmazonHelp AI Support Agent

Brand: **AmazonHelp** (from the Kaggle "Customer Support on Twitter" dataset,
brand-filtered slice provided as `data/raw/AmazonHelp_sample.csv`, ~305k rows).

All numbers below are from `outputs/metrics.json`, produced by
`bash scripts/run_all.sh mock 100000` (raw-row subsample = 100,000, golden
set n = 200). **Read "What's misleading about my headline number" (Section 5)
before quoting any number from this report out of context** — the `mock`
numbers are a reproducibility smoke test, not evidence of reply quality.

## 1. Problem framing

**What "good" means for this brand.** AmazonHelp handles an extremely high
volume of low-stakes, repetitive requests (delivery status, "how do I
return this") mixed with a smaller number of high-stakes ones (refunds,
account access, angry/repeat contacts). For a brand at this scale, "good"
is **not** "the bot resolves everything" — it's:

1. **Never do the risky thing badly.** A wrong auto-reply on a refund or
   account issue is far more expensive (chargebacks, trust, PR) than an
   unnecessary escalation. So the escalation policy is deliberately biased
   toward precision on "don't escalate" (i.e., high recall on true
   escalations, tolerating some over-escalation) — see `src/escalation.py`.
2. **Resolve the boring 60-70% cheaply and consistently**, freeing human
   agents for the cases that need judgment.
3. **Never hallucinate specifics** (order numbers, refund amounts, exact
   dates) that could mislead a customer or create a false commitment.

**What I chose not to build:**
- No multi-turn dialogue management. The pipeline scores/replies to a single
  incoming message; a production system would need conversation state
  (has this customer already been asked to DM once? did they reply?).
- No fine-tuned classifier. With ~2-4k clean English pairs from a 100k-row
  subsample, a fine-tuned intent model would overfit; a 7-way rule set +
  LLM classifier is more defensible at this data volume (see Decision Log #2).
- No non-English support. ~90%+ of the raw AmazonHelp traffic in this sample
  is filtered out at the language-detection step (see Section 5) — this is
  an English-only prototype.
- No real Twitter posting/integration — this is an offline batch pipeline
  you point at a CSV.

## 2. Results vs. baselines

Three systems evaluated on the same 200-example golden set:

| | Intent accuracy | Intent macro-F1 | Escalation precision | Escalation recall | Escalation F1 |
|---|---|---|---|---|---|
| **Trivial baseline** (majority intent, canned reply, never escalate) | 0.455 | 0.089 | 0.000 | 0.000 | 0.000 |
| **Simple baseline** (keyword rules + 1-NN reply copy + keyword escalation) | 0.845 | 0.811 | 0.500 | 0.045 | 0.083 |
| **System** (LLM intent + TF-IDF retrieval + LLM-drafted reply + hybrid escalation) | 0.845 | 0.811 | 0.333 | 0.114 | 0.170 |

Notes on reading this table (see Section 5 for the full misleading-number
discussion):
- **Intent accuracy is identical between Simple and System in `--llm mock`
  mode** because the mock LLM's intent call *is* the same rule-based
  classifier under the hood (see `src/llm_client.py::MockClient`). This
  row will diverge once run with `--llm claude`; that comparison is the one
  that actually matters and is not included here (see Section 6, "what I'd
  do next").
- **Escalation recall is bad for everyone** (0-11%) against a 22% gold
  escalation rate. The hybrid system beats the keyword-only baseline on
  recall (0.114 vs 0.045) because of the hard-rule + "money-intent-without-
  grounding" rules in `src/escalation.py`, but both are far too
  conservative — most gold-escalations in this set are "sentiment is bad
  but doesn't hit a keyword," which only a real LLM judgment call (not mock)
  is likely to catch. This is failure mode #1 below.
- Reply-quality numbers (`pct_grounded`, `mean_overall_score`, etc.) are
  **omitted from this table on purpose** — the `mock` judge scored the
  system's own extractive-template replies against itself and is
  meaningless (see Section 5).

## 3. Failure analysis — top 5 failure modes

Drawn from reading `outputs/eval_results.csv` and the golden set by hand.

1. **Escalation recall is far too low (11% vs a 22% gold rate).**
   Example: *"Thanks for showing your concern but my issue is still not
   resolved nor I have received the product I had purchase. Had already
   shared the details in links for 10 times"* — this is a textbook
   repeat-contact escalation, but it doesn't contain any of the hard-rule
   trigger words (no "lawyer," no "third time" exactly), so it fell through
   to the LLM judgment step. **Hypothesis:** the hard-rule list is too
   narrow and too literal; a real LLM call (not mock) plus a broader
   sentiment/repetition signal (e.g., "already told you," "again," "still")
   would catch more of these — but this needs to be validated against
   `--llm claude`, not assumed.

2. **TF-IDF retrieval fails on paraphrases with no lexical overlap.**
   Example: customer says *"I've just had an alleged failed delivery
   attempt even though I've been at home all day"* — no historical case
   used the phrase "alleged failed delivery attempt," so the nearest
   neighbor pulled in an unrelated case about a different topic, and the
   generated reply hallucinated a courier-specific detail. **Hypothesis:**
   semantic (embedding-based) retrieval would generalize better than
   lexical TF-IDF for this; TF-IDF was chosen for offline reproducibility
   (Decision Log #7), and this is the cost of that choice.

3. **Ambiguous / context-free tweets get confidently mis-classified.**
   Example: *"So that's it!!"* and *"I've just had this"* — these are
   almost certainly replies deep in a thread where the actual complaint is
   in an earlier tweet not visible to our single-tweet pipeline, but the
   system classifies them as `general_or_feedback` and drafts a generic
   reply with reasonable-sounding but ungrounded confidence.
   **Hypothesis:** the pipeline needs thread context (the parent tweet(s)),
   not just the leaf message — see "what I'd do next."

4. **Occasional hallucinated personalization.** In a few drafted replies
   the (mock) generator invented a customer name that was never given
   (e.g., addressing someone as "Peter" or "Jennifer" when no name was in
   the message or retrieved case). This is exactly the failure mode
   `judge.py`'s `correct` criterion is designed to catch, and it's a strong
   argument for never shipping ungrounded replies without that check.

5. **Non-English and mixed-language traffic is silently dropped, not
   handled.** ~85-90% of raw rows are excluded by the `langdetect` filter in
   `data_prep.py`. In a real deployment this isn't a "failure" so much as an
   **undocumented scope gap** if someone assumes the reported metrics apply
   to all AmazonHelp traffic — which is exactly why it's called out again in
   Section 5.

## 4. LLM-as-judge vs. human agreement

`eval/judge_agreement.py` samples N examples from `outputs/eval_results.csv`
and compares the judge's `overall_score` (1-5) to a human's independent
1-5 rating of the same (customer message, draft reply) pairs.

**Run performed for this report:** 30 examples from the `mock`-mode eval run
were hand-rated (`outputs/human_ratings_template.csv`, `human_overall_score`
column). Result (`outputs/judge_agreement_mock.txt`):

- Mean human score: **2.47 / 5** — most mock-generated replies were
  templated, off-topic, or partially mismatched to the actual message.
- Mock judge score: **constant at 5/5 for all 30** — the mock judge's
  crude word-overlap heuristic is trivially satisfied because it compares
  the draft reply against a prompt that contains that same draft reply.
- **Spearman rho: undefined (judge output has zero variance). Exact match:
  0%. Cohen's kappa (good/bad @ threshold 4): 0.0.**

**This is a real, important finding, not a bug being swept under the rug:
the mock judge has zero measured agreement with a human and must not be
used to make any quality claim.** It exists solely so the pipeline is
runnable end-to-end without an API key. **Before trusting any reply-quality
number, rerun `scripts/run_all.sh claude ...` with a real `ANTHROPIC_API_KEY`
and redo this agreement check on the `claude`-mode outputs** — that
number, not the mock one, is the one that belongs in a real submission.

## 5. What is misleading about my headline number?

If you only read one section, read this one.

- **"84.5% intent accuracy" sounds strong but the label distribution is
  skewed** (91/200 golden examples are `general_or_feedback`). The trivial
  majority-class baseline already gets 45.5% for free; macro-F1 (0.81) is
  the more honest number, and even that is inflated by the fact that
  `general_or_feedback` is also the system's fallback/default class, so
  disagreements tend to land on the *harder*, rarer intents where mistakes
  matter more (refunds, account) — exactly where macro-F1 is most sensitive
  and reported here, but worth re-stating: don't quote the accuracy number
  alone.
- **The system's intent accuracy is numerically identical to the "simple"
  keyword-only baseline in this report.** That's an artifact of the `mock`
  LLM mode (see Section 2), not evidence that an LLM adds nothing over
  rules — it's evidence that **this specific run doesn't test the LLM at
  all**. The real comparison requires `--llm claude` and is not in this
  report.
- **Reply-quality numbers from the mock run are not included in the results
  table for exactly this reason** — see Section 4. A `mean_overall_score`
  computed by a judge with zero human agreement is worse than no number at
  all, because it looks quantitative.
- **The golden set's escalation gold labels are first-pass heuristic labels
  (`gold_escalate_firstpass`), not fully human-audited** — see
  `eval/build_golden_set.py`'s docstring and `eval/labeling_guidelines.md`.
  Anyone using this repo for a real submission needs to actually review
  the `_reviewed` columns before trusting escalation precision/recall.
- **~85-90% of the raw dataset (non-English traffic) is invisible to every
  number in this report.** "200 golden examples" sounds like a broad sample
  of AmazonHelp traffic; it's a sample of the English-language subset only.
- **"200 golden examples" came from a 100,000-row subsample of a 305,008-row
  raw file**, which itself only yields ~2-4k usable (customer, resolved
  reply) pairs after language filtering and pairing. The golden set is
  therefore drawn from a few percent of the full historical traffic, and a
  full-dataset run would very likely surface intents/phrasings not present
  here (see `data_prep.py --sample 0` to disable subsampling).

## 6. What I'd do next with one more week

1. **Run the real judge.** Everything in Sections 2-4 needs to be redone
   with `--llm claude` and real API calls, then the judge-agreement check
   re-run on that output. This is the single highest-priority next step —
   nothing above should be treated as a quality claim about the actual
   agent until this is done.
2. **Fix escalation recall.** Widen the hard-rule set based on failure mode
   #1, and/or add a lightweight sentiment/repetition score
   (e.g., "already," "still," "again," repeated exclamation/caps) as a
   feature the LLM escalation call is explicitly given, rather than relying
   on it to infer urgency unaided.
3. **Swap TF-IDF for embedding retrieval** (e.g., a local sentence-embedding
   model) to fix failure mode #2, with an ablation comparing reply-quality
   scores between the two retrieval methods on the same golden set.
4. **Add thread context.** Pull in the 1-2 preceding tweets in the thread
   (already present in the raw data via `in_response_to_tweet_id`) so
   context-free leaf messages (failure mode #3) can be classified/answered
   correctly.
5. **Full-dataset pairing + human-reviewed golden set.** Rerun
   `data_prep.py` without subsampling, and have an actual second person
   review 100% of the golden set's `_reviewed` columns per
   `eval/labeling_guidelines.md`, plus a second human rater for the
   judge-agreement check (inter-rater reliability on the humans themselves,
   not just judge-vs-one-human).
