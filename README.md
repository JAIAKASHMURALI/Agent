# AmazonHelp AI Support Agent

An AI-assisted customer support prototype built for the Hiver SDE Intern take-home assignment.

The system learns from historical AmazonHelp conversations on Twitter/X and performs three tasks:

1. Classifies a customer message into an intent.
2. Retrieves historically similar AmazonHelp conversations.
3. Drafts a historically grounded response, or escalates the request to a human.

The goal is not just automation, but demonstrating **when the system should and should not be trusted** — see [`report.md`](report.md) Section 5 ("What is misleading about my headline number") before quoting any metric below out of context.

## 1. Problem

Customer-support conversations on Twitter are noisy, short, multi-turn, and often missing context from earlier in the thread. The system therefore needs to:

- identify the customer's main intent,
- find relevant historical support examples,
- produce a reply grounded in those examples rather than a generic one,
- avoid confidently auto-handling cases where automation is unreliable (money, account access, legal/safety signals, or messages too ambiguous to answer safely).

## 2. Dataset

**Dataset:** `thoughtvector/customer-support-on-twitter` (Kaggle), ~3M tweets across many brands.

**Target brand:** AmazonHelp

**Provided slice:** `data/raw/AmazonHelp_sample.csv`, ~305,000 rows already filtered to AmazonHelp threads.

Customer tweets were paired with the first AmazonHelp reply that directly answered them, using `in_response_to_tweet_id` / `response_tweet_id`, then filtered to English only (`langdetect`) since the taxonomy, judge rubric, and reply drafting are only validated on English. From a 100,000-row subsample, this produced **6,448 clean (customer, agent_reply) pairs**. The full 305k-row file is not fully processed by default, per the assignment's "we will not run your code on the full dataset" rule (`--sample 0` disables subsampling if you want a full run).

## 3. Intent Taxonomy

Seven intents were defined by reading ~300 sampled AmazonHelp customer tweets by hand before writing any rules:

| Intent | Description |
|---|---|
| `delivery_issue` | Late, missing, lost, or mis-tracked shipments |
| `order_cancel_refund` | Cancellations, refunds, double/incorrect charges |
| `product_defect` | Item arrived broken, wrong, or faulty |
| `returns_exchange` | How/where to return or exchange an item already received |
| `account_payment` | Login, password, billing, membership, payment method issues |
| `app_tech_issue` | Bugs/errors in the Amazon app, website, Fire TV, Kindle, Alexa/Echo |
| `general_or_feedback` | Praise, general questions, or anything not clearly one of the above |

Kept deliberately coarse rather than Banking77-style fine-grained (see `decision_log.md` #2): retail Twitter support is dominated by a handful of high-frequency themes, and a finer taxonomy would mostly split hairs within the catch-all bucket without changing what the agent needs to *do* differently.

## 4. System Architecture

```
Customer Message
       |
       v
Intent Classifier (rules + LLM)
       |
       +--------------------+
       |                    |
       v                    v
Intent Label          Historical Retrieval (TF-IDF)
                            |
                            v
                     Similar AmazonHelp
                       Resolved Cases
                            |
                            v
                    Escalation Decision
                    (hard rules -> LLM)
                       /           \
                      /             \
                 Auto-Handle      Escalate
                   |                 |
                   v                 v
          Grounded Draft Reply   Human Review
           (LLM + retrieved
              evidence)
```

## 5. Technical Approach

### Intent Classification
A keyword/regex rule set (`src/intents.py`) acts as both the "simple baseline" and a fast offline fallback; an LLM classifier (`classify_llm`) is used by the full system and falls back to the rules if its output can't be parsed.

### Historical Reply Retrieval
`src/retrieval.py` builds a TF-IDF (1-2 gram) index over the 6,448 historical pairs and retrieves the top-k most similar past customer messages by cosine similarity. TF-IDF was chosen over embeddings so the pipeline needs zero model downloads / zero internet, keeping the "reproduce in under 15 minutes" promise honest on a locked-down grading machine — see `decision_log.md` #7 and `report.md` failure mode #2 for where this breaks down (paraphrases with no lexical overlap).

### Reply Drafting
`src/reply_generator.py` passes the retrieved historical cases to the LLM and instructs it to follow the brand's established moves (apologize, ask to DM, point to a help link) **without inventing** order numbers, refund amounts, or dates not given by the customer — this constraint mirrors the judge rubric's `correct` criterion directly.

### Escalation Decision
`src/escalation.py` escalates when:
- a hard rule fires (legal threat, fraud/safety signal, explicit request for a human) — these are non-overridable and checked *before* any LLM call;
- the intent is money/account-sensitive (`order_cancel_refund`, `account_payment`) **and** no similar historical case clears a similarity floor;
- otherwise, an LLM call weighs sentiment/urgency and grounding strength, returning a decision **and a stated reason** (never just a boolean).

If the LLM's escalation output can't be parsed, the system fails safe and escalates rather than guesses.

## 6. Golden Evaluation Set

A separate evaluation set of **200 customer-support examples** was built (`eval/build_golden_set.py`), stratified by intent using sqrt(bucket-size) weighting so rare intents aren't drowned out by high-frequency traffic:

| Intent | Count |
|---|---:|
| `general_or_feedback` | 91 |
| `delivery_issue` | 41 |
| `order_cancel_refund` | 27 |
| `account_payment` | 15 |
| `app_tech_issue` | 12 |
| `product_defect` | 8 |
| `returns_exchange` | 6 |

Gold escalation rate: **22%**.

Each example contains the customer message, the actual historical AmazonHelp reply, a first-pass heuristic intent label, and a first-pass heuristic escalation label — produced by an **independent** labeling routine, not the same code as the system under test, to avoid grading the system against its own rules. `eval/labeling_guidelines.md` is the rubric a human reviewer follows to correct these into verified ground truth (`gold_intent_reviewed` / `gold_escalate_reviewed`, with a `reviewed_by_human` flag). **These columns should be manually audited before being trusted as ground truth** — see `report.md` Section 5.

## 7. Evaluation Results

All results below are from `outputs/metrics.json`, produced with `--llm mock` (see Section 9 for why this matters).

### Intent Classification

| Metric | System | Simple baseline | Trivial baseline |
|---|---:|---:|---:|
| Accuracy | 84.5% | 84.5% | 45.5% |
| Macro F1 | 81.1% | 81.1% | 8.9% |

System and simple-baseline accuracy are identical in mock mode because the mock LLM's intent call *is* the rule-based classifier under the hood — this comparison only becomes meaningful once run with `--llm claude` (see Section 12).

### Escalation Decision

| Metric | System | Simple baseline | Trivial baseline |
|---|---:|---:|---:|
| Precision | 33.3% | 50.0% | 0.0% |
| Recall | 11.4% | 4.6% | 0.0% |
| F1 | 17.0% | 8.3% | 0.0% |
| Predicted escalation rate | 7.5% | 2.0% | 0.0% |
| Gold escalation rate | 22.0% | 22.0% | 22.0% |

Recall is the headline weakness here for every system: none catch anywhere near the true 22% escalation rate. The hybrid system beats the keyword-only baseline on recall (hard rules + money-intent-without-grounding rules) but is still far too conservative — see `report.md` failure mode #1.

## 8. Baselines

**Trivial baseline** (`baselines/trivial_baseline.py`): always predicts the majority intent (`general_or_feedback`), always returns the same canned reply, never escalates. Exists purely to give the real system a floor to beat.

**Simple baseline** (`baselines/simple_baseline.py`): keyword-rule intent, 1-nearest-neighbor reply copy (no generation/adaptation), keyword-only escalation. No LLM calls at all — isolates how much value the LLM generation/judgment step adds over pure retrieval + rules.

## 9. Reply Quality Evaluation

An LLM-as-judge rubric (`eval/judge.py`) scores drafted replies on **grounded / correct / polite / actionable** plus a 1-5 overall score. To check whether this judge can be trusted, 30 examples were **hand-rated by a human independently** and compared against the judge's score (`eval/judge_agreement.py`, results in `outputs/judge_agreement_mock.txt`):

| | Result |
|---|---:|
| Mean human score (1-5) | 2.47 |
| Mean mock-judge score | 5.00 (constant) |
| Spearman correlation | undefined (zero judge variance) |
| Exact match rate | 0.0% |
| Cohen's kappa (good ≥4 threshold) | 0.0 |

**The mock judge has zero measured agreement with a human and must not be used to make any reply-quality claim.** It exists only so the pipeline is runnable end-to-end without an API key. This is reported as a real finding, not hidden — running `--llm claude` and redoing this check is priority #1 in Section 12.

## 10. Top Failure Modes

**1. Escalation recall is far too low (11% vs. a 22% gold rate).**
Example: *"...my issue is still not resolved nor I have received the product... shared the details in links for 10 times"* — a textbook repeat-contact escalation that hits no hard-rule keyword and falls through to LLM judgment. The hard-rule list is likely too narrow.

**2. TF-IDF retrieval fails on paraphrases with no lexical overlap.**
Example: *"I've just had an alleged failed delivery attempt even though I've been at home all day"* — no historical case shares this phrasing, so the nearest neighbor pulled an unrelated case and the draft reply hallucinated an irrelevant detail.

**3. Context-free, ambiguous tweets are confidently mis-handled.**
Messages like *"So that's it!!"* or *"I've just had this"* are almost certainly thread replies where the real complaint is in an earlier tweet the single-message pipeline never sees.

**4. Occasional hallucinated personalization.**
A few drafted replies invented a customer name never given in the message or the retrieved case — exactly what the judge's `correct` criterion exists to catch.

**5. Non-English traffic is silently dropped, not handled.**
~85-90% of raw rows are excluded by the language filter; every number in this README applies only to the English subset.

## 11. What Is Misleading About My Headline Number?

The headline "84.5% intent accuracy" should not be read as "the agent is 84.5% good." It's identical to the rule-only simple baseline in this mock-mode run (Section 7), it's measured against a skewed label distribution where the majority class alone gets 45.5% for free, and it says nothing about reply quality or escalation judgment. The reply-quality numbers from the mock run are likewise omitted from the results table on purpose, because a judge with zero measured human agreement (Section 9) produces numbers that look quantitative but aren't evidence. See `report.md` Section 5 for the full discussion, including the golden set's label-review status and the dataset's language coverage gap.

## 12. What I Would Do With One More Week

1. Run the real judge — redo Sections 7-9 with `--llm claude` and real API calls; nothing above should be treated as a quality claim until this is done.
2. Fix escalation recall — widen the hard-rule set and add an explicit sentiment/repetition signal rather than relying on the LLM to infer urgency unaided.
3. Swap TF-IDF for embedding-based retrieval to fix paraphrase failures, with an ablation on the same golden set.
4. Add thread context (the 1-2 preceding tweets) so context-free leaf messages can be classified/answered correctly.
5. Full-dataset pairing plus a fully human-reviewed golden set, with a second rater for inter-rater reliability on the judge-agreement check itself.

## 13. Repository Structure

```text
repo/
├── data/
│   ├── raw/AmazonHelp_sample.csv       # provided AmazonHelp-filtered slice
│   └── processed/pairs.csv              # generated: cleaned (customer, reply) pairs
├── src/
│   ├── data_prep.py
│   ├── intents.py
│   ├── retrieval.py
│   ├── reply_generator.py
│   ├── escalation.py
│   ├── llm_client.py
│   ├── pipeline.py
│   └── evaluate.py
├── baselines/
│   ├── trivial_baseline.py
│   └── simple_baseline.py
├── eval/
│   ├── build_golden_set.py
│   ├── labeling_guidelines.md
│   ├── golden_set.csv
│   ├── judge.py
│   └── judge_agreement.py
├── outputs/
│   ├── eval_results.csv
│   ├── metrics.json
│   ├── human_ratings_template.csv
│   └── judge_agreement_mock.txt
├── scripts/run_all.sh
├── requirements.txt
├── report.md
├── decision_log.md
└── README.md
```

## 14. How to Run

### 1. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Reproduce the headline results (< 15 minutes, no API key needed)
```bash
bash scripts/run_all.sh mock 100000
```
This subsamples 100,000 raw rows, builds clean pairs, builds the 200-example golden set, runs a 5-example live pipeline demo, then runs the full evaluation harness (system + both baselines), writing `outputs/metrics.json` and `outputs/eval_results.csv`.

### 4. Run with a real LLM (Claude)
```bash
export ANTHROPIC_API_KEY=sk-ant-...
bash scripts/run_all.sh claude 100000
#option 2 (gemini model)
 export GEMINI_API_KEY="your api key .."
  bash scripts/run_all.sh gemini 100000
#option 3 (groq model)
 export GROQ_API_KEY="gsk_your_key_here"
bash scripts/run_all.sh groq 100000

This is the mode whose output should actually be trusted for quality claims (Section 9/12). Budget more time — each of the 200 golden examples makes 4 API calls."

### 5. Try one message ad-hoc
```bash
python3 src/pipeline.py --llm mock --text "My package says delivered but I never got it, this is the second time this month"
```

### 6. Check judge-vs-human agreement
```bash
python3 eval/judge_agreement.py --results outputs/eval_results.csv --n 30   # writes a template to fill in by hand
python3 eval/judge_agreement.py --template outputs/human_ratings_template.csv --score
```

## 15. Decision Log

Non-obvious design decisions are documented in [`decision_log.md`](decision_log.md), including: reading raw data by hand before writing intent rules, choosing a coarse 7-intent taxonomy, keeping baselines LLM-free, switching from an ASCII-ratio to `langdetect` for language filtering, pairing customer tweets with only the *first* brand reply, using non-overridable hard rules for high-stakes escalation, choosing TF-IDF over embeddings for offline reproducibility, building a deterministic mock LLM fallback, using an independent labeling function for golden-set first-pass labels, sqrt-weighted stratified sampling, and defaulting escalation gold labels to "escalate when unsure."

## 16. Limitations and Trustworthiness

This is a prototype, not a production customer-support system.

- The provided dataset slice is noisy and multi-turn; the pipeline only looks at single messages, not full thread context.
- Golden-set labels are first-pass heuristic labels and need human review before being trusted as ground truth (Section 6).
- Retrieval is TF-IDF, which fails on paraphrases with no lexical overlap (Section 10).
- Every number in this README was produced in `--llm mock` mode; the real, evaluated numbers require `--llm claude` with an API key and have not yet been produced (Section 12, #1).
- The LLM-judge's agreement with a human was measured and found to be essentially zero in mock mode (Section 9) — this must be re-measured in `claude` mode before the judge is trusted.
- The system covers English-language traffic only (~10-15% of the raw AmazonHelp data).

## 17. Conclusion

This project demonstrates an end-to-end approach for turning noisy AmazonHelp Twitter conversations into an AI-assisted support workflow: classifying incoming messages, retrieving historically similar resolved cases, drafting a grounded reply, and deciding whether to auto-handle or escalate — with a stated reason either way. The evaluation is built to expose weaknesses rather than hide them: escalation recall is currently too low, the offline mock judge has no measured agreement with a human, and the intent classifier's advantage over a keyword-only baseline has not yet been demonstrated with a real LLM. Those three gaps, not the headline accuracy number, are the honest summary of where this system stands.
