# AmazonHelp AI Support Agent

An AI-assisted customer support prototype built for the **Hiver SDE Intern Take-Home Assignment**.

The system learns from historical AmazonHelp conversations on Twitter/X and performs three core tasks:

1. **Intent Classification** — identifies the customer's primary support intent.
2. **Historical Retrieval** — retrieves historically similar AmazonHelp conversations.
3. **Response Generation & Escalation** — drafts a historically grounded response or escalates the request to a human when automation should not be trusted.

> **Important:** This project is designed to demonstrate not only automation, but also **when the system should and should not be trusted**. Before interpreting any metric, read [`report.md`](report.md), especially Section 5, *"What is misleading about my headline number?"*

---

## 1. Problem Statement

Customer-support conversations on Twitter/X are noisy, short, multi-turn, and frequently missing context from earlier messages in the conversation.

The system therefore needs to:

* Identify the customer's main intent.
* Retrieve relevant historical support examples.
* Generate a response grounded in those historical examples rather than producing a generic response.
* Avoid confidently auto-handling cases where automation may be unreliable.
* Escalate high-risk or ambiguous cases to a human when appropriate.
* Avoid inventing customer-specific information such as order numbers, refund amounts, dates, or names.

---

## 2. Dataset

### Dataset

**Source:** `thoughtvector/customer-support-on-twitter` (Kaggle)

The original dataset contains approximately **3 million tweets** across multiple brands.

### Target Brand

**AmazonHelp**

### Provided Dataset Slice

```text
data/raw/AmazonHelp_sample.csv
```

The provided slice contains approximately **305,000 AmazonHelp-related rows**.

### Data Preparation

Customer tweets were paired with the first AmazonHelp reply that directly answered them using:

* `in_response_to_tweet_id`
* `response_tweet_id`

The data was then filtered to English-language conversations using `langdetect`.

The taxonomy, evaluation rubric, and response-generation process in this project are validated only for English-language examples.

Using a **100,000-row subsample**, the preprocessing pipeline produced:

**6,448 clean `(customer, agent_reply)` pairs**

The full 305k-row dataset is not processed by default because the assignment explicitly states that the evaluator will not run the code on the full dataset.

To disable subsampling and process the complete dataset:

```bash
--sample 0
```

---

## 3. Intent Taxonomy

Seven coarse-grained intents were defined after manually inspecting approximately 300 sampled AmazonHelp customer tweets.

| Intent                | Description                                                                                 |
| --------------------- | ------------------------------------------------------------------------------------------- |
| `delivery_issue`      | Late, missing, lost, or incorrectly tracked shipments                                       |
| `order_cancel_refund` | Order cancellation, refunds, duplicate or incorrect charges                                 |
| `product_defect`      | Item arrived broken, incorrect, damaged, or faulty                                          |
| `returns_exchange`    | Returning or exchanging an item that was already received                                   |
| `account_payment`     | Login, password, billing, membership, or payment-method issues                              |
| `app_tech_issue`      | Problems with Amazon app, website, Fire TV, Kindle, Alexa/Echo, or other technical services |
| `general_or_feedback` | Praise, general questions, feedback, or anything not clearly covered above                  |

The taxonomy is deliberately coarse rather than highly fine-grained.

Retail Twitter support is dominated by a relatively small number of recurring themes. A much finer taxonomy would mostly divide the catch-all category into smaller subcategories without necessarily changing the action the support agent needs to take.

The taxonomy decision is documented in:

```text
decision_log.md
```

See Decision Log **#2** for the rationale.

---

# 4. System Architecture

```text
                    Customer Message
                           |
                           v
              +-------------------------+
              | Intent Classifier       |
              | Rules + LLM             |
              +-------------------------+
                           |
                           v
                    Intent Label
                           |
                           v
              +-------------------------+
              | Historical Retrieval    |
              | TF-IDF + Cosine         |
              | Similarity              |
              +-------------------------+
                           |
                           v
              Similar AmazonHelp Cases
                           |
                           v
              +-------------------------+
              | Escalation Decision     |
              | Hard Rules -> LLM       |
              +-------------------------+
                       /        \
                      /          \
                     v            v
              Auto-Handle      Escalate
                  |                |
                  v                v
          Grounded Draft       Human Review
              Reply
                  |
                  v
             Customer
```

---

# 5. Technical Approach

## 5.1 Intent Classification

Implementation:

```text
src/intents.py
```

The project uses two levels of classification:

### Rule-Based Classifier

A keyword/regex rule set provides:

* A simple baseline.
* A fast offline fallback.
* Deterministic behavior when the LLM is unavailable.

### LLM Classifier

The full system can use an LLM classifier through:

```text
classify_llm
```

If the LLM output cannot be parsed into a valid intent, the system falls back to the rule-based classifier.

---

## 5.2 Historical Reply Retrieval

Implementation:

```text
src/retrieval.py
```

The retrieval system:

1. Builds a TF-IDF index over the **6,448 cleaned historical pairs**.
2. Uses word n-grams from 1-2.
3. Represents customer messages as TF-IDF vectors.
4. Calculates cosine similarity.
5. Retrieves the top-k historically similar customer conversations.

### Why TF-IDF?

TF-IDF was selected instead of embedding-based retrieval because it:

* Requires no model downloads.
* Requires no external embedding service.
* Works completely offline.
* Is lightweight.
* Makes the project easier to reproduce on a locked-down grading machine.

However, this creates an important limitation:

> TF-IDF performs poorly when two messages express the same problem using very different wording.

This failure mode is discussed in [`report.md`](report.md).

---

## 5.3 Reply Generation

Implementation:

```text
src/reply_generator.py
```

The retrieved historical conversations are provided to the LLM as supporting evidence.

The model is instructed to:

* Follow the support patterns seen in historical AmazonHelp conversations.
* Produce a useful and actionable response.
* Remain polite.
* Ground the answer in the retrieved evidence.
* Avoid inventing information.

The system specifically avoids fabricating:

* Order numbers.
* Refund amounts.
* Delivery dates.
* Customer names.
* Other customer-specific information that was not provided.

This constraint directly reflects the evaluation rubric's `correct` criterion.

---

## 5.4 Escalation Decision

Implementation:

```text
src/escalation.py
```

The escalation system uses a hybrid strategy.

### Hard Escalation Rules

Certain signals trigger escalation before an LLM decision is made.

Examples include:

* Legal threats.
* Fraud or safety signals.
* Explicit requests for a human agent.

These rules are **non-overridable**.

### Money / Account Sensitive Cases

Cases involving:

```text
order_cancel_refund
account_payment
```

can be escalated when no sufficiently similar historical case passes the configured similarity threshold.

### LLM-Based Decision

For other cases, the LLM considers:

* Sentiment.
* Urgency.
* Repeated contact.
* Grounding strength.
* Available historical evidence.

The output contains both:

```text
decision
reason
```

rather than returning only a boolean.

If the LLM escalation response cannot be parsed, the system **fails safe and escalates** instead of guessing.

---

# 6. Golden Evaluation Set

A separate evaluation set of **200 customer-support examples** was created using:

```text
eval/build_golden_set.py
```

The examples were stratified by intent using sqrt(bucket-size) weighting so that rare intents were not completely dominated by high-frequency traffic.

## Intent Distribution

| Intent                |   Count |
| --------------------- | ------: |
| `general_or_feedback` |      91 |
| `delivery_issue`      |      41 |
| `order_cancel_refund` |      27 |
| `account_payment`     |      15 |
| `app_tech_issue`      |      12 |
| `product_defect`      |       8 |
| `returns_exchange`    |       6 |
| **Total**             | **200** |

### Gold Escalation Rate

```text
22.0%
```

Each evaluation example contains:

* Customer message.
* Actual historical AmazonHelp reply.
* First-pass heuristic intent label.
* First-pass heuristic escalation label.

The first-pass labels are generated by an **independent labeling routine**, rather than directly copying the system-under-test's output.

This reduces the risk of evaluating the system against its own rules.

Human reviewers can correct these labels using:

```text
eval/labeling_guidelines.md
```

The reviewed fields are:

```text
gold_intent_reviewed
gold_escalate_reviewed
reviewed_by_human
```

> **Important:** The first-pass labels should be manually audited before being treated as verified ground truth.

See [`report.md`](report.md) for the detailed discussion.

---

# 7. Evaluation Results

The following results were generated from:

```text
outputs/metrics.json
```

using:

```bash
--llm mock
```

Therefore, these numbers should be interpreted as **pipeline/reproducibility results**, not evidence of real LLM quality.

---

## 7.1 Intent Classification

| Metric   | System | Simple Baseline | Trivial Baseline |
| -------- | -----: | --------------: | ---------------: |
| Accuracy |  84.5% |           84.5% |            45.5% |
| Macro F1 |  81.1% |           81.1% |             8.9% |

In mock mode, the system and simple baseline have identical intent performance because the mock LLM's intent implementation uses the same rule-based classifier underneath.

Therefore, this comparison does **not** demonstrate an advantage from LLM-based classification.

A meaningful LLM comparison requires running with a real LLM configuration.

---

## 7.2 Escalation Decision

| Metric                    | System | Simple Baseline | Trivial Baseline |
| ------------------------- | -----: | --------------: | ---------------: |
| Precision                 |  33.3% |           50.0% |             0.0% |
| Recall                    |  11.4% |            4.6% |             0.0% |
| F1                        |  17.0% |            8.3% |             0.0% |
| Predicted Escalation Rate |   7.5% |            2.0% |             0.0% |
| Gold Escalation Rate      |  22.0% |           22.0% |            22.0% |

The major weakness is **escalation recall**.

The system identifies substantially fewer escalation cases than the gold evaluation set contains.

This is a key limitation rather than a result to hide.

See [`report.md`](report.md) Failure Mode #1.

---

# 8. Baselines

## 8.1 Trivial Baseline

Implementation:

```text
baselines/trivial_baseline.py
```

Behavior:

* Always predicts the majority intent.
* Always returns the same canned response.
* Never escalates.

Purpose:

> Establish a minimum performance floor.

---

## 8.2 Simple Baseline

Implementation:

```text
baselines/simple_baseline.py
```

Behavior:

* Keyword-rule intent classification.
* 1-nearest-neighbor historical reply.
* No response generation.
* Keyword-only escalation.
* No LLM calls.

Purpose:

> Isolate the value added by the LLM generation and judgment stages.

---

# 9. Reply Quality Evaluation

Implementation:

```text
eval/judge.py
```

The project contains an LLM-as-judge evaluation rubric covering:

* Groundedness.
* Correctness.
* Politeness.
* Actionability.
* Overall score from 1-5.

To determine whether the judge itself is trustworthy, **30 examples were independently rated by a human**.

The results are stored in:

```text
outputs/judge_agreement_mock.txt
```

## Mock Judge Agreement

| Metric                 |                               Result |
| ---------------------- | -----------------------------------: |
| Mean Human Score (1-5) |                                 2.47 |
| Mean Mock-Judge Score  |                                 5.00 |
| Spearman Correlation   | Undefined due to zero judge variance |
| Exact Match Rate       |                                 0.0% |
| Cohen's Kappa          |                                  0.0 |

### Important Finding

The mock judge has **zero measured agreement with the human ratings**.

Therefore:

> **The mock judge must not be used to make reply-quality claims.**

The mock judge exists only to allow the pipeline to execute end-to-end without requiring an API key.

This limitation is intentionally reported rather than hidden.

---

# 10. Top Failure Modes

## Failure Mode 1 — Escalation Recall Is Too Low

Current escalation recall:

```text
11.4%
```

Compared with the gold escalation rate:

```text
22.0%
```

Example pattern:

> A customer reports that their issue remains unresolved after repeatedly providing details.

This type of repeat-contact escalation may not contain obvious hard-rule keywords and can therefore reach the LLM judgment stage.

### Possible Improvement

Add stronger signals for:

* Repeated contact.
* Long unresolved cases.
* Increasing frustration.
* Multiple previous support attempts.
* Strong urgency language.

---

## Failure Mode 2 — TF-IDF Retrieval Misses Paraphrases

Example:

> "I've just had an alleged failed delivery attempt even though I've been at home all day."

A historically similar case may exist but use completely different vocabulary.

Because TF-IDF depends on lexical overlap, the retriever may select an unrelated conversation.

### Possible Improvement

Replace or augment TF-IDF with embedding-based retrieval.

---

## Failure Mode 3 — Missing Thread Context

Short messages such as:

```text
"So that's it!!"
```

or:

```text
"I've just had this"
```

may be replies to an earlier tweet.

The current pipeline sees only the individual message.

Consequently, it may not have enough information to determine:

* The customer's actual problem.
* The correct intent.
* Whether escalation is required.
* What response should be generated.

### Possible Improvement

Include the previous 1-2 tweets from the conversation thread.

---

## Failure Mode 4 — Occasional Hallucinated Personalization

Some generated replies may introduce information that was never provided.

Examples include:

* Customer names.
* Specific order information.
* Dates.
* Other unsupported details.

This violates the grounding requirement.

The response-generation prompt therefore explicitly prohibits invented customer-specific information.

---

## Failure Mode 5 — Non-English Traffic Is Dropped

Approximately **85-90% of raw rows are excluded by the English-language filter**.

Therefore:

> All evaluation numbers in this README apply only to the English-language subset.

The current system does not attempt multilingual support.

---

# 11. What Is Misleading About the Headline Number?

The headline:

```text
84.5% intent accuracy
```

should **not** be interpreted as:

> "The AI support agent is 84.5% good."

There are several reasons.

### 1. Mock Mode

The evaluation was performed using:

```text
--llm mock
```

The mock intent implementation uses the same rule-based classifier as the simple baseline.

Therefore, the system's:

```text
84.5% accuracy
```

is identical to the simple baseline.

---

### 2. Class Distribution

The majority intent alone achieves:

```text
45.5%
```

accuracy.

Therefore, accuracy alone does not adequately describe performance across all intents.

Macro F1 is also reported to account for class-level performance.

---

### 3. Intent Accuracy Does Not Measure Reply Quality

A system can correctly identify an intent and still generate a poor response.

Intent classification therefore measures only one component of the complete support workflow.

---

### 4. Escalation Performance Is Weak

The system's escalation recall is only:

```text
11.4%
```

against a:

```text
22.0%
```

gold escalation rate.

This is important because a support agent must know when **not** to automate.

---

### 5. Mock Judge Cannot Validate Reply Quality

The mock judge showed:

```text
0.0 Cohen's kappa
0.0% exact agreement
```

with the human ratings.

Therefore, mock-mode reply-quality scores are not evidence of real response quality.

---

# 12. What I Would Do With One More Week

## Priority 1 — Run the Real Judge

Run the complete evaluation using a real LLM and repeat the judge-agreement experiment.

This would provide evidence about whether the response-quality evaluation is meaningful.

---

## Priority 2 — Improve Escalation Recall

Expand the hard-rule set and introduce explicit signals for:

* Repeat contact.
* Unresolved issues.
* Strong frustration.
* Urgency.
* Safety/fraud/legal language.

---

## Priority 3 — Improve Retrieval

Experiment with embedding-based retrieval.

Run an ablation comparing:

```text
TF-IDF
vs.
Embedding Retrieval
```

on the same evaluation set.

---

## Priority 4 — Add Thread Context

Include the preceding 1-2 tweets where available.

This should help with context-free messages.

---

## Priority 5 — Improve Evaluation Quality

Create a fully human-reviewed golden set and use a second independent reviewer to measure inter-rater reliability.

---

# 13. Repository Structure

```text
repo/
│
├── data/
│   ├── raw/
│   │   └── AmazonHelp_sample.csv
│   │
│   └── processed/
│       └── pairs.csv
│
├── src/
│   ├── data_prep.py
│   ├── intents.py
│   ├── retrieval.py
│   ├── reply_generator.py
│   ├── escalation.py
│   ├── llm_client.py
│   ├── pipeline.py
│   └── evaluate.py
│
├── baselines/
│   ├── trivial_baseline.py
│   └── simple_baseline.py
│
├── eval/
│   ├── build_golden_set.py
│   ├── labeling_guidelines.md
│   ├── golden_set.csv
│   ├── judge.py
│   └── judge_agreement.py
│
├── outputs/
│   ├── eval_results.csv
│   ├── metrics.json
│   ├── human_ratings_template.csv
│   └── judge_agreement_mock.txt
│
├── scripts/
│   └── run_all.sh
│
├── requirements.txt
├── report.md
├── decision_log.md
├── README.md
└── .gitignore
```

> Generated files such as processed datasets and evaluation outputs may be regenerated by the pipeline depending on the repository version.

---

# 14. Installation & Setup

## Requirements

Recommended environment:

* Python 3.10+
* Git
* Bash shell
* Internet connection for real LLM API calls
* A Groq, Claude, or Gemini API key for real LLM evaluation

The mock mode does **not** require an API key.

---

## 14.1 Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Agent
```

---

## 14.2 Create a Virtual Environment

### Linux / macOS / Git Bash

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
python -m venv venv
venv\Scripts\activate
```

---

## 14.3 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 15. Running the Project

## 15.1 Recommended First Run — Mock Mode

The easiest way to verify that the repository works is:

```bash
bash scripts/run_all.sh mock 100000
```

This mode requires **no API key**.

The pipeline performs:

```text
100,000 raw rows
        ↓
Data preprocessing
        ↓
Clean customer/reply pairs
        ↓
Golden evaluation set
        ↓
5-example pipeline demonstration
        ↓
200-example evaluation
        ↓
Baseline evaluation
        ↓
Metrics generation
```

Expected output files include:

```text
outputs/metrics.json
outputs/eval_results.csv
```

This is the recommended first command for an evaluator.

---

# 16. Running With a Real LLM

Real LLM execution requires an API key.

> **Never commit an API key to GitHub.**

The repository should contain only a template such as:

```text
.env.example
```

Example:

```env
GROQ_API_KEY=your_groq_api_key_here
```

The real key should remain local or be supplied through an environment variable.

---

## 16.1 Groq

Set the environment variable.

### Git Bash

```bash
export GROQ_API_KEY="gsk_your_key_here"
```

Then run:

```bash
bash scripts/run_all.sh groq 100000
```

### Windows PowerShell

```powershell
$env:GROQ_API_KEY="gsk_your_key_here"
```

Then:

```powershell
bash scripts/run_all.sh groq 100000
```

For a permanent Windows user environment variable:

```powershell
[Environment]::SetEnvironmentVariable(
    "GROQ_API_KEY",
    "gsk_your_key_here",
    "User"
)
```

After setting a permanent environment variable, restart Git Bash / VS Code before running the project.

---

## 16.2 Claude

Set:

```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

Then:

```bash
bash scripts/run_all.sh claude 100000
```

On Windows PowerShell:

```powershell
$env:ANTHROPIC_API_KEY="your_api_key_here"
```

---

## 16.3 Gemini

Set:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Then:

```bash
bash scripts/run_all.sh gemini 100000
```

On Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

---

# 17. Run a Single Message

The pipeline can also process an individual message.

Example:

```bash
python src/pipeline.py \
  --llm mock \
  --text "My package says delivered but I never got it, this is the second time this month"
```

The pipeline returns the predicted:

* Intent.
* Retrieved historical examples.
* Escalation decision.
* Escalation reason.
* Draft response.

---

# 18. Judge vs Human Agreement

To generate the human-rating template:

```bash
python eval/judge_agreement.py \
  --results outputs/eval_results.csv \
  --n 30
```

This creates:

```text
outputs/human_ratings_template.csv
```

After manually entering the human ratings:

```bash
python eval/judge_agreement.py \
  --template outputs/human_ratings_template.csv \
  --score
```

This calculates agreement statistics between the judge and human ratings.

---

# 19. Reproducibility

The project is designed so that an evaluator can first run the entire pipeline without an API key.

### Fast reproducibility check

```bash
bash scripts/run_all.sh mock 100000
```

### Real LLM evaluation

```bash
bash scripts/run_all.sh groq 100000
```

or:

```bash
bash scripts/run_all.sh claude 100000
```

or:

```bash
bash scripts/run_all.sh gemini 100000
```

The exact runtime depends on:

* Hardware.
* Python environment.
* Dataset size.
* API latency.
* API rate limits.
* LLM provider.

---

# 20. Security & API Keys

API keys are intentionally **not included in this repository**.

Before pushing to GitHub, verify that no secret is present:

```bash
git status
```

Check tracked files:

```bash
git ls-files
```

The `.gitignore` should contain:

```gitignore
.env
*.pyc
__pycache__/
```

If using environment variables, the source code should access them through:

```python
os.environ.get("GROQ_API_KEY")
```

rather than hard-coding credentials.

> **Never commit real API keys, passwords, tokens, or private credentials to the repository.**

---

# 21. Decision Log

Non-obvious engineering decisions are documented in:

```text
decision_log.md
```

Important decisions include:

1. Reading raw data manually before writing intent rules.
2. Using a coarse seven-intent taxonomy.
3. Keeping baselines LLM-free.
4. Using `langdetect` for language filtering.
5. Pairing customer tweets with the first directly answering brand reply.
6. Using non-overridable hard rules for high-risk escalation.
7. Choosing TF-IDF instead of embeddings for offline reproducibility.
8. Creating a deterministic mock LLM fallback.
9. Using an independent labeling function for golden-set first-pass labels.
10. Using sqrt-weighted stratified sampling.
11. Defaulting escalation labels to "escalate when unsure."

---

# 22. Limitations & Trustworthiness

This project is a **prototype**, not a production customer-support system.

Important limitations include:

### Dataset

The provided dataset is noisy and multi-turn.

The current pipeline processes individual messages rather than complete conversation threads.

### Golden Labels

The evaluation set initially contains first-pass heuristic labels.

These require human review before being treated as verified ground truth.

### Retrieval

TF-IDF retrieval can fail when semantically similar messages use different vocabulary.

### Language Coverage

Only English-language traffic is currently processed.

Approximately 85-90% of raw rows are removed by the language filter.

### Mock LLM

The current reported metrics were generated using:

```text
--llm mock
```

Therefore, they should not be interpreted as evidence of real LLM performance.

### Judge Reliability

The mock judge demonstrated essentially zero agreement with the independent human ratings.

Therefore, mock judge results should not be used to claim strong response quality.

### Escalation

Escalation recall is currently too low and requires further improvement.

---

# 23. Key Findings

The most important findings from the current evaluation are:

```text
Intent accuracy:
84.5%

Intent macro F1:
81.1%

Gold escalation rate:
22.0%

System escalation recall:
11.4%

Mock judge / human Cohen's kappa:
0.0
```

The results demonstrate that the pipeline is executable and that the intent classifier performs substantially above the trivial majority-class baseline.

However, the results also expose important weaknesses:

* The intent classifier does not yet demonstrate an advantage over the simple rule-based baseline in mock mode.
* Escalation recall is too low.
* TF-IDF retrieval struggles with paraphrases.
* Thread context is missing.
* The mock judge cannot currently validate response quality.
* English-only filtering excludes most of the raw dataset.

These limitations are intentionally documented as part of the evaluation.

---

# 24. Conclusion

This project demonstrates an end-to-end AI-assisted customer-support workflow built around historical AmazonHelp Twitter/X conversations.

The system:

```text
Customer Message
       ↓
Intent Classification
       ↓
Historical Retrieval
       ↓
Escalation Decision
       ↓
 ┌───────────────┐
 │               │
Auto-Handle   Human Review
 │
 ↓
Grounded Reply
```

The primary objective is not simply to maximize a single metric.

Instead, the project demonstrates:

* How to classify customer-support requests.
* How to retrieve historically similar conversations.
* How to generate grounded responses.
* How to identify cases requiring human review.
* How to evaluate against simple baselines.
* How to identify weaknesses in the evaluation itself.
* How to distinguish reproducibility from genuine model quality.

The current results show a functioning end-to-end prototype, while also making clear that **escalation recall, retrieval robustness, thread context, human-reviewed labels, and real-LLM judge validation remain the main areas for improvement**.

For the detailed analysis, failure cases, and interpretation of the evaluation results, see:

* [`report.md`](report.md)
* [`decision_log.md`](decision_log.md)
* [`eval/labeling_guidelines.md`](eval/labeling_guidelines.md)

---

## Quick Start

For an evaluator who wants to verify the project quickly:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Agent

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

bash scripts/run_all.sh mock 100000
```

No API key is required for the mock run.

For real LLM evaluation, configure the appropriate API key and run:

```bash
bash scripts/run_all.sh groq 100000
```

or:

```bash
bash scripts/run_all.sh claude 100000
```

or:

```bash
bash scripts/run_all.sh gemini 100000
```

