# Decision Log

Plain list of non-obvious decisions and why, in roughly the order they came up.

1. **Read ~300 raw customer tweets by hand before writing any intent rules.**
   Rather than guessing a taxonomy from the assignment brief, I skimmed the
   raw CSV's `text` column filtered to inbound=True to see what people
   actually contact AmazonHelp about, then wrote the 7-intent taxonomy in
   `src/intents.py` to match observed frequency and specificity, not a
   textbook taxonomy.

2. **7 coarse intents, not Banking77's 77.** Banking77 is designed for a
   narrow, well-separated domain (banking queries). Retail Twitter support
   is dominated by a handful of high-frequency themes (delivery, refunds,
   defects, returns, account, app bugs, everything-else). A 77-way taxonomy
   here would mostly split hairs within `general_or_feedback`-equivalent
   buckets without changing what the agent needs to *do* differently.

3. **Kept the "trivial" and "simple" baselines LLM-free and offline.**
   This makes them free to run repeatedly, gives a hard floor that doesn't
   depend on API availability, and cleanly isolates "what does the LLM step
   add" as a single comparison (system vs. simple baseline), rather than
   confounding retrieval, generation, and classification changes together.

4. **Used `langdetect` instead of an ASCII-character-ratio heuristic for
   English filtering.** An early version used "%% of characters that are
   ASCII letters" as a proxy for English and it silently let through
   French/Spanish/Portuguese text (Latin alphabets look "ASCII" too). Caught
   this by eyeballing pipeline demo output during development — see
   `report.md` Section 5 for how much of the dataset this filter removes.

5. **Only pair a customer tweet with the *first* AmazonHelp reply that
   directly answered it** (`data_prep.py::build_pairs`), not the whole
   downstream thread. A thread can wander into unrelated territory after
   the first exchange; the first reply is the cleanest "how did the brand
   initially respond to this problem" signal, which is what "grounded in
   how this brand has historically resolved similar issues" is asking for.

6. **Escalation uses hard, non-overridable keyword rules *before* asking
   the LLM anything** (legal threats, fraud, safety, explicit human
   request). A false "auto-handle" on these is much more costly than an
   unnecessary escalation, so I didn't want a single LLM call's judgment
   to be the only thing standing between a self-harm mention and an
   automated reply. Also fail-safe: if the LLM's escalation JSON can't be
   parsed, we escalate rather than guess (`escalation.py::decide_escalation`).

7. **TF-IDF (not embeddings) for historical-case retrieval.** Chosen so the
   entire pipeline runs with zero model downloads / zero internet, keeping
   the "reproduce in under 15 minutes" promise honest on an offline or
   locked-down grading machine. Documented as a known limitation (report.md
   failure mode #2) rather than pretending it's the best possible choice.

8. **Built a deterministic `MockClient` fallback for every LLM call, used
   as the default `--llm` mode.** The assignment must be gradeable without
   assuming the grader has (or wants to spend) API credits. The mock is
   intentionally crude and is never used to make a quality claim — see
   report.md Section 4, where the mock judge's near-zero agreement with a
   human is reported as a real finding, not hidden.

9. **Golden-set first-pass labels use an independent labeling function**
   (`eval/build_golden_set.py::heuristic_label_intent`), not the same code
   as the system-under-test's rule classifier (`src/intents.py`). Grading a
   system against labels produced by its own rule-matching code would
   inflate the intent-accuracy number for exactly the cases the rules were
   written to catch.

10. **Golden set is stratified by intent using sqrt(bucket size) weighting**,
    not raw frequency. Pure frequency sampling would give ~6 examples total
    for `returns_exchange`/`product_defect` out of 200 in a 100k-row
    subsample; sqrt-weighting keeps the sample closer to real traffic mix
    while still guaranteeing enough examples per rare intent to say
    anything about them (see `eval/build_golden_set.py::stratified_sample`).

11. **Escalation gold labels default to "escalate when unsure."** Per
    `eval/labeling_guidelines.md`: a false auto-handle is worse than an
    unnecessary escalation for a brand at this scale, so both the system's
    rules and the labeling guidance encode the same asymmetric cost.

12. **`gold_intent_reviewed` / `gold_escalate_reviewed` columns exist
    separately from the `_firstpass` columns and start as an unreviewed
    copy**, with an explicit `reviewed_by_human` flag defaulting to `False`.
    This makes it impossible to accidentally treat auto-generated labels as
    human-verified ground truth without someone actively going through and
    flipping that flag — see the loud warning in
    `eval/build_golden_set.py`'s docstring and report.md Section 5.

13. **Data prep defaults to subsampling the raw 305k-row file (100k rows)**
    rather than processing everything, per the assignment's explicit "we
    will not run your code on the full dataset" rule. `--sample 0` disables
    this if a full run is ever wanted.

14. **Kept reply drafts to ≤280 characters and forbade inventing order
    numbers / refund amounts / dates in the system prompt**
    (`reply_generator.py::REPLY_SYSTEM_PROMPT`), directly mirroring the
    real @AmazonHelp account's format and the `judge.py` `correct` rubric
    criterion, so the generation constraint and the evaluation criterion
    are checking the same thing rather than talking past each other.
