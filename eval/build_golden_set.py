"""
build_golden_set.py
--------------------
Builds the 150-250 example golden evaluation set (deliverable #2).

Sampling method (documented, and repeated in report.md):
1. Start from the cleaned English (customer_text, agent_reply) pairs.
2. Stratify by rule-based intent bucket (src/intents.py) so rare-but-important
   intents (e.g. account_payment, order_cancel_refund) aren't drowned out by
   the much more common delivery_issue / general_or_feedback traffic.
3. Within each stratum, sample proportionally to sqrt(bucket size) rather than
   raw size -- a compromise between "reflect real traffic mix" and "get enough
   examples of rare intents to say anything statistically meaningful about them."
4. Cap total at --n (default 200).

Labeling method (documented -- read this before trusting the labels):
This script produces FIRST-PASS labels using a stricter, independent labeling
routine (`heuristic_label_intent` / `heuristic_label_escalate` below) that is
deliberately NOT the same code path as the rule-based classifier being
evaluated (src/intents.py), to avoid the eval harness grading the system
against its own rules. These first-pass labels are meant to be spot-checked
and corrected by a human before being trusted as ground truth -- see
`gold_intent_reviewed` / `gold_escalate_reviewed` columns, which start as a
copy of the first-pass label and a `reviewed=False` flag. eval/labeling_guidelines.md
documents exactly how a human reviewer should adjudicate each column.
IMPORTANT: for a real submission, you (the assignment author) should sit down
and actually review/correct the `_reviewed` columns for all 150-250 rows --
we pre-filled them to make the harness runnable, but "hand-labelled by us" in
the assignment brief means a human looked at every row.
"""
import argparse
import math
import re
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from intents import classify_rule_based, INTENTS  # noqa: E402
from escalation import hard_rule_check  # noqa: E402

NEGATIVE_WORDS = [
    "angry", "furious", "ridiculous", "unacceptable", "worst", "terrible",
    "awful", "disgusted", "disappointed", "frustrat", "still not", "again",
    "third time", "week later", "no response", "ignored", "useless",
]


def heuristic_label_intent(text: str) -> str:
    """Independent second labeling pass with slightly different, tighter
    keyword sets than src/intents.py, used only to build golden first-pass
    labels (never used by the system under test)."""
    t = text.lower()
    if re.search(r"\brefund|cancel|charged twice|double charg|chargeback\b", t):
        return "order_cancel_refund"
    if re.search(r"\bbroken|damaged|defect|doesn'?t work|wrong item|faulty|missing part\b", t):
        return "product_defect"
    if re.search(r"\breturn label|send back|exchange|drop ?off|pickup\b", t):
        return "returns_exchange"
    if re.search(r"\bpassword|log ?in|account (locked|hacked)|billing|prime membership|gift card\b", t):
        return "account_payment"
    if re.search(r"\bapp crash|app (bug|error)|fire ?tv|firestick|prime video|kindle|alexa|echo|website (down|error)\b", t):
        return "app_tech_issue"
    if re.search(r"\bdeliver|shipp|track|package|parcel|courier|arrived|where is my order\b", t):
        return "delivery_issue"
    return "general_or_feedback"


def heuristic_label_escalate(text: str, intent: str):
    hard_hit, reason = hard_rule_check(text)
    if hard_hit:
        return True, reason
    neg_hits = sum(1 for w in NEGATIVE_WORDS if w in text.lower())
    if neg_hits >= 2:
        return True, f"strong negative sentiment ({neg_hits} negative cue words)"
    if intent in ("order_cancel_refund", "account_payment"):
        return True, f"{intent} typically requires account/order access a bot shouldn't have"
    return False, "routine, low-risk request"


def stratified_sample(pairs: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    pairs = pairs.copy()
    pairs["stratum"] = pairs["customer_text"].apply(lambda t: classify_rule_based(t).intent)
    sizes = pairs["stratum"].value_counts()
    weights = np.sqrt(sizes)
    weights = weights / weights.sum()
    per_stratum_n = (weights * n).round().astype(int)
    # fix rounding so total == n as closely as possible
    diff = n - per_stratum_n.sum()
    if diff != 0:
        biggest = per_stratum_n.idxmax()
        per_stratum_n[biggest] += diff

    parts = []
    rng = np.random.RandomState(seed)
    for stratum, k in per_stratum_n.items():
        pool = pairs[pairs["stratum"] == stratum]
        k = min(k, len(pool))
        if k <= 0:
            continue
        parts.append(pool.sample(n=k, random_state=rng.randint(0, 1_000_000)))
    return pd.concat(parts).drop_duplicates("customer_tweet_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/processed/pairs.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="eval/golden_set.csv")
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs)
    sample = stratified_sample(pairs, args.n, args.seed)

    sample["gold_intent_firstpass"] = sample["customer_text"].apply(heuristic_label_intent)
    esc = sample["customer_text"].apply(
        lambda t: heuristic_label_escalate(t, heuristic_label_intent(t)))
    sample["gold_escalate_firstpass"] = esc.apply(lambda x: x[0])
    sample["gold_escalate_reason_firstpass"] = esc.apply(lambda x: x[1])

    # Reviewed columns start as a copy; a human should edit these directly.
    sample["gold_intent_reviewed"] = sample["gold_intent_firstpass"]
    sample["gold_escalate_reviewed"] = sample["gold_escalate_firstpass"]
    sample["reviewed_by_human"] = False
    sample["reviewer_notes"] = ""

    cols = ["customer_tweet_id", "customer_text", "agent_reply",
            "gold_intent_firstpass", "gold_escalate_firstpass", "gold_escalate_reason_firstpass",
            "gold_intent_reviewed", "gold_escalate_reviewed", "reviewed_by_human", "reviewer_notes"]
    sample = sample[cols]
    sample.to_csv(args.out, index=False)
    print(f"Wrote {len(sample)} golden examples to {args.out}")
    print(sample["gold_intent_firstpass"].value_counts())
    print("Escalate rate (first pass):", sample["gold_escalate_firstpass"].mean().round(3))


if __name__ == "__main__":
    main()
