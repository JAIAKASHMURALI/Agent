"""
simple_baseline.py
--------------------
The "simple baseline" required by the assignment. No LLM calls at all:
- Intent: keyword/regex rules (src/intents.py::classify_rule_based).
- Reply: copy the agent_reply from the single most similar historical case
  verbatim (nearest-neighbor retrieval, no generation/adaptation).
- Escalation: keyword rules only (src/escalation.py::hard_rule_check), plus
  "escalate if no similar historical case found at all".

This isolates how much value the LLM step (generation + judgment) adds over
pure retrieval + rules, which is exactly the comparison report.md needs.
"""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from intents import classify_rule_based  # noqa: E402
from escalation import hard_rule_check  # noqa: E402
from retrieval import HistoricalCaseIndex  # noqa: E402


def predict(golden_df: pd.DataFrame, index: HistoricalCaseIndex) -> pd.DataFrame:
    rows = []
    for _, row in golden_df.iterrows():
        text = row["customer_text"]
        intent = classify_rule_based(text).intent
        similar = index.query(text, k=1, exclude_id=row.get("customer_tweet_id"))
        reply = similar[0]["agent_reply"] if similar else (
            "Hi, sorry for the trouble! Please DM us your order details so we can help. ^AB")
        hard_hit, reason = hard_rule_check(text)
        escalate = hard_hit or (len(similar) == 0)
        rows.append({
            **row.to_dict(),
            "pred_intent": intent,
            "pred_reply": reply,
            "pred_escalate": escalate,
            "pred_escalate_reason": reason or ("no similar historical case found" if len(similar) == 0 else "n/a"),
        })
    return pd.DataFrame(rows)
