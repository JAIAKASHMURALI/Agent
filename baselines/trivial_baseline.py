"""
trivial_baseline.py
--------------------
The "trivial baseline" required by the assignment:
- Intent: always predict the majority class from the training pairs.
- Reply: always the same canned message, regardless of content.
- Escalation: never escalate.

This exists purely to give the real system's numbers a floor to beat, and to
make clear when a metric (like "62% intent accuracy") is not actually
impressive on a skewed label distribution (see report.md's misleading-number
section).
"""
import pandas as pd

CANNED_REPLY = ("Hi, thanks for reaching out! We're sorry for any inconvenience. "
                "Please send us a DM with your order details so we can help. ^AB")


def fit_majority_intent(golden_df: pd.DataFrame, intent_col: str = "gold_intent_reviewed") -> str:
    return golden_df[intent_col].value_counts().idxmax()


def predict(golden_df: pd.DataFrame, majority_intent: str) -> pd.DataFrame:
    out = golden_df.copy()
    out["pred_intent"] = majority_intent
    out["pred_reply"] = CANNED_REPLY
    out["pred_escalate"] = False
    return out
