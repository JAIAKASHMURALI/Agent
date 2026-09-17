"""
pipeline.py
-----------
End-to-end AI support agent: intent -> retrieval -> grounded reply -> escalation.
Runnable standalone (prints one result) or imported by evaluate.py.
"""
import argparse
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from intents import classify_rule_based, classify_llm  # noqa: E402
from retrieval import HistoricalCaseIndex  # noqa: E402
from reply_generator import generate_reply  # noqa: E402
from escalation import decide_escalation  # noqa: E402
from llm_client import get_client  # noqa: E402


def run_one(text: str, index: HistoricalCaseIndex, client, exclude_id=None, k=3):
    intent_result = classify_llm(text, client)
    similar = index.query(text, k=k, exclude_id=exclude_id)
    reply = generate_reply(text, similar, client)
    esc = decide_escalation(text, intent_result.intent, similar, client)
    return {
        "customer_text": text,
        "intent": intent_result.intent,
        "intent_source": intent_result.matched_rule,
        "top_similarity": similar[0]["similarity"] if similar else 0.0,
        "n_similar_cases": len(similar),
        "draft_reply": reply,
        "escalate": esc["escalate"],
        "escalate_reason": esc["reason"],
        "escalate_source": esc["source"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/processed/pairs.csv")
    ap.add_argument("--llm", choices=["mock", "claude","gemini","groq"], default="mock")
    ap.add_argument("--text", default=None, help="Single ad-hoc message to run through the pipeline")
    ap.add_argument("--n_demo", type=int, default=5, help="If --text not given, demo on N random pairs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs)
    index = HistoricalCaseIndex(pairs)
    client = get_client(args.llm)

    if args.text:
        result = run_one(args.text, index, client)
        for k, v in result.items():
            print(f"{k}: {v}")
        return

    sample = pairs.sample(n=min(args.n_demo, len(pairs)), random_state=args.seed)
    for _, row in sample.iterrows():
        result = run_one(row["customer_text"], index, client, exclude_id=row["customer_tweet_id"])
        print("=" * 80)
        print("CUSTOMER:", result["customer_text"])
        print("INTENT:", result["intent"], f"(via {result['intent_source']})")
        print("DRAFT REPLY:", result["draft_reply"])
        print("ESCALATE:", result["escalate"], "-", result["escalate_reason"])
        print("(actual historical agent reply was:", row["agent_reply"], ")")


if __name__ == "__main__":
    main()
