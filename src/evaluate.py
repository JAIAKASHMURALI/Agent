"""
evaluate.py
-----------
Runs the full system AND both baselines on the golden set, computes:
- Intent accuracy / macro-F1 vs gold_intent_reviewed
- Escalation precision/recall/F1 vs gold_escalate_reviewed
- Reply quality via LLM-judge (judge.py) -- overall_score distribution,
  % grounded, % correct, % polite, % actionable
Writes outputs/eval_results.csv (per-example) and outputs/metrics.json (summary).
"""
import argparse
import json
import sys
import os
import time
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baselines"))

from intents import classify_rule_based, classify_llm  # noqa: E402
from retrieval import HistoricalCaseIndex  # noqa: E402
from reply_generator import generate_reply  # noqa: E402
from escalation import decide_escalation  # noqa: E402
from llm_client import get_client  # noqa: E402
from judge import judge_reply  # noqa: E402
import trivial_baseline  # noqa: E402
import simple_baseline  # noqa: E402


def run_system(golden: pd.DataFrame, index: HistoricalCaseIndex, client, k=3):
    rows = []
    t0 = time.time()
    for i, row in golden.iterrows():
        text = row["customer_text"]
        intent_result = classify_llm(text, client)
        similar = index.query(text, k=k, exclude_id=row.get("customer_tweet_id"))
        reply = generate_reply(text, similar, client)
        esc = decide_escalation(text, intent_result.intent, similar, client)
        j = judge_reply(text, reply, similar, client)
        rows.append({
            **row.to_dict(),
            "pred_intent": intent_result.intent,
            "draft_reply": reply,
            "pred_escalate": esc["escalate"],
            "pred_escalate_reason": esc["reason"],
            "judge_grounded": j.get("grounded"),
            "judge_correct": j.get("correct"),
            "judge_polite": j.get("polite"),
            "judge_actionable": j.get("actionable"),
            "judge_overall_score": j.get("overall_score"),
        })
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(golden)} ({time.time()-t0:.0f}s elapsed)")
    return pd.DataFrame(rows)


def compute_intent_metrics(df: pd.DataFrame, pred_col: str, gold_col: str = "gold_intent_reviewed"):
    return {
        "accuracy": round(accuracy_score(df[gold_col], df[pred_col]), 4),
        "macro_f1": round(f1_score(df[gold_col], df[pred_col], average="macro", zero_division=0), 4),
    }


def compute_escalation_metrics(df: pd.DataFrame, pred_col: str, gold_col: str = "gold_escalate_reviewed"):
    y_true = df[gold_col].astype(bool)
    y_pred = df[pred_col].astype(bool)
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "escalation_rate_pred": round(y_pred.mean(), 4),
        "escalation_rate_gold": round(y_true.mean(), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/processed/pairs.csv")
    ap.add_argument("--golden", default="eval/golden_set.csv")
    ap.add_argument("--llm", choices=["mock", "claude"], default="mock")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out_results", default="outputs/eval_results.csv")
    ap.add_argument("--out_metrics", default="outputs/metrics.json")
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs)
    golden = pd.read_csv(args.golden)
    index = HistoricalCaseIndex(pairs)
    client = get_client(args.llm)

    print(f"Running SYSTEM ({args.llm}) on {len(golden)} golden examples...")
    sys_results = run_system(golden, index, client, k=args.k)
    sys_results.to_csv(args.out_results, index=False)

    print("Running TRIVIAL baseline...")
    majority_intent = trivial_baseline.fit_majority_intent(golden)
    trivial_results = trivial_baseline.predict(golden, majority_intent)

    print("Running SIMPLE baseline...")
    simple_results = simple_baseline.predict(golden, index)

    metrics = {
        "n_examples": len(golden),
        "llm_mode": args.llm,
        "system": {
            "intent": compute_intent_metrics(sys_results, "pred_intent"),
            "escalation": compute_escalation_metrics(sys_results, "pred_escalate"),
            "reply_quality": {
                "mean_overall_score": round(pd.to_numeric(sys_results["judge_overall_score"], errors="coerce").mean(), 3),
                "pct_grounded": round(sys_results["judge_grounded"].astype(bool).mean(), 3),
                "pct_correct": round(sys_results["judge_correct"].astype(bool).mean(), 3),
                "pct_polite": round(sys_results["judge_polite"].astype(bool).mean(), 3),
                "pct_actionable": round(sys_results["judge_actionable"].astype(bool).mean(), 3),
            },
        },
        "trivial_baseline": {
            "intent": compute_intent_metrics(trivial_results, "pred_intent"),
            "escalation": compute_escalation_metrics(trivial_results, "pred_escalate"),
            "majority_intent_used": majority_intent,
        },
        "simple_baseline": {
            "intent": compute_intent_metrics(simple_results, "pred_intent"),
            "escalation": compute_escalation_metrics(simple_results, "pred_escalate"),
        },
    }

    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
    with open(args.out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"\nWrote per-example results to {args.out_results}")
    print(f"Wrote summary metrics to {args.out_metrics}")


if __name__ == "__main__":
    main()
