"""
judge_agreement.py
-------------------
Deliverable #3 requires "evidence of how well your judge agrees with a
human." This script:
1. Takes eval output (outputs/eval_results.csv, produced by src/evaluate.py),
   which already has the LLM judge's overall_score (1-5) per example.
2. Samples --n rows (default 30) for a human to additionally rate 1-5 by
   hand, writing them into outputs/human_ratings_template.csv.
3. Once the human column `human_overall_score` is filled in, run this script
   again with --score to compute agreement: Spearman correlation, exact
   match rate, and "within 1 point" rate, plus binary
   agreement on a good(>=4)/bad(<4) threshold with Cohen's kappa.

You (the assignment author) need to actually fill in human_overall_score by
reading the 30 sampled rows yourself -- that's the "evidence" the assignment
is asking for. We don't fabricate that column for you.
"""
import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def cohens_kappa(a, b):
    a = np.array(a)
    b = np.array(b)
    n = len(a)
    po = (a == b).mean()
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def make_template(results_path: str, out_path: str, n: int, seed: int):
    df = pd.read_csv(results_path)
    sample = df.sample(n=min(n, len(df)), random_state=seed).copy()
    sample["human_overall_score"] = ""  # to be filled in by hand, 1-5
    sample[["customer_text", "draft_reply", "judge_overall_score",
            "human_overall_score"]].to_csv(out_path, index=False)
    print(f"Wrote {len(sample)} rows to {out_path}. Fill in human_overall_score (1-5) by hand, "
          f"then rerun with --score.")


def score(template_path: str):
    df = pd.read_csv(template_path)
    df = df[df["human_overall_score"] != ""].copy()
    df["human_overall_score"] = df["human_overall_score"].astype(int)
    df["judge_overall_score"] = df["judge_overall_score"].astype(int)
    if len(df) < 5:
        print("Need at least 5 human-rated rows to compute agreement meaningfully.")
        return
    rho, p = spearmanr(df["judge_overall_score"], df["human_overall_score"])
    exact = (df["judge_overall_score"] == df["human_overall_score"]).mean()
    within1 = (df["judge_overall_score"] - df["human_overall_score"]).abs().le(1).mean()
    judge_good = (df["judge_overall_score"] >= 4).astype(int)
    human_good = (df["human_overall_score"] >= 4).astype(int)
    kappa = cohens_kappa(judge_good, human_good)
    print(f"n={len(df)}")
    print(f"Spearman rho: {rho:.3f} (p={p:.3f})")
    print(f"Exact match rate: {exact:.3f}")
    print(f"Within-1-point rate: {within1:.3f}")
    print(f"Cohen's kappa (good>=4 threshold): {kappa:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/eval_results.csv")
    ap.add_argument("--template", default="outputs/human_ratings_template.csv")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--score", action="store_true", help="Score an already-filled-in template")
    args = ap.parse_args()

    if args.score:
        score(args.template)
    else:
        make_template(args.results, args.template, args.n, args.seed)
