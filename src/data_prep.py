"""
data_prep.py
------------
Loads the raw AmazonHelp Twitter-support CSV (a brand-filtered slice of the
Kaggle "Customer Support on Twitter" dataset), reconstructs conversation
threads, and produces a clean table of (customer_message -> AmazonHelp reply)
pairs that we treat as "historically resolved" grounding examples.

Why this shape:
- The raw data is a flat table of tweets with `response_tweet_id` /
  `in_response_to_tweet_id` pointers. A single customer message can have
  multiple downstream replies; we only want the *first* AmazonHelp reply
  that directly answered a given inbound customer tweet, because that is
  the cleanest "problem -> resolution move" signal.
- We restrict to English text with a cheap heuristic (ratio of ASCII
  alphabetic characters) because our intent taxonomy, judge rubric, and
  reply drafting are all designed/validated on English. This is called out
  explicitly in the report's "what's misleading about my headline number"
  section: the agent has NOT been validated on the ~15% non-English traffic.
"""
import re
import argparse
import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 13  # deterministic langdetect output

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")


def clean_text(t: str) -> str:
    t = URL_RE.sub("", str(t))
    t = MENTION_RE.sub("", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def is_probably_english(t: str) -> bool:
    """Use langdetect (offline, no model download) rather than an ASCII-ratio
    heuristic: this dataset has real French/Spanish/Portuguese/Tagalog traffic
    that is still >55% ASCII letters and would slip past a naive character
    filter (see decision_log.md item #4)."""
    if len(t.split()) < 2:
        return False
    try:
        return detect(t) == "en"
    except LangDetectException:
        return False


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"tweet_id": str, "in_response_to_tweet_id": str,
                                   "response_tweet_id": str})
    df["inbound"] = df["inbound"].astype(str).str.upper() == "TRUE"
    return df


def build_pairs(df: pd.DataFrame, brand: str = "AmazonHelp") -> pd.DataFrame:
    """Return one row per (customer tweet -> first brand reply)."""
    by_id = df.set_index("tweet_id", drop=False)

    customer_msgs = df[df["inbound"] & df["in_response_to_tweet_id"].notna()].copy()
    # Also include the first tweet of a thread (a customer tweeting *at* the brand
    # with no in_response_to, e.g. a fresh complaint)
    customer_root = df[(df["inbound"]) & (df["text"].str.contains("@" + brand, na=False))].copy()
    customers = pd.concat([customer_msgs, customer_root]).drop_duplicates("tweet_id")

    rows = []
    for _, crow in customers.iterrows():
        cid = crow["tweet_id"]
        resp_ids = str(crow.get("response_tweet_id", "")) if pd.notna(crow.get("response_tweet_id")) else ""
        candidate_ids = [r.strip() for r in resp_ids.split(",") if r.strip() and r.strip() != "nan"]
        agent_reply = None
        for rid in candidate_ids:
            if rid in by_id.index:
                cand = by_id.loc[rid]
                if isinstance(cand, pd.DataFrame):
                    cand = cand.iloc[0]
                if (not cand["inbound"]) and cand["author_id"] == brand:
                    agent_reply = cand
                    break
        if agent_reply is None:
            continue
        cust_clean = clean_text(crow["text"])
        reply_clean = clean_text(agent_reply["text"])
        if not cust_clean or not reply_clean:
            continue
        if not (is_probably_english(cust_clean) and is_probably_english(reply_clean)):
            continue
        rows.append({
            "customer_tweet_id": cid,
            "customer_text": cust_clean,
            "agent_tweet_id": agent_reply["tweet_id"],
            "agent_reply": reply_clean,
            "created_at": crow["created_at"],
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/AmazonHelp_sample.csv")
    ap.add_argument("--brand", default="AmazonHelp")
    ap.add_argument("--out", default="data/processed/pairs.csv")
    ap.add_argument("--sample", type=int, default=0,
                     help="If >0, randomly subsample the raw table before processing "
                          "(speeds up local runs; the grading rubric expects a subsample).")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    df = load_raw(args.raw)
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed)
    pairs = build_pairs(df, brand=args.brand)
    pairs.to_csv(args.out, index=False)
    print(f"Loaded {len(df)} raw rows -> {len(pairs)} clean English (customer, agent_reply) pairs")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
