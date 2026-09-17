"""
intents.py
----------
Defines the intent taxonomy for AmazonHelp customer messages (derived by
reading ~300 sampled customer tweets by hand before writing any rules —
see decision_log.md item #1) and a keyword/regex classifier that acts as:
  (a) the "simple baseline" required by the assignment, and
  (b) a fast, free, offline pre-labeler used to seed the golden set and as a
      fallback when no LLM API key is configured (see llm_client.py).

Taxonomy (7 intents). We deliberately kept this small: Banking77-style
77-way taxonomies are built for banking queries with narrow, well-separated
topics; retail support chatter is dominated by a handful of high-frequency
themes, and a coarse taxonomy is what the downstream reply-retrieval and
escalation logic actually need. See report.md "Problem framing".
"""
import re
from dataclasses import dataclass
from typing import List, Tuple

INTENTS = [
    "delivery_issue",       # late / missing / wrong package, tracking questions
    "order_cancel_refund",  # cancellations, refunds, chargebacks
    "product_defect",       # item broken, damaged, doesn't work, wrong item received
    "returns_exchange",     # how to return / exchange, return label, pickup
    "account_payment",      # login, password, payment method, billing, Prime membership
    "app_tech_issue",       # app/website bugs, can't stream (Fire TV, Prime Video), errors
    "general_or_feedback",  # praise, general questions, feedback, anything else
]

_RULES: List[Tuple[str, re.Pattern]] = [
    ("delivery_issue", re.compile(
        r"\b(deliver(y|ed|ing)?|shipp(ed|ing)|track(ing)?|package|parcel|courier|"
        r"hasn'?t arrived|never arrived|still (haven'?t|has ?not) (got|received)|"
        r"where is my (order|package)|late|delayed|lost (my )?(package|order))\b", re.I)),
    ("order_cancel_refund", re.compile(
        r"\b(refund(ed|s)?|cancel(led|ling|lation)?|chargeback|money back|"
        r"reimburse(d|ment)?|charged (me )?twice|double charged)\b", re.I)),
    ("product_defect", re.compile(
        r"\b(broken|damage[d]?|defect(ive)?|doesn'?t work|not working|stopped working|"
        r"wrong item|missing (part|item)s?|faulty|dead on arrival|dead pixel|cracked)\b", re.I)),
    ("returns_exchange", re.compile(
        r"\b(return(ed|ing|s)?( label)?|exchange(d|ing)?|send (it|this) back|"
        r"pickup|drop ?off)\b", re.I)),
    ("account_payment", re.compile(
        r"\b(log ?in|log ?out|password|account (locked|hacked|suspended|access)|"
        r"payment (method|failed|declined)|billing|prime membership|subscription|"
        r"credit card|gift card( balance)?|two[- ]factor)\b", re.I)),
    ("app_tech_issue", re.compile(
        r"\b(app (crash|crashing|bug|error)|website (down|error|not loading)|"
        r"fire ?tv|firestick|prime video|kindle|alexa|echo|error (code|message)|"
        r"won'?t (load|open|play)|glitch)\b", re.I)),
]


@dataclass
class IntentResult:
    intent: str
    matched_rule: str
    confidence: float  # 1.0 for rule hit, 0.3 default fallback


def classify_rule_based(text: str) -> IntentResult:
    """First rule that matches wins; ties broken by list order above, which is
    ordered from (in our reading of the data) most to least frequent/urgent.
    Falls back to general_or_feedback."""
    for intent, pattern in _RULES:
        if pattern.search(text):
            return IntentResult(intent=intent, matched_rule=pattern.pattern[:40] + "...", confidence=1.0)
    return IntentResult(intent="general_or_feedback", matched_rule="<no rule matched>", confidence=0.3)


LLM_INTENT_SYSTEM_PROMPT = f"""You are an intent classifier for Amazon customer-support
Twitter messages. Classify the customer's message into exactly one of these intents:
{", ".join(INTENTS)}.
Definitions:
- delivery_issue: late, missing, lost, or mis-tracked shipments.
- order_cancel_refund: wants a cancellation, refund, or was charged incorrectly.
- product_defect: item arrived broken/wrong/faulty.
- returns_exchange: wants to return or exchange an item they already have.
- account_payment: login, password, billing, membership/subscription, payment method issues.
- app_tech_issue: bug/crash/error in the Amazon app, website, or a device (Fire TV, Kindle, Alexa).
- general_or_feedback: anything else, including praise, general questions, or ambiguous messages.
Respond with ONLY the intent label, nothing else."""


def classify_llm(text: str, client) -> IntentResult:
    """client: an object exposing .complete(system, user) -> str (see llm_client.py)."""
    raw = client.complete(LLM_INTENT_SYSTEM_PROMPT, text).strip().lower()
    for intent in INTENTS:
        if intent in raw:
            return IntentResult(intent=intent, matched_rule="llm", confidence=0.9)
    # LLM returned something unparseable -> fall back to rules rather than guessing
    fallback = classify_rule_based(text)
    return IntentResult(intent=fallback.intent, matched_rule="llm_unparsed->rule_fallback", confidence=0.3)
