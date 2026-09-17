"""
escalation.py
-------------
Decides whether a message should be auto-handled or escalated to a human,
with a stated reason (deliverable #3).

Design: hybrid, deliberately conservative.
1. Hard rules fire first for high-stakes / high-certainty triggers (legal
   threats, safety, fraud, self-harm mentions, explicit "human/manager"
   requests). These always escalate regardless of what an LLM says, because
   a false "auto-handle" here is much more costly than an unnecessary
   escalation. See decision_log.md item #6.
2. Otherwise we ask the LLM to weigh: message sentiment/urgency, whether the
   intent is one we have strong grounding for (>=1 similar historical case
   with similarity above a floor), and whether the drafted reply required
   any invented specifics. Low-grounding + negative sentiment -> escalate.
3. Every decision returns a machine-checkable reason string, never just a
   boolean, so a human reviewer (or our eval harness) can audit *why*.
"""
import json
import re

HARD_ESCALATE_PATTERNS = [
    (r"\b(lawyer|attorney|legal action|sue|lawsuit)\b", "legal threat mentioned"),
    (r"\b(fraud|scam|unauthorized charge|stolen (card|account))\b", "possible fraud"),
    (r"\b(suicide|kill myself|self ?harm)\b", "safety / self-harm signal"),
    (r"\b(speak (to|with) a (human|manager|person)|real person|actual human)\b",
     "explicit request for a human"),
    (r"\b(class action|bbb complaint|better business bureau|ftc)\b", "regulatory/legal escalation risk"),
]

ESCALATION_SYSTEM_PROMPT = """You decide whether an Amazon customer-support Twitter
message should be auto-handled by a bot or escalated to a human agent.
Consider: how negative/urgent the customer sounds, whether this looks like a routine,
well-understood request (delivery status, simple return question) vs. something needing
account access, money movement, or judgment calls (goodwill refunds, repeated failures).
Respond with ONLY a JSON object: {"escalate": true|false, "reason": "<one short sentence>"}"""


def hard_rule_check(text: str):
    for pattern, reason in HARD_ESCALATE_PATTERNS:
        if re.search(pattern, text, re.I):
            return True, reason
    return False, None


def decide_escalation(customer_text: str, intent: str, similar_cases: list,
                       client, similarity_floor: float = 0.15):
    hard_hit, hard_reason = hard_rule_check(customer_text)
    if hard_hit:
        return {"escalate": True, "reason": f"hard rule: {hard_reason}", "source": "rule"}

    grounded = any(c["similarity"] >= similarity_floor for c in similar_cases)
    if intent in ("order_cancel_refund", "account_payment") and not grounded:
        return {"escalate": True,
                "reason": "money/account-sensitive intent with no strongly similar historical "
                          "resolution to ground a safe auto-reply",
                "source": "rule"}

    prompt = (f"Customer message: {customer_text}\nDetected intent: {intent}\n"
              f"Has a grounded historical resolution (similarity>={similarity_floor}): {grounded}\n")
    raw = client.complete(ESCALATION_SYSTEM_PROMPT, prompt).strip()
    try:
        parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        return {"escalate": bool(parsed.get("escalate", True)),
                "reason": parsed.get("reason", "llm_no_reason"),
                "source": "llm"}
    except Exception:
        # Fail safe: if we can't parse the LLM's decision, escalate rather than guess.
        return {"escalate": True, "reason": "could not parse LLM escalation output; failing safe",
                "source": "fallback"}
