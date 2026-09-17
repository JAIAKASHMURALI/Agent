"""
judge.py
--------
LLM-as-judge rubric for reply quality (deliverable #3's "evaluation
harness" requirement). Scores a drafted reply against the customer message
and (optionally) the retrieved grounding cases on four binary/graded criteria,
plus a 1-5 overall score.

Rubric criteria (chosen to catch the failure modes we actually saw while
reading the data -- see decision_log.md):
- grounded:   does the reply follow the brand's actual established playbook
              moves (DM handoff, help-center link, apology + policy) rather
              than inventing a generic/off-brand response?
- correct:    does it avoid inventing specifics (order numbers, refund
              amounts, dates) not present in the customer's message?
- polite:     appropriate tone for a support agent (empathetic, not curt,
              not robotic-repetitive)?
- actionable: does the customer know what happens next / what to do next?
"""
import json

JUDGE_SYSTEM_PROMPT = """You are grading a customer-support reply against a rubric.
You will see the CUSTOMER MESSAGE, the DRAFT REPLY, and (if available) one or more
SIMILAR PAST CASES showing how this brand has actually handled similar issues before.

Score the DRAFT REPLY on these criteria:
- grounded (true/false): does it follow the brand's established playbook moves shown in
  the past cases (e.g., ask to DM, point to a help link, apologize + explain policy),
  rather than being generic or off-brand?
- correct (true/false): does it avoid inventing specific order numbers, refund amounts,
  tracking numbers, or dates that were not given by the customer?
- polite (true/false): is the tone empathetic and professional?
- actionable (true/false): does the customer clearly know what to do or expect next?
- overall_score (integer 1-5): overall reply quality, 5 = excellent, 1 = unusable.

Respond with ONLY a JSON object with keys: grounded, correct, polite, actionable, overall_score."""


def build_judge_prompt(customer_text: str, draft_reply: str, similar_cases: list) -> str:
    parts = [f"CUSTOMER MESSAGE:\n{customer_text}\n", f"DRAFT REPLY:\n{draft_reply}\n"]
    if similar_cases:
        parts.append("SIMILAR PAST CASES:")
        for c in similar_cases:
            parts.append(f"- Customer: {c['customer_text']}\n  Agent replied: {c['agent_reply']}")
    return "\n".join(parts)


def judge_reply(customer_text: str, draft_reply: str, similar_cases: list, client) -> dict:
    prompt = build_judge_prompt(customer_text, draft_reply, similar_cases)
    raw = client.complete(JUDGE_SYSTEM_PROMPT, prompt).strip()
    try:
        parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except Exception:
        parsed = {"grounded": False, "correct": False, "polite": False,
                   "actionable": False, "overall_score": 1, "parse_error": True}
    return parsed
