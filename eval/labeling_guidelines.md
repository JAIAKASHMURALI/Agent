# Golden Set Labeling Guidelines

The golden set (`eval/golden_set.csv`) has `_firstpass` columns filled in
automatically by `build_golden_set.py`, and `_reviewed` columns that a human
should actually look at and correct. This file is the rubric a reviewer
follows while doing that.

## 1. Intent (`gold_intent_reviewed`)

Read the customer's tweet (with @mentions/URLs stripped) and pick exactly one:

| Intent | Pick this when... | Not this when... |
|---|---|---|
| `delivery_issue` | Core complaint is about the shipment itself: late, lost, mis-tracked, wrong address. | The package arrived but the **item inside** is wrong/broken -> `product_defect`. |
| `order_cancel_refund` | Customer wants money back or an order stopped, including double-charges. | Customer wants to send an item back for a refund -> usually `returns_exchange` unless they explicitly only mention money. |
| `product_defect` | Item received is broken, wrong, faulty, missing parts. | Customer just wants to send it back with no complaint about condition -> `returns_exchange`. |
| `returns_exchange` | Process question: how/where to return, return label, pickup, exchange for a different size/color. | |
| `account_payment` | Login, password, billing, membership, payment method, gift card balance. | |
| `app_tech_issue` | Bug/crash/error in app, website, Fire TV, Kindle, Alexa/Echo. | |
| `general_or_feedback` | Praise, general/ambiguous question, anything not clearly one of the above. | This is the "catch-all" — use sparingly; if two labels are plausible, prefer the more specific one. |

If a tweet plausibly fits two intents, pick the one that determines **what the
agent needs to do next** (that's what the reply/escalation logic cares about).

## 2. Escalation (`gold_escalate_reviewed`, boolean)

Ask: *"Could a well-templated bot reply safely resolve this without account
access, moving money, or human judgment about an exception to policy?"*

Escalate (`True`) if any of:
- Legal threats, fraud/safety mentions, explicit request for a human.
- Requires refund/cancellation decision, or account access (login/billing).
- Customer signals this is a repeat/unresolved contact ("this is the third
  time...", "still not fixed", "no one replied").
- Message is angry/abusive enough that a templated reply would likely make
  things worse.

Auto-handle (`False`) if:
- Simple status/how-to question with a standard, safe answer (tracking info,
  "how do I return X", generic troubleshooting step).
- Positive/neutral feedback needing only an acknowledgement.

When unsure, escalate — false "auto-handle" is the more expensive mistake for
a real brand (see `report.md`, "Problem framing").

## 3. Mechanics

- Set `reviewed_by_human` to `True` once you've actually read the row.
- Use `reviewer_notes` for anything ambiguous — these notes feed directly into
  `report.md`'s failure analysis section.
- Don't relabel more than ~10% of rows purely on vibes; if you disagree with
  more than that, the first-pass heuristics (or the taxonomy itself) probably
  need fixing, not just the individual labels — flag that in decision_log.md.
