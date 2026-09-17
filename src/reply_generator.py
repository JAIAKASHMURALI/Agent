"""
reply_generator.py
-------------------
Drafts a reply for an incoming customer message, grounded in the top-k most
similar historically-resolved cases (deliverable #2's "grounded" requirement).

Design choice: we pass the retrieved historical replies to the LLM as
context and explicitly instruct it to follow the brand's established *moves*
(ask for DM, point to a help link, apologize + explain policy) rather than
copy specific order numbers / personal details from the retrieved examples
verbatim. This is checked by the LLM-judge rubric's "grounded" and
"correct" criteria (no hallucinated order-specific facts).
"""

REPLY_SYSTEM_PROMPT = """You are an Amazon customer-support agent replying on Twitter as
@AmazonHelp. Write ONE short reply (max 280 characters) to the customer's message.

Rules:
- Match the tone, structure and typical moves (apologize, ask to DM, point to a help
  link, explain policy) used in the SIMILAR PAST CASE(S) shown below — that is how this
  brand has actually resolved this kind of issue.
- NEVER invent specific order numbers, refund amounts, tracking numbers, or dates that
  were not given to you by the customer.
- If the past cases ask the customer to move to DM for account/order details, do the same.
- Be empathetic but concise. Do not make promises the brand can't reliably keep
  (e.g. exact refund timing) unless the retrieved cases show that as standard language.
- Output ONLY the reply text, nothing else (no labels, no quotes)."""


def build_reply_prompt(customer_text: str, similar_cases: list) -> str:
    parts = [f"CUSTOMER MESSAGE:\n{customer_text}\n"]
    if similar_cases:
        parts.append("SIMILAR PAST CASE(S) THIS BRAND HANDLED:")
        for i, c in enumerate(similar_cases, 1):
            parts.append(
                f"[{i}] (similarity={c['similarity']:.2f}) Customer said: {c['customer_text']}\n"
                f"    Agent replied: {c['agent_reply']}\n"
            )
    else:
        parts.append("SIMILAR PAST CASE(S): none found above the similarity floor — "
                      "draft a generic, safe, on-brand reply instead.")
    parts.append("\nDRAFT REPLY:")
    return "\n".join(parts)


def generate_reply(customer_text: str, similar_cases: list, client) -> str:
    prompt = build_reply_prompt(customer_text, similar_cases)
    return client.complete(REPLY_SYSTEM_PROMPT, prompt).strip()
