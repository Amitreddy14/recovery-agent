"""Pre-debit notification.

The RBI E-mandate Framework 2026 requires notice at least 24 hours before a
recurring debit, stating merchant name, amount, debit date and time, mandate
reference, reason for the debit, and how to opt out.

This is one of the three places an LLM is used in this project, and the
design principle is the same in all three: **the model drafts, the system
verifies.** Copy is generated for readability; the required fields are then
checked against `configs/compliance/policy.yaml` by string containment, and a
draft missing any of them is rejected rather than sent.

That ordering matters. A fluent notification missing its opt-out instruction
is a compliance failure that reads perfectly well, which is exactly the class
of error a language model produces and a human reviewer skims past. The
validator does not care how the sentence reads.

A deterministic template is always available and is what runs when no API key
is configured. The LLM improves phrasing; it is never load-bearing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

from recovery.domain.entities import Mandate

# Field name in the policy YAML -> how it must appear in the rendered message.
FIELD_MARKERS: dict[str, str] = {
    "merchant_name": "merchant",
    "amount": "amount",
    "debit_datetime": "datetime",
    "mandate_reference": "reference",
    "debit_reason": "reason",
    "opt_out_instruction": "opt_out",
}


@dataclass(frozen=True)
class NotificationContent:
    """A drafted notification and the facts it must convey."""

    merchant_name: str
    amount_paise: int
    debit_at: datetime
    mandate_reference: str
    debit_reason: str
    opt_out_instruction: str
    body: str
    drafted_by: str

    @property
    def amount_rupees(self) -> str:
        return f"{self.amount_paise / 100:,.2f}"


class NotificationValidationError(ValueError):
    """Raised when a draft omits a legally required field."""


def render_template(
    *,
    merchant_name: str,
    mandate: Mandate,
    debit_at: datetime,
    debit_reason: str,
) -> NotificationContent:
    """Deterministic notification. Always compliant, never elegant."""
    reference = mandate.mandate_id
    opt_out = "Reply STOP to cancel this debit or manage the mandate in your app."
    amount = f"{mandate.debit_amount_paise / 100:,.2f}"
    body = (
        f"{merchant_name} will debit Rs {amount} from your account on "
        f"{debit_at.strftime('%d %b %Y at %H:%M')} against mandate "
        f"{reference}. Reason: {debit_reason}. {opt_out}"
    )
    return NotificationContent(
        merchant_name=merchant_name,
        amount_paise=mandate.debit_amount_paise,
        debit_at=debit_at,
        mandate_reference=reference,
        debit_reason=debit_reason,
        opt_out_instruction=opt_out,
        body=body,
        drafted_by="template",
    )


def validate(content: NotificationContent, required_fields: list[str]) -> list[str]:
    """Return the required fields absent from the rendered body.

    Checks the *rendered text*, not the structured fields. A notification
    whose data is correct but whose body omits the amount is still
    non-compliant, because the body is what the customer receives.
    """
    body = content.body.lower()
    checks: dict[str, bool] = {
        "merchant_name": content.merchant_name.lower() in body,
        "amount": content.amount_rupees.lower() in body,
        "debit_datetime": content.debit_at.strftime("%d %b %Y").lower() in body,
        "mandate_reference": content.mandate_reference.lower() in body,
        "debit_reason": content.debit_reason.lower() in body,
        "opt_out_instruction": _mentions_opt_out(body),
    }
    missing: list[str] = []
    for field in required_fields:
        # Fail closed on an unrecognised field. A requirement the validator
        # does not know how to check is unenforced, and an unenforced
        # requirement is worse than an absent one because it looks satisfied
        # (INC-021).
        if field not in checks or not checks[field]:
            missing.append(field)
    return missing


def _mentions_opt_out(body: str) -> bool:
    return any(token in body for token in ("stop", "opt out", "opt-out", "cancel"))


def draft_with_llm(
    base: NotificationContent,
    required_fields: list[str],
    *,
    model: str = "claude-sonnet-4-6",
) -> NotificationContent:
    """Ask a model to improve the phrasing, then verify it kept the facts.

    Falls back to the template on any failure — no key, API error, or a draft
    that drops a required field. The fallback is silent by design: a
    compliant plain message is strictly better than an elegant
    non-compliant one, and there is no scenario where waiting for a nicer
    draft is worth delaying a mandated notification.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return base

    try:
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=400,
            system=(
                "You rewrite regulatory payment notifications for clarity. You "
                "must preserve every fact exactly: merchant name, amount, date "
                "and time, mandate reference, reason, and the opt-out "
                "instruction. Do not add facts. Do not remove any. Reply with "
                "the message text only, under 320 characters."
            ),
            messages=[{"role": "user", "content": base.body}],
        )
        text = "".join(block.text for block in response.content if block.type == "text").strip()
    except Exception:
        return base

    if not text:
        return base

    candidate = NotificationContent(
        merchant_name=base.merchant_name,
        amount_paise=base.amount_paise,
        debit_at=base.debit_at,
        mandate_reference=base.mandate_reference,
        debit_reason=base.debit_reason,
        opt_out_instruction=base.opt_out_instruction,
        body=text,
        drafted_by=f"llm:{model}",
    )
    if validate(candidate, required_fields):
        # The draft lost a required field. Discard it — this is the whole
        # reason the validator runs after generation rather than trusting the
        # instruction in the system prompt.
        return base
    return candidate


def compose(
    *,
    merchant_name: str,
    mandate: Mandate,
    debit_at: datetime,
    debit_reason: str,
    required_fields: list[str],
    use_llm: bool = False,
) -> NotificationContent:
    """Produce a validated notification, or raise.

    The template is validated too. A required field added to the policy YAML
    that the template does not render should break loudly here rather than
    ship silently non-compliant messages.
    """
    base = render_template(
        merchant_name=merchant_name,
        mandate=mandate,
        debit_at=debit_at,
        debit_reason=debit_reason,
    )
    missing = validate(base, required_fields)
    if missing:
        raise NotificationValidationError(
            f"template omits required fields {missing}; "
            "policy.yaml and render_template have diverged"
        )
    return draft_with_llm(base, required_fields) if use_llm else base
