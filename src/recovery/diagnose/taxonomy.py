"""Failure taxonomy.

This layer is deliberately deterministic. A failure reason maps to a
recoverability class by rule, not by model, because the mapping is knowledge
we already have — Razorpay's error contract tells us what happened, and no
amount of training data would improve on "an expired card cannot be fixed by
retrying it."

Using a model here would be the wrong tool: it would be slower, unexplainable
in an audit trail, and less accurate than the rule. The learned component
belongs one layer up, where the question is genuinely uncertain — *is this
issuer degraded right now* — and in Phase 5, where the question is *would an
intervention change this outcome*.
"""

from __future__ import annotations

from enum import StrEnum

from recovery.domain.enums import DeclineClass, FailureReason


class Recoverability(StrEnum):
    """What kind of remedy, if any, this failure admits."""

    TRANSIENT = "transient"
    """Infrastructure fault. Resolves on its own or on a well-timed retry.
    Requires no customer contact, which makes it the cheapest money to
    recover."""

    FUNDING = "funding"
    """The customer cannot pay right now but likely could later. Timing is
    the whole problem: retry too early and the attempt is wasted."""

    INSTRUMENT = "instrument"
    """The payment instrument itself is unusable. Retrying the same rail
    cannot work; the customer must supply something else."""

    AUTHENTICATION = "authentication"
    """The customer failed a challenge. Partially recoverable, but repeated
    attempts risk locking the instrument."""

    INTENT = "intent"
    """The customer chose not to pay. Retrying is noise and contacting them
    is usually value-destroying. Distinguishing this class from FUNDING is
    the single most valuable judgement in the whole taxonomy."""

    TERMINAL = "terminal"
    """No action can recover this. A revoked mandate is revoked."""

    UNKNOWN = "unknown"
    """Unmapped reason. Routed to human review rather than guessed at."""


RECOVERABILITY: dict[FailureReason, Recoverability] = {
    FailureReason.BANK_DOWNTIME: Recoverability.TRANSIENT,
    FailureReason.GATEWAY_TECHNICAL_ERROR: Recoverability.TRANSIENT,
    FailureReason.NETWORK_ERROR: Recoverability.TRANSIENT,
    FailureReason.PAYMENT_TIMEOUT: Recoverability.TRANSIENT,
    FailureReason.INSUFFICIENT_FUNDS: Recoverability.FUNDING,
    FailureReason.PAYMENT_LIMIT_EXCEEDED: Recoverability.FUNDING,
    FailureReason.CARD_EXPIRED: Recoverability.INSTRUMENT,
    FailureReason.INVALID_OTP: Recoverability.AUTHENTICATION,
    FailureReason.AFA_REQUIRED: Recoverability.AUTHENTICATION,
    FailureReason.PAYMENT_CANCELLED_BY_USER: Recoverability.INTENT,
    FailureReason.MANDATE_REVOKED: Recoverability.TERMINAL,
    FailureReason.MANDATE_EXPIRED: Recoverability.TERMINAL,
    FailureReason.OTHER: Recoverability.UNKNOWN,
}

# Classes recoverable without spending a customer contact. This is the
# distinction that drives cost: a silent retry is ~50 paise, a message is ~35
# paise plus the risk of irritating a sleeping dog.
SILENT_CLASSES: frozenset[Recoverability] = frozenset(
    {Recoverability.TRANSIENT, Recoverability.FUNDING}
)

# Classes where no automated action has any prospect of working.
HOPELESS_CLASSES: frozenset[Recoverability] = frozenset({Recoverability.TERMINAL})


def classify(reason: FailureReason) -> Recoverability:
    return RECOVERABILITY.get(reason, Recoverability.UNKNOWN)


def is_silently_recoverable(reason: FailureReason) -> bool:
    return classify(reason) in SILENT_CLASSES


def is_terminal(reason: FailureReason) -> bool:
    return classify(reason) in HOPELESS_CLASSES


def expected_decline_class(reason: FailureReason) -> DeclineClass | None:
    """Which NPCI decline class this reason should have arrived under.

    Used as a consistency check: a reason that disagrees with its recorded
    decline class indicates upstream data corruption, and we would rather
    surface that than diagnose confidently from a contradictory record.
    """
    transient = {
        FailureReason.BANK_DOWNTIME,
        FailureReason.GATEWAY_TECHNICAL_ERROR,
        FailureReason.NETWORK_ERROR,
        FailureReason.PAYMENT_TIMEOUT,
    }
    business = {
        FailureReason.INSUFFICIENT_FUNDS,
        FailureReason.PAYMENT_LIMIT_EXCEEDED,
        FailureReason.INVALID_OTP,
        FailureReason.CARD_EXPIRED,
        FailureReason.PAYMENT_CANCELLED_BY_USER,
    }
    if reason in transient:
        return DeclineClass.TECHNICAL
    if reason in business:
        return DeclineClass.BUSINESS
    return None


