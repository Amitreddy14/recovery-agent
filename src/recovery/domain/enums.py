"""Controlled vocabularies for the recovery domain.

Failure taxonomy mirrors Razorpay's error contract (source / step / reason)
so that generated cases and live API errors are describable by one schema.
"""

from __future__ import annotations

from enum import StrEnum


class PaymentMethod(StrEnum):
    UPI = "upi"
    CARD = "card"
    NETBANKING = "netbanking"
    WALLET = "wallet"
    EMANDATE = "emandate"


class ErrorSource(StrEnum):
    """Who originated the failure. Razorpay exposes this on every error."""

    CUSTOMER = "customer"
    BUSINESS = "business"
    GATEWAY = "gateway"
    BANK = "bank"
    NETWORK = "network"


class PaymentStep(StrEnum):
    """Where in the flow the payment died."""

    PAYMENT_INITIATION = "payment_initiation"
    PAYMENT_AUTHENTICATION = "payment_authentication"
    PAYMENT_AUTHORIZATION = "payment_authorization"
    PAYMENT_CAPTURE = "payment_capture"


class FailureReason(StrEnum):
    """Failure reasons we model.

    Deliberately a *subset* of Razorpay's full list: these are the reasons
    that differ in recoverability. Reasons with identical treatment are
    collapsed, and anything unmodelled maps to OTHER rather than being
    silently misclassified.
    """

    # --- Technical declines: recoverable by timing, no customer contact ---
    BANK_DOWNTIME = "bank_downtime"
    GATEWAY_TECHNICAL_ERROR = "gateway_technical_error"
    NETWORK_ERROR = "network_error"
    PAYMENT_TIMEOUT = "payment_timeout"

    # --- Business declines: recoverable, needs customer state to change ---
    INSUFFICIENT_FUNDS = "insufficient_funds"
    PAYMENT_LIMIT_EXCEEDED = "payment_limit_exceeded"
    INVALID_OTP = "invalid_otp"
    CARD_EXPIRED = "card_expired"

    # --- Intent signals: retrying is usually value-destroying ---
    PAYMENT_CANCELLED_BY_USER = "payment_cancelled_by_user"

    # --- Mandate-specific ---
    MANDATE_REVOKED = "mandate_revoked"
    MANDATE_EXPIRED = "mandate_expired"
    AFA_REQUIRED = "afa_required"

    OTHER = "other"


class DeclineClass(StrEnum):
    """NPCI's split. Published per-bank monthly, so our generator can be
    calibrated against real rates rather than invented ones."""

    TECHNICAL = "technical"  # bank / NPCI infrastructure. NPCI target <1%
    BUSINESS = "business"  # user side. NPCI target <5% (Circular OC-149)


class MandateCategory(StrEnum):
    """Drives the AFA-free ceiling under the RBI E-mandate Framework, 2026.

    INSURANCE_PREMIUM, MUTUAL_FUND and CREDIT_CARD_BILL carry a Rs 1,00,000
    ceiling; everything else is capped at Rs 15,000.
    """

    INSURANCE_PREMIUM = "insurance_premium"
    MUTUAL_FUND = "mutual_fund"
    CREDIT_CARD_BILL = "credit_card_bill"
    SUBSCRIPTION = "subscription"
    UTILITY = "utility"
    LOAN_EMI = "loan_emi"
    OTHER = "other"


class CaseType(StrEnum):
    PAYMENT_FAILURE = "payment_failure"
    MANDATE_FAILURE = "mandate_failure"
    UPCOMING_AT_RISK = "upcoming_at_risk"  # pre-failure, prevention path


class CaseState(StrEnum):
    """Explicit states of the recovery state machine.

    Every transition is logged. There are no implicit states — this is the
    reason we did not adopt an agent framework.
    """

    INGESTED = "ingested"
    DIAGNOSED = "diagnosed"
    SCORED = "scored"
    DECIDED = "decided"
    GATE_PASSED = "gate_passed"
    GATE_BLOCKED = "gate_blocked"
    SCHEDULED = "scheduled"
    EXECUTING = "executing"
    RECOVERED = "recovered"
    ABANDONED = "abandoned"
    ESCALATED = "escalated"
    FAILED = "failed"


class ActionType(StrEnum):
    """The bounded action set.

    Closed by design. The agent may never invent an action; anything outside
    this set routes to ESCALATE_HUMAN.
    """

    NO_ACTION = "no_action"
    RETRY_NOW = "retry_now"
    RETRY_SCHEDULED = "retry_scheduled"
    RETRY_ALTERNATE_RAIL = "retry_alternate_rail"
    SEND_PAYMENT_LINK = "send_payment_link"
    PRE_DEBIT_NUDGE = "pre_debit_nudge"
    ESCALATE_HUMAN = "escalate_human"


class Channel(StrEnum):
    NONE = "none"  # silent actions (retries) consume no contact budget
    SMS = "sms"
    EMAIL = "email"
    WHATSAPP = "whatsapp"


class GateVerdict(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"
    DEFER = "defer"  # allowed, but not yet — e.g. outside contact hours
