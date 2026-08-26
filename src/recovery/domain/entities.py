"""Core entities.

Money is represented in **paise** (integer) throughout. Floating-point rupees
are never used for arithmetic anywhere in this codebase.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from recovery.domain.enums import (
    CaseState,
    CaseType,
    DeclineClass,
    ErrorSource,
    FailureReason,
    MandateCategory,
    PaymentMethod,
    PaymentStep,
)

Paise = int


class Frozen(BaseModel):
    """Immutable base. Domain objects are never mutated in place; state
    changes produce new records so the ledger stays append-only."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Merchant(Frozen):
    merchant_id: str
    name: str
    vertical: str
    supported_methods: tuple[PaymentMethod, ...]


class Customer(Frozen):
    customer_id: str
    # Observable features the policy is allowed to use.
    tenure_days: int = Field(ge=0)
    prior_payment_count: int = Field(ge=0)
    prior_failure_count: int = Field(ge=0)
    prior_recovery_count: int = Field(ge=0)
    issuer: str
    preferred_method: PaymentMethod
    dnd_registered: bool = False
    contacts_last_7d: int = Field(default=0, ge=0)
    contacts_last_30d: int = Field(default=0, ge=0)

    @property
    def prior_failure_rate(self) -> float:
        total = self.prior_payment_count + self.prior_failure_count
        return self.prior_failure_count / total if total else 0.0


class PaymentError(Frozen):
    """Mirrors Razorpay's error object shape."""

    source: ErrorSource
    step: PaymentStep
    reason: FailureReason
    decline_class: DeclineClass
    description: str = ""


class PaymentAttempt(Frozen):
    attempt_id: str
    order_id: str
    customer_id: str
    merchant_id: str
    amount_paise: Paise = Field(gt=0)
    method: PaymentMethod
    issuer: str
    attempted_at: datetime
    succeeded: bool
    error: PaymentError | None = None

    @field_validator("error")
    @classmethod
    def _error_iff_failed(cls, v: PaymentError | None, info: object) -> PaymentError | None:
        return v


class Mandate(Frozen):
    """A registered e-mandate under the RBI framework."""

    mandate_id: str
    customer_id: str
    merchant_id: str
    category: MandateCategory
    max_amount_paise: Paise = Field(gt=0)
    debit_amount_paise: Paise = Field(gt=0)
    registered_at: datetime
    valid_until: datetime
    next_debit_at: datetime
    active: bool = True
    consecutive_failures: int = Field(default=0, ge=0)

    @property
    def afa_free_ceiling_paise(self) -> Paise:
        """AFA-free ceiling per RBI Digital Payments E-mandate Framework, 2026.

        Rs 1,00,000 for insurance premiums, mutual fund subscriptions and
        credit card bill payments; Rs 15,000 for everything else.
        """
        elevated = {
            MandateCategory.INSURANCE_PREMIUM,
            MandateCategory.MUTUAL_FUND,
            MandateCategory.CREDIT_CARD_BILL,
        }
        return 100_000_00 if self.category in elevated else 15_000_00

    @property
    def requires_afa(self) -> bool:
        return self.debit_amount_paise > self.afa_free_ceiling_paise


class RecoveryCase(Frozen):
    """The unit of work. One case = one at-risk rupee amount."""

    case_id: str
    case_type: CaseType
    merchant_id: str
    customer_id: str
    amount_at_risk_paise: Paise = Field(gt=0)
    created_at: datetime
    state: CaseState = CaseState.INGESTED

    attempt_id: str | None = None
    mandate_id: str | None = None

    # Case-level budgets. Enforced by the compliance engine, not by callers.
    max_attempts: int = Field(default=3, ge=0)
    max_contacts: int = Field(default=2, ge=0)
    attempts_used: int = Field(default=0, ge=0)
    contacts_used: int = Field(default=0, ge=0)

    # Hard-stop flags.
    customer_opted_out: bool = False
    dispute_raised: bool = False
    promise_to_pay_at: datetime | None = None

    @property
    def attempts_remaining(self) -> int:
        return max(0, self.max_attempts - self.attempts_used)

    @property
    def contacts_remaining(self) -> int:
        return max(0, self.max_contacts - self.contacts_used)

    @property
    def is_terminal(self) -> bool:
        return self.state in {
            CaseState.RECOVERED,
            CaseState.ABANDONED,
            CaseState.FAILED,
        }
