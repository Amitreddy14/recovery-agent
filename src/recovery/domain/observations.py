"""The observable contract between the world and the policy.

`CaseFeatures` is everything a policy may see about a case. `LoggedDecision`
is what the historical policy did and with what probability.

These live in `domain` rather than `world` deliberately. They are the
interface, not part of the simulator, and putting them here lets CI forbid
decision-making code from importing `recovery.world` *at all* - including
`world.timeline`, whose `is_degraded()` is precisely the ground truth the
diagnosis layer is supposed to infer (ADR-0011).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from recovery.domain.enums import (
    ActionType,
    CaseType,
    DeclineClass,
    FailureReason,
    MandateCategory,
    PaymentMethod,
)


class CaseFeatures(BaseModel):
    """Everything the policy may observe. No latent traits, no outcomes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    case_type: CaseType
    created_at: datetime

    amount_paise: int = Field(gt=0)
    method: PaymentMethod
    issuer: str
    reason: FailureReason
    decline_class: DeclineClass

    customer_id: str
    tenure_days: int = Field(ge=0)
    prior_payment_count: int = Field(ge=0)
    prior_failure_count: int = Field(ge=0)
    prior_recovery_count: int = Field(ge=0)
    contacts_last_30d: int = Field(ge=0)
    dnd_registered: bool

    hour_of_day: int = Field(ge=0, le=23)
    day_of_month: int = Field(ge=1, le=31)
    days_since_salary: int = Field(ge=0)

    mandate_category: MandateCategory | None = None
    consecutive_mandate_failures: int = Field(default=0, ge=0)

    issuer_failures_last_hour: int = Field(default=0, ge=0)
    issuer_volume_last_hour: int = Field(default=0, ge=0)

    @property
    def prior_failure_rate(self) -> float:
        total = self.prior_payment_count + self.prior_failure_count
        return self.prior_failure_count / total if total else 0.0

    @property
    def issuer_failure_rate_last_hour(self) -> float:
        """The observable proxy for issuer degradation.

        The policy must infer a bad moment from this rather than being told.
        It is noisy by construction - a low-volume issuer produces a very
        unreliable estimate, which is exactly the difficulty a production
        system faces.
        """
        if self.issuer_volume_last_hour == 0:
            return 0.0
        return self.issuer_failures_last_hour / self.issuer_volume_last_hour


class LoggedDecision(BaseModel):
    """What the historical policy did, and with what probability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    action: ActionType
    propensity: float = Field(gt=0.0, le=1.0)
    is_holdout: bool
    policy_name: str


class RealizedOutcome(BaseModel):
    """What actually happened after the logged action was taken.

    This *is* observable in production — a merchant knows whether the retry
    worked. Only the counterfactuals are hidden, which is exactly the gap the
    uplift model has to bridge: one outcome per case, six unobserved.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    action: ActionType
    recovered: bool
    mandate_cancelled: bool = False
