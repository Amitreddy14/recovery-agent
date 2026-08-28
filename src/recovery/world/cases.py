"""Observable case features and the historical logging policy.

`CaseFeatures` is everything the policy is allowed to see. It is serialised
to a different file from the oracle, so the separation is physical as well as
enforced by CI.

The logging policy imitates what a merchant does today - retry almost
everything - with epsilon-greedy exploration and a randomised holdout. Both
exist so the logged data supports unbiased off-policy evaluation. Without
recorded propensities, an IPS or doubly-robust estimator is undefined, which
is why `Decision.propensity` is a required field (ADR-0005).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from recovery.domain.enums import (
    ActionType,
    CaseType,
    DeclineClass,
    FailureReason,
    MandateCategory,
    PaymentMethod,
)

EXPLORATION_EPSILON = 0.15
"""Fraction of non-holdout cases where the logging policy acts at random."""

HOLDOUT_FRACTION = 0.20
"""Fraction assigned uniformly at random across the full action set,
including true no-action control. This slice carries the unbiased estimates."""

LOGGABLE_ACTIONS: tuple[ActionType, ...] = (
    ActionType.NO_ACTION,
    ActionType.RETRY_NOW,
    ActionType.RETRY_SCHEDULED,
    ActionType.RETRY_ALTERNATE_RAIL,
    ActionType.SEND_PAYMENT_LINK,
    ActionType.PRE_DEBIT_NUDGE,
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


def _naive_preference(features: CaseFeatures) -> ActionType:
    """What a typical merchant does today.

    Retry almost everything, message when a retry has already failed. It is
    deliberately unsophisticated: this is the behaviour our policy has to
    beat, and it is also the source of the biased logged data that off-policy
    evaluation has to correct for.
    """
    if features.case_type == CaseType.MANDATE_FAILURE:
        if features.consecutive_mandate_failures >= 2:
            return ActionType.SEND_PAYMENT_LINK
        return ActionType.RETRY_SCHEDULED
    if features.reason == FailureReason.CARD_EXPIRED:
        return ActionType.SEND_PAYMENT_LINK
    return ActionType.RETRY_NOW


def log_action(features: CaseFeatures, rng: np.random.Generator) -> LoggedDecision:
    """Choose an action and record the probability with which it was chosen.

    Propensities must be exact, not approximate: an IPS estimator divides by
    them, so a wrong value silently biases every downstream estimate rather
    than producing a visible error.
    """
    n = len(LOGGABLE_ACTIONS)

    if rng.random() < HOLDOUT_FRACTION:
        action = LOGGABLE_ACTIONS[int(rng.integers(n))]
        return LoggedDecision(
            case_id=features.case_id,
            action=action,
            propensity=1.0 / n,
            is_holdout=True,
            policy_name="uniform_holdout",
        )

    preferred = _naive_preference(features)
    if rng.random() < EXPLORATION_EPSILON:
        action = LOGGABLE_ACTIONS[int(rng.integers(n))]
    else:
        action = preferred

    # Exact propensity under epsilon-greedy: the preferred action can also be
    # reached through the exploration branch, so its probability is the sum of
    # both paths.
    uniform_share = EXPLORATION_EPSILON / n
    propensity = (
        (1.0 - EXPLORATION_EPSILON) + uniform_share if action == preferred else uniform_share
    )

    return LoggedDecision(
        case_id=features.case_id,
        action=action,
        propensity=propensity * (1.0 - HOLDOUT_FRACTION),
        is_holdout=False,
        policy_name="naive_epsilon_greedy",
    )
