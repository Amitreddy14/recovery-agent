"""The historical logging policy.

Imitates what a merchant does today - retry almost everything - with
epsilon-greedy exploration and a randomised holdout. Both exist so the logged
data supports unbiased off-policy evaluation: without recorded propensities an
IPS or doubly-robust estimator is undefined, which is why
`Decision.propensity` is a required field (ADR-0005).

The observable schemas themselves live in `recovery.domain.observations`.
"""

from __future__ import annotations

import numpy as np

from recovery.domain.enums import ActionType, CaseType, FailureReason
from recovery.domain.observations import CaseFeatures, LoggedDecision

__all__ = [
    "EXPLORATION_EPSILON",
    "HOLDOUT_FRACTION",
    "LOGGABLE_ACTIONS",
    "CaseFeatures",
    "LoggedDecision",
    "log_action",
]

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
