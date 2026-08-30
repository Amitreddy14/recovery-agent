"""Off-policy evaluation, and the validation of it.

The problem this solves in production: you have logged data from an old
policy and you want to know what a new policy would have earned, without
deploying it. The counterfactual is unobservable, so you use an estimator —
and you have no way to check whether the estimator is any good, because
checking would require the very counterfactual you lack.

That is the argument for building this world. Here the counterfactuals *are*
observable, so the estimators can be scored against truth before being
trusted somewhere truth is unavailable. This module is therefore not a
convenience: it is the reason the synthetic-data choice is methodologically
correct rather than a compromise.

Three estimators, in increasing sophistication:

* **IPS** — reweight logged rewards by 1/propensity. Unbiased, high variance.
  A single case logged with propensity 0.02 carries fifty times the weight of
  one logged at 1.0, so a handful of rare actions can dominate the estimate.
* **SNIPS** — the same, normalised by the sum of weights. Slightly biased,
  much lower variance. Usually the better trade.
* **Doubly robust** — combines a reward model with IPS correction. Consistent
  if *either* the reward model or the propensities are right, which is why it
  is the standard choice when neither can be fully trusted.

If the reported errors here were small only because the estimators were tuned
against the truth they are scored on, the exercise would be circular. They
are not: each estimator sees logged actions, rewards and propensities, and
nothing else.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from recovery.domain.enums import ActionType
from recovery.world.oracle.outcomes import PotentialOutcomes

MIN_PROPENSITY = 1e-3
"""Floor on propensity before inverting.

Without it, a single mis-logged near-zero propensity produces a weight in the
thousands and the estimate becomes that one case. Clipping introduces a small
bias in exchange for a finite variance, which is the right trade when the
alternative is an estimator that occasionally returns nonsense.
"""


@dataclass(frozen=True)
class OPEResult:
    estimator: str
    estimate_paise: float
    truth_paise: float
    n: int

    @property
    def error_paise(self) -> float:
        return self.estimate_paise - self.truth_paise

    @property
    def relative_error(self) -> float:
        return self.error_paise / self.truth_paise if self.truth_paise else float("inf")

    @property
    def absolute_relative_error(self) -> float:
        return abs(self.relative_error)


def _weights(
    logged_actions: Sequence[ActionType],
    propensities: Sequence[float],
    target_actions: Sequence[ActionType],
) -> np.ndarray:
    """Importance weights: 1/p where the target agrees with the log, else 0.

    A logged action the target policy would not have taken contributes
    nothing — we have no evidence about what the target would have earned
    there, and inventing some is exactly what the estimator must not do.
    """
    p = np.clip(np.asarray(propensities, dtype=np.float64), MIN_PROPENSITY, 1.0)
    agrees = np.asarray(
        [a is b for a, b in zip(logged_actions, target_actions, strict=True)],
        dtype=np.float64,
    )
    return agrees / p


def ips(
    rewards_paise: Sequence[float],
    logged_actions: Sequence[ActionType],
    propensities: Sequence[float],
    target_actions: Sequence[ActionType],
) -> float:
    w = _weights(logged_actions, propensities, target_actions)
    r = np.asarray(rewards_paise, dtype=np.float64)
    return float(np.sum(w * r) / len(r))


def snips(
    rewards_paise: Sequence[float],
    logged_actions: Sequence[ActionType],
    propensities: Sequence[float],
    target_actions: Sequence[ActionType],
) -> float:
    """Self-normalised IPS: divide by the realised weight mass rather than n.

    When the target policy agrees with the log on only a fraction of cases,
    IPS divides by the full n and understates. SNIPS divides by what it
    actually observed.
    """
    w = _weights(logged_actions, propensities, target_actions)
    r = np.asarray(rewards_paise, dtype=np.float64)
    denominator = float(np.sum(w))
    return float(np.sum(w * r) / denominator) if denominator > 0 else 0.0


def doubly_robust(
    rewards_paise: Sequence[float],
    logged_actions: Sequence[ActionType],
    propensities: Sequence[float],
    target_actions: Sequence[ActionType],
    reward_model: dict[ActionType, float],
) -> float:
    """DR = model prediction + IPS-corrected residual.

    Consistent if either the reward model or the propensities are correct.
    The model supplies a baseline everywhere, including cases where the
    target disagrees with the log; the IPS term corrects it only where
    evidence exists.
    """
    w = _weights(logged_actions, propensities, target_actions)
    r = np.asarray(rewards_paise, dtype=np.float64)
    predicted_target = np.asarray(
        [reward_model.get(a, 0.0) for a in target_actions], dtype=np.float64
    )
    predicted_logged = np.asarray(
        [reward_model.get(a, 0.0) for a in logged_actions], dtype=np.float64
    )
    return float(np.mean(predicted_target + w * (r - predicted_logged)))


def fit_reward_model(
    rewards_paise: Sequence[float], logged_actions: Sequence[ActionType]
) -> dict[ActionType, float]:
    """Mean logged reward per action.

    Deliberately crude — a per-action average, not a conditional model. DR's
    claim is that it tolerates a wrong reward model as long as propensities
    are right, and using a weak model here is what puts that claim under
    test rather than assuming it.
    """
    totals: dict[ActionType, list[float]] = {}
    for action, reward in zip(logged_actions, rewards_paise, strict=True):
        totals.setdefault(action, []).append(float(reward))
    return {a: float(np.mean(v)) for a, v in totals.items()}


def true_policy_value(
    outcomes: Sequence[PotentialOutcomes],
    amounts_paise: Sequence[int],
    target_actions: Sequence[ActionType],
) -> float:
    """What the target policy would actually have earned.

    Reads the counterfactual directly. Available only because this is a
    simulated world, and the entire point of the comparison below.
    """
    total = 0.0
    for outcome, amount, action in zip(outcomes, amounts_paise, target_actions, strict=True):
        if outcome.recovered.get(action, False):
            total += amount
    return total / len(target_actions)


def validate_estimators(
    *,
    outcomes: Sequence[PotentialOutcomes],
    amounts_paise: Sequence[int],
    logged_actions: Sequence[ActionType],
    propensities: Sequence[float],
    target_actions: Sequence[ActionType],
) -> list[OPEResult]:
    """Score every estimator against ground truth.

    The rewards passed to the estimators are the *logged* ones — what was
    actually observed under the action actually taken. The truth is computed
    separately from the counterfactuals. No estimator sees it.
    """
    rewards = [
        float(amount) if outcome.recovered.get(action, False) else 0.0
        for outcome, amount, action in zip(outcomes, amounts_paise, logged_actions, strict=True)
    ]
    truth = true_policy_value(outcomes, amounts_paise, target_actions)
    model = fit_reward_model(rewards, logged_actions)
    n = len(target_actions)

    return [
        OPEResult("ips", ips(rewards, logged_actions, propensities, target_actions), truth, n),
        OPEResult("snips", snips(rewards, logged_actions, propensities, target_actions), truth, n),
        OPEResult(
            "doubly_robust",
            doubly_robust(rewards, logged_actions, propensities, target_actions, model),
            truth,
            n,
        ),
    ]


def effective_sample_size(
    logged_actions: Sequence[ActionType],
    propensities: Sequence[float],
    target_actions: Sequence[ActionType],
) -> float:
    """How much evidence the estimate actually rests on.

    A low ESS relative to n means a few heavily-weighted cases dominate, and
    the estimate is fragile regardless of how close it happens to land. Worth
    reporting alongside the error, because a lucky estimate on an ESS of 40 is
    not the same result as an accurate one on 4,000.
    """
    w = _weights(logged_actions, propensities, target_actions)
    if not np.any(w):
        return 0.0
    return float(np.sum(w) ** 2 / np.sum(w**2))
