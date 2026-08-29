"""Targeting policies and the comparison that tests Claim 1.

Four rankings of the same population:

* **uplift** — rank by estimated `p(recovery | act) - p(recovery | leave alone)`
* **risk** — rank by estimated `1 - p(recovery | leave alone)`, i.e. who is
  most likely to be lost. This is what almost every production system does.
* **random** — the null.
* **oracle** — rank by true uplift. Computable only here, and reported as a
  ceiling rather than a competitor.

The risk ranking is the one that matters. It is not a strawman: predicting who
will churn and intervening on the highest scores is standard practice, it is
what a well-run team does with a churn model, and it is wrong for a reason
that is easy to state and hard to notice. It ranks sure things and lost causes
at the top — the sure things because they look risky and recover anyway, the
lost causes because they look risky and never recover — while persuadables,
who by definition sit in the middle of the risk distribution, are pushed down.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from recovery.uplift.learners import TLearner, TreatmentData, XLearner
from recovery.uplift.metrics import QiniCurve, qini_curve, uplift_at_k


@dataclass(frozen=True)
class TargetingResult:
    name: str
    qini: QiniCurve
    uplift_at_10: float
    uplift_at_30: float

    @property
    def coefficient(self) -> float:
        return self.qini.coefficient


def split_arms(
    x: np.ndarray,
    treated_mask: np.ndarray,
    outcomes: np.ndarray,
    propensity: np.ndarray,
) -> tuple[TreatmentData, TreatmentData]:
    control = TreatmentData(
        x=x[~treated_mask], y=outcomes[~treated_mask], propensity=propensity[~treated_mask]
    )
    treated = TreatmentData(
        x=x[treated_mask], y=outcomes[treated_mask], propensity=propensity[treated_mask]
    )
    return control, treated


def rank_by_risk(model: TLearner | XLearner, x: np.ndarray) -> np.ndarray:
    """The conventional approach: target whoever is least likely to recover."""
    return 1.0 - model.predict_baseline(x)


def compare_targeting(
    *,
    x_train: np.ndarray,
    treated_train: np.ndarray,
    y_train: np.ndarray,
    propensity_train: np.ndarray,
    x_eval: np.ndarray,
    treated_eval: np.ndarray,
    y_eval: np.ndarray,
    propensity_eval: np.ndarray,
    true_uplift_eval: Sequence[float] | None = None,
    seed: int = 0,
) -> list[TargetingResult]:
    """Fit the learners on logged data, evaluate on the randomised holdout.

    The split is the methodological point. Training data comes from a biased
    logging policy, which is realistic — a merchant's history reflects what
    they chose to do. Evaluation happens only on the uniformly randomised
    holdout, where treatment assignment is independent of covariates, so the
    Qini numbers are unbiased rather than an artefact of the logging policy's
    preferences.
    """
    control, treated = split_arms(x_train, treated_train, y_train, propensity_train)

    t_learner = TLearner(seed=seed).fit(control, treated)
    x_learner = XLearner(seed=seed).fit(control, treated)

    rng = np.random.default_rng(seed)
    rankings: dict[str, np.ndarray] = {
        "uplift_x_learner": x_learner.predict_uplift(x_eval, propensity_eval),
        "uplift_t_learner": t_learner.predict_uplift(x_eval),
        "risk_ranking": rank_by_risk(x_learner, x_eval),
        "random": rng.random(len(x_eval)),
    }
    if true_uplift_eval is not None:
        rankings["oracle_uplift"] = np.asarray(true_uplift_eval)

    results: list[TargetingResult] = []
    for name, scores in rankings.items():
        results.append(
            TargetingResult(
                name=name,
                qini=qini_curve(scores, treated_eval, y_eval),
                uplift_at_10=uplift_at_k(scores, treated_eval, y_eval, 0.10),
                uplift_at_30=uplift_at_k(scores, treated_eval, y_eval, 0.30),
            )
        )
    return results
