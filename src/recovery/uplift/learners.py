"""Uplift learners.

Implemented directly rather than imported from `causalml`. Two reasons: the
X-learner is about sixty lines and writing it makes its assumptions explicit,
and the propensity weighting in step 4 has to use *our* logged propensities
rather than an estimated propensity score, which is a modification a library
default would obscure.

The problem both learners solve: we observe one outcome per case. A customer
who was retried was not also messaged, so `Y(retry) - Y(message)` is never
observed for anyone. Uplift learning estimates that difference from the
covariate structure.

**Why the X-learner and not just the T-learner.** The T-learner fits one model
per arm and subtracts. That is fine when arms are balanced, but ours are not:
the naive logging policy sends most cases to `RETRY_NOW`, so `NO_ACTION` and
`PRE_DEBIT_NUDGE` are comparatively starved. When one arm has far less data,
its model is noisier, and subtracting a noisy estimate from a precise one
produces uplift dominated by the noisy arm's error. The X-learner cross-fits —
it uses the well-estimated arm to impute treatment effects for the sparse arm —
and then blends the two estimates by propensity, weighting each where it is
most reliable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 30,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "verbose": -1,
}
"""Deliberately conservative. Shallow trees and a high minimum leaf size,
because uplift is a difference of two noisy quantities and an overfitted arm
model produces confident nonsense rather than a visibly bad fit."""


@dataclass
class TreatmentData:
    """Observations for one arm."""

    x: np.ndarray
    y: np.ndarray
    propensity: np.ndarray

    def __post_init__(self) -> None:
        if not (len(self.x) == len(self.y) == len(self.propensity)):
            raise ValueError("x, y and propensity must have equal length")

    @property
    def n(self) -> int:
        return len(self.y)


class TLearner:
    """Fit one outcome model per arm; uplift is the difference."""

    name = "t_learner"

    def __init__(self, params: dict[str, Any] | None = None, seed: int = 0) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.seed = seed
        self.mu0: LGBMClassifier | None = None
        self.mu1: LGBMClassifier | None = None
        self._control_rate = 0.0
        self._treated_rate = 0.0

    def fit(self, control: TreatmentData, treated: TreatmentData) -> TLearner:
        self._control_rate = float(control.y.mean()) if control.n else 0.0
        self._treated_rate = float(treated.y.mean()) if treated.n else 0.0

        self.mu0 = self._fit_arm(control)
        self.mu1 = self._fit_arm(treated)
        return self

    def _fit_arm(self, data: TreatmentData) -> LGBMClassifier | None:
        # A single-class arm carries no signal. Returning None and falling back
        # to the observed base rate is honest; fitting would raise or, worse,
        # silently predict a constant that looks like a model.
        if data.n < 30 or len(np.unique(data.y)) < 2:
            return None
        model = LGBMClassifier(**self.params, random_state=self.seed)
        model.fit(data.x, data.y)
        return model

    def _predict_arm(
        self, model: LGBMClassifier | None, x: np.ndarray, fallback: float
    ) -> np.ndarray:
        if model is None:
            return np.full(len(x), fallback)
        proba = np.asarray(model.predict_proba(x), dtype=np.float64)
        return proba[:, 1]

    def predict_uplift(self, x: np.ndarray) -> np.ndarray:
        p1 = self._predict_arm(self.mu1, x, self._treated_rate)
        p0 = self._predict_arm(self.mu0, x, self._control_rate)
        return np.asarray(p1 - p0, dtype=np.float64)

    def predict_baseline(self, x: np.ndarray) -> np.ndarray:
        """P(recovery | no action). This is what a *risk* model predicts, and
        ranking on it is the baseline the uplift approach has to beat."""
        return self._predict_arm(self.mu0, x, self._control_rate)


class XLearner:
    """Cross-fitted uplift estimator, blended by logged propensity."""

    name = "x_learner"

    def __init__(self, params: dict[str, Any] | None = None, seed: int = 0) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.seed = seed
        self.base = TLearner(params, seed)
        self.tau0: LGBMRegressor | None = None
        self.tau1: LGBMRegressor | None = None
        self._mean_uplift = 0.0

    def fit(self, control: TreatmentData, treated: TreatmentData) -> XLearner:
        # Stage 1: outcome models per arm, as in the T-learner.
        self.base.fit(control, treated)

        # Stage 2: impute the unobserved half of each unit's effect.
        # For a treated unit we know Y(1) and predict Y(0); the difference is
        # an imputed individual effect. For a control unit, the reverse.
        d1 = treated.y - self.base._predict_arm(self.base.mu0, treated.x, self.base._control_rate)
        d0 = self.base._predict_arm(self.base.mu1, control.x, self.base._treated_rate) - control.y

        # Stage 3: regress the imputed effects on covariates.
        self.tau1 = self._fit_tau(treated.x, d1)
        self.tau0 = self._fit_tau(control.x, d0)

        both = np.concatenate([d1, d0]) if (treated.n and control.n) else np.array([0.0])
        self._mean_uplift = float(both.mean())
        return self

    def _fit_tau(self, x: np.ndarray, d: np.ndarray) -> LGBMRegressor | None:
        if len(d) < 30:
            return None
        model = LGBMRegressor(**self.params, random_state=self.seed)
        model.fit(x, d)
        return model

    def predict_uplift(self, x: np.ndarray, propensity: np.ndarray | None = None) -> np.ndarray:
        """Blend the two effect models.

        Weighting by propensity is the point of the X-learner: `tau0` is fitted
        on control units and is more reliable where treatment was rare, `tau1`
        the reverse. Using our *logged* propensities rather than an estimated
        score removes a source of error a library default would introduce,
        since we know the assignment probabilities exactly (ADR-0005).
        """
        t1 = (
            np.asarray(self.tau1.predict(x), dtype=np.float64)
            if self.tau1 is not None
            else np.full(len(x), self._mean_uplift)
        )
        t0 = (
            np.asarray(self.tau0.predict(x), dtype=np.float64)
            if self.tau0 is not None
            else np.full(len(x), self._mean_uplift)
        )
        if propensity is None:
            return np.asarray(0.5 * (t0 + t1), dtype=np.float64)
        g = np.clip(propensity, 1e-6, 1 - 1e-6)
        return np.asarray(g * t0 + (1.0 - g) * t1, dtype=np.float64)

    def predict_baseline(self, x: np.ndarray) -> np.ndarray:
        return self.base.predict_baseline(x)
