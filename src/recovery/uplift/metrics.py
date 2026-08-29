"""Uplift evaluation metrics.

Implemented from the definition rather than imported, because the whole
content of the Qini curve is one correction term and it is worth being
explicit about it.

**Why AUC is the wrong metric here.** A classifier is scored on whether it
ranks outcomes correctly. An uplift model must be scored on whether it ranks
*treatment effects* correctly — and the treatment effect of any individual is
never observed, because a customer is either retried or not. So there is no
per-row label to compare against, and no confusion matrix to build.

**What Qini does instead.** Rank all cases by predicted uplift. Walk down the
ranking and, at each depth k, ask how many treated units in the top k
recovered versus how many control units did:

    Qini(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k)

The ratio term is the correction, and it is the reason a naive
"treated minus control" count is misleading: the two groups at depth k are
almost never the same size, so the control count has to be rescaled to what it
would have been at the treated group's size. Without it, an arm the logging
policy favoured looks better purely for being larger.

A model that ranks persuadables first climbs steeply and then flattens. A
model that ranks sleeping dogs first goes *negative* — which is the property
that makes Qini the right metric for this project, since a curve dipping below
zero is a direct measurement of value destroyed by contacting the wrong people.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

WARMUP_DEPTH = 0.05
"""Depth below which curve values are treated as noise.

At 1% depth a 2,000-case holdout has twenty observations split across two
arms, so the corrected difference swings on single events. Any statement
about a curve going negative has to be made past this point."""


class QiniCurve(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    depths: tuple[float, ...]
    values: tuple[float, ...]
    coefficient: float
    max_value: float
    max_depth: float
    min_value: float
    min_value_after_warmup: float

    @property
    def destroys_value(self) -> bool:
        """True if the curve goes meaningfully negative — the model is ranking
        sleeping dogs highly and contacting them costs more than it gains.

        Measured after the warmup depth. The first few points of a Qini curve
        rest on one or two observations, so the global minimum is usually
        sampling noise rather than a property of the ranking (INC-015).
        """
        return self.min_value_after_warmup < 0


def qini_curve(
    uplift_scores: np.ndarray,
    treated: np.ndarray,
    outcomes: np.ndarray,
    n_points: int = 100,
) -> QiniCurve:
    """Compute the Qini curve for one ranking.

    `treated` is a boolean array: was this unit treated, as opposed to left
    alone. `outcomes` is the observed binary outcome. Both come from logged
    data; neither requires a counterfactual.
    """
    if not (len(uplift_scores) == len(treated) == len(outcomes)):
        raise ValueError("scores, treated and outcomes must have equal length")
    n = len(uplift_scores)
    if n == 0:
        raise ValueError("empty input")

    order = np.argsort(-uplift_scores, kind="stable")
    t = np.asarray(treated, dtype=bool)[order]
    y = np.asarray(outcomes, dtype=float)[order]

    cum_treated = np.cumsum(t)
    cum_control = np.cumsum(~t)
    cum_y_treated = np.cumsum(y * t)
    cum_y_control = np.cumsum(y * ~t)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(cum_control > 0, cum_treated / np.maximum(cum_control, 1), 0.0)
    curve = cum_y_treated - cum_y_control * ratio

    idx = np.unique(np.linspace(0, n - 1, min(n_points, n)).astype(int))
    depths = tuple(float((i + 1) / n) for i in idx)
    values = tuple(float(curve[i]) for i in idx)

    # Qini coefficient: area between the curve and the diagonal a random
    # ranking would trace, normalised by the number of cases so batches of
    # different sizes are comparable.
    final = float(curve[-1])
    random_line = np.array([d * final for d in depths])
    actual = np.array(values)
    coefficient = float(np.trapezoid(actual - random_line, depths)) / max(final, 1e-9)

    peak = int(np.argmax(curve))
    warmup = max(1, int(n * WARMUP_DEPTH))
    return QiniCurve(
        depths=depths,
        values=values,
        coefficient=coefficient,
        max_value=float(curve[peak]),
        max_depth=float((peak + 1) / n),
        min_value=float(curve.min()),
        min_value_after_warmup=float(curve[warmup:].min()) if n > warmup else 0.0,
    )


def auuc(uplift_scores: np.ndarray, treated: np.ndarray, outcomes: np.ndarray) -> float:
    """Area under the uplift curve, normalised by batch size."""
    curve = qini_curve(uplift_scores, treated, outcomes)
    return float(np.trapezoid(curve.values, curve.depths))


def uplift_at_k(
    uplift_scores: np.ndarray,
    treated: np.ndarray,
    outcomes: np.ndarray,
    k: float = 0.3,
) -> float:
    """Corrected uplift within the top `k` fraction of the ranking.

    This is the number that answers "if we could only contact 30% of these
    customers, how much extra recovery would we get" — closer to the operating
    question than the area under the whole curve.
    """
    n = len(uplift_scores)
    cutoff = max(1, int(n * k))
    order = np.argsort(-uplift_scores, kind="stable")[:cutoff]
    t = np.asarray(treated, dtype=bool)[order]
    y = np.asarray(outcomes, dtype=float)[order]

    n_t, n_c = int(t.sum()), int((~t).sum())
    if n_t == 0 or n_c == 0:
        return 0.0
    return float(y[t].mean() - y[~t].mean())
