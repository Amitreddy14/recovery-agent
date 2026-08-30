"""Robustness sweep.

Every result so far was measured under one set of Tier-2 assumptions — the
reason mixes, degradation dynamics and salary-window effects registered in
`calibration/assumptions.py`. Those are modelling choices, not published
data, and a result that holds only at the values we happened to pick is not a
result.

This module perturbs them across a grid and reports where the policy wins and
where it does not. Naming the losing region is the point: an approach that
wins everywhere usually means the world was built to let it.

**Paired comparison.** Every policy is evaluated on the *same* generated
world with the *same* seed, and the reported quantity is the paired
difference. This matters because between-world variance is large (INC-016
cost several hours to that lesson) while the paired difference is far more
stable — the world's noise cancels when both policies face it. Comparing
independent runs would need an order of magnitude more configurations to say
anything.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from recovery.calibration.models import WorldParameters
from recovery.domain.enums import FailureReason


@dataclass(frozen=True)
class SweepPoint:
    """One configuration of the Tier-2 assumption space."""

    cancellation_horizon: int
    insufficient_funds_share: float
    cancelled_share: float
    mandate_bd_uplift: float
    salary_window_multiplier: float

    def label(self) -> str:
        return (
            f"horizon={self.cancellation_horizon} "
            f"funds={self.insufficient_funds_share:.2f} "
            f"cancel={self.cancelled_share:.2f} "
            f"mandate={self.mandate_bd_uplift:.2f} "
            f"salary={self.salary_window_multiplier:.2f}"
        )


@dataclass
class SweepResult:
    point: SweepPoint
    seed: int
    net_by_policy: dict[str, int]

    def paired_delta(self, policy: str, baseline: str) -> int:
        return self.net_by_policy[policy] - self.net_by_policy[baseline]

    def beats(self, policy: str, baseline: str) -> bool:
        return self.paired_delta(policy, baseline) > 0


def default_grid() -> list[SweepPoint]:
    """The assumption space actually swept.

    Ranges are wide where the assumption is weakly grounded and narrow where
    it is not. `cancelled_share` gets the widest range because the split
    between INSUFFICIENT_FUNDS and PAYMENT_CANCELLED_BY_USER is the assumption
    the whole targeting argument is most sensitive to: the two have opposite
    correct treatments, and nothing published pins the ratio.
    """
    return [
        SweepPoint(
            cancellation_horizon=horizon,
            insufficient_funds_share=funds,
            cancelled_share=cancelled,
            mandate_bd_uplift=mandate,
            salary_window_multiplier=salary,
        )
        for horizon, funds, cancelled, mandate, salary in itertools.product(
            (3, 12, 24),  # a quarter, a year, two years of remaining value
            (0.28, 0.42, 0.56),
            (0.10, 0.22, 0.40),
            (1.0, 1.35),
            (0.30, 0.45, 0.70),
        )
    ]


def sample_grid(points: Sequence[SweepPoint], n: int, seed: int = 0) -> list[SweepPoint]:
    """Take a reproducible subset when the full grid is too slow.

    Deterministic by seed so a reported sweep can be re-run exactly. A random
    subset that cannot be reproduced is not evidence.
    """
    import random

    rng = random.Random(seed)
    if n >= len(points):
        return list(points)
    return rng.sample(list(points), n)


def perturb(params: WorldParameters, point: SweepPoint) -> WorldParameters:
    """Apply one sweep point to the world parameters.

    Tier-1 values — per-issuer TD/BD rates and volume shares from the NPCI
    snapshot — are untouched. Only the assumptions move (ADR-0006). The
    provenance travels with the perturbed parameters so a swept world can
    still say where its empirical half came from.
    """
    bd_mix = dict(params.bd_reason_mix)
    bd_mix[FailureReason.INSUFFICIENT_FUNDS] = point.insufficient_funds_share
    bd_mix[FailureReason.PAYMENT_CANCELLED_BY_USER] = point.cancelled_share

    # Renormalise the remaining reasons into whatever is left, preserving
    # their relative proportions. Rescaling everything uniformly would move
    # assumptions the sweep point did not name.
    fixed = {
        FailureReason.INSUFFICIENT_FUNDS,
        FailureReason.PAYMENT_CANCELLED_BY_USER,
    }
    remaining_mass = 1.0 - point.insufficient_funds_share - point.cancelled_share
    others = {k: v for k, v in params.bd_reason_mix.items() if k not in fixed}
    other_total = sum(others.values())
    if remaining_mass <= 0 or other_total <= 0:
        raise ValueError(f"sweep point leaves no mass for other reasons: {point.label()}")
    for reason, weight in others.items():
        bd_mix[reason] = remaining_mass * weight / other_total

    return params.model_copy(
        update={
            "bd_reason_mix": bd_mix,
            "mandate_bd_uplift_factor": point.mandate_bd_uplift,
            "salary_window_bd_multiplier": point.salary_window_multiplier,
        }
    )


@dataclass
class SweepSummary:
    policy: str
    baseline: str
    results: list[SweepResult]

    @property
    def wins(self) -> int:
        return sum(1 for r in self.results if r.beats(self.policy, self.baseline))

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total else 0.0

    @property
    def median_delta_paise(self) -> float:
        deltas = sorted(r.paired_delta(self.policy, self.baseline) for r in self.results)
        if not deltas:
            return 0.0
        mid = len(deltas) // 2
        return float(deltas[mid]) if len(deltas) % 2 else (deltas[mid - 1] + deltas[mid]) / 2.0

    @property
    def worst_delta_paise(self) -> int:
        return min(
            (r.paired_delta(self.policy, self.baseline) for r in self.results),
            default=0,
        )

    def losing_points(self) -> list[SweepPoint]:
        """Configurations where the policy loses.

        Reported explicitly. A sweep that only shows the wins is a
        presentation, not an analysis.
        """
        return [r.point for r in self.results if not r.beats(self.policy, self.baseline)]

    def describe_losses(self) -> str:
        losses = self.losing_points()
        if not losses:
            return (
                "no losing configurations in this grid — widen the ranges "
                "before treating that as a strong claim"
            )
        horizons = sorted({p.cancellation_horizon for p in losses})
        cancels = sorted({p.cancelled_share for p in losses})
        return (
            f"{len(losses)} losing configurations, concentrated at "
            f"horizon in {horizons} and cancelled_share in {cancels}"
        )
