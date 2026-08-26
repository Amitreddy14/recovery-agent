"""Tier-2 parameters: our assumptions, not published data.

Everything in this module is a modelling choice. NPCI publishes the aggregate
TD/BD split per bank but not the reason-level composition inside those
buckets, so the mixes below are assumptions.

They are isolated here for three reasons:

1. A reviewer can see the full extent of what we assumed in one file.
2. Phase 9's robustness sweep perturbs exactly these values, so results can
   be reported across the assumption space rather than at one lucky point.
3. Nothing empirical can drift into this file without being obvious in review.

Every entry carries a `rationale` in the docstring. "It seemed reasonable" is
not a rationale; if we cannot justify a number, it should be swept wide.
"""

from __future__ import annotations

from typing import Final

from recovery.domain.enums import FailureReason

# --- Reason composition within Technical Declines --------------------------
# Rationale: TD is infrastructure failure. Bank-side unavailability dominates
# because it is the modal cause of NPCI-classified TD; gateway and network
# errors are a smaller share; timeouts are the tail. Swept +/- 50% relative
# in Phase 9.

TD_REASON_MIX: Final[dict[FailureReason, float]] = {
    FailureReason.BANK_DOWNTIME: 0.55,
    FailureReason.GATEWAY_TECHNICAL_ERROR: 0.20,
    FailureReason.NETWORK_ERROR: 0.15,
    FailureReason.PAYMENT_TIMEOUT: 0.10,
}

# --- Reason composition within Business Declines ---------------------------
# Rationale: BD is user-side. Insufficient funds is the largest single cause
# in consumer payments; authentication failure and explicit cancellation
# follow; limit breaches and expired instruments are smaller. The split
# between INSUFFICIENT_FUNDS and PAYMENT_CANCELLED_BY_USER matters more than
# any other assumption in this file, because the two have opposite correct
# treatments — so it is swept widest.

BD_REASON_MIX: Final[dict[FailureReason, float]] = {
    FailureReason.INSUFFICIENT_FUNDS: 0.42,
    FailureReason.INVALID_OTP: 0.18,
    FailureReason.PAYMENT_CANCELLED_BY_USER: 0.22,
    FailureReason.PAYMENT_LIMIT_EXCEEDED: 0.10,
    FailureReason.CARD_EXPIRED: 0.08,
}

# --- Issuer degradation dynamics -------------------------------------------
# Rationale: TD is not uniform in time. Issuers degrade in episodes -
# maintenance windows, capacity events, financial year-end rushes - during
# which their TD rate multiplies for a bounded period before recovering.
# NPCI's monthly per-bank figure is the *average* over such episodes, so a
# simulator that applies it uniformly would understate clustering and make
# retry timing look worthless.
#
# This is the assumption the "retry timing against issuer health" thesis
# depends on, so it is swept hardest and reported separately.

DEGRADATION_EPISODE_RATE_PER_DAY: Final[float] = 0.08
"""Probability an issuer enters a degraded episode on a given day."""

DEGRADATION_DURATION_HOURS_RANGE: Final[tuple[float, float]] = (0.5, 6.0)
"""Episode length, sampled uniformly."""

DEGRADATION_TD_MULTIPLIER_RANGE: Final[tuple[float, float]] = (4.0, 20.0)
"""TD rate multiplier while degraded. Calibration rescales the baseline so
the long-run mean still matches the published NPCI figure."""

# --- Mandate failure dynamics ----------------------------------------------
# Rationale: recurring debits fail for different reasons than one-off
# payments - balance timing dominates, and failures cluster away from salary
# dates. Autopay volumes are published but the failure composition is not.

MANDATE_BD_UPLIFT_FACTOR: Final[float] = 1.35
"""Recurring debits fail on BD more often than one-off payments, because the
customer is not present to correct a balance shortfall."""

SALARY_DAY_OF_MONTH: Final[int] = 1
SALARY_WINDOW_DAYS: Final[int] = 3
SALARY_WINDOW_BD_MULTIPLIER: Final[float] = 0.45
"""BD rate falls inside the post-salary window. Drives the timing thesis for
scheduled retries."""


ASSUMPTION_REGISTRY: Final[dict[str, str]] = {
    "TD_REASON_MIX": "Composition of technical declines by reason",
    "BD_REASON_MIX": "Composition of business declines by reason",
    "DEGRADATION_EPISODE_RATE_PER_DAY": "Issuer degradation episode frequency",
    "DEGRADATION_DURATION_HOURS_RANGE": "Issuer degradation episode length",
    "DEGRADATION_TD_MULTIPLIER_RANGE": "TD multiplier during degradation",
    "MANDATE_BD_UPLIFT_FACTOR": "Recurring vs one-off BD rate ratio",
    "SALARY_DAY_OF_MONTH": "Assumed salary credit date",
    "SALARY_WINDOW_DAYS": "Length of the post-salary window",
    "SALARY_WINDOW_BD_MULTIPLIER": "BD rate reduction post-salary",
}
"""Every Tier-2 assumption, by name. Phase 9's sweep iterates this registry,
so an assumption added without registering it will be caught by test."""
