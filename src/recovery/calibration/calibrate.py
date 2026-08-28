"""Derive `WorldParameters` from an ingested snapshot.

The one piece of real arithmetic here is separating an issuer's *baseline*
technical decline rate from its *published* rate.

NPCI's monthly per-bank TD is an average that already contains degradation
episodes. If the generator applied that average uniformly in time, it would
be simulating a world with no clustering at all - and the retry-timing thesis
would be untestable by construction, because there would never be a bad
moment to avoid.

So we invert it. With an issuer degraded for a fraction `f` of the time at a
multiplier `m`:

    published = baseline * (1 - f) + baseline * m * f
              = baseline * (1 - f + m * f)

    baseline  = published / (1 - f + m * f)

The generator then reproduces the published mean *and* has clustered
episodes, which is the behaviour the recovery policy has to cope with.
"""

from __future__ import annotations

from recovery.calibration import assumptions
from recovery.calibration.models import (
    IssuerProfile,
    NpciSnapshot,
    WorldParameters,
)


def degraded_time_fraction(
    episode_rate_per_day: float,
    duration_hours_range: tuple[float, float],
) -> float:
    """Long-run fraction of time an issuer spends degraded."""
    lo, hi = duration_hours_range
    mean_duration_hours = (lo + hi) / 2.0
    fraction = episode_rate_per_day * mean_duration_hours / 24.0
    return min(fraction, 0.99)


def solve_baseline_td(
    published_td_rate: float,
    degraded_fraction: float,
    multiplier: float,
) -> float:
    """Invert the mixture so baseline + episodes reproduces the published mean."""
    inflation = 1.0 - degraded_fraction + multiplier * degraded_fraction
    if inflation <= 0:
        raise ValueError(f"non-positive inflation factor: {inflation}")
    return published_td_rate / inflation


def calibrate(
    snapshot: NpciSnapshot,
    *,
    episode_rate_per_day: float | None = None,
    duration_hours_range: tuple[float, float] | None = None,
    td_multiplier: float | None = None,
) -> WorldParameters:
    """Turn a snapshot into frozen generator parameters.

    Tier-2 assumptions may be overridden by keyword - that is how Phase 9's
    robustness sweep perturbs them. Tier-1 values from the snapshot cannot be
    overridden at all.
    """
    episode_rate = (
        episode_rate_per_day
        if episode_rate_per_day is not None
        else assumptions.DEGRADATION_EPISODE_RATE_PER_DAY
    )
    duration_range = (
        duration_hours_range
        if duration_hours_range is not None
        else assumptions.DEGRADATION_DURATION_HOURS_RANGE
    )
    multiplier = (
        td_multiplier
        if td_multiplier is not None
        else sum(assumptions.DEGRADATION_TD_MULTIPLIER_RANGE) / 2.0
    )

    fraction = degraded_time_fraction(episode_rate, duration_range)
    total_volume = snapshot.total_volume_millions

    profiles = tuple(
        IssuerProfile(
            bank_name=stat.bank_name,
            volume_share=stat.volume_millions / total_volume,
            published_td_rate=stat.td_rate,
            published_bd_rate=stat.bd_rate,
            baseline_td_rate=solve_baseline_td(stat.td_rate, fraction, multiplier),
            degradation_episode_rate_per_day=episode_rate,
            degradation_td_multiplier=multiplier,
        )
        for stat in snapshot.issuers
    )

    return WorldParameters(
        provenance=snapshot.provenance,
        issuers=profiles,
        td_reason_mix=dict(assumptions.TD_REASON_MIX),
        bd_reason_mix=dict(assumptions.BD_REASON_MIX),
        mandate_bd_uplift_factor=assumptions.MANDATE_BD_UPLIFT_FACTOR,
        salary_window_bd_multiplier=assumptions.SALARY_WINDOW_BD_MULTIPLIER,
        assumption_names=tuple(sorted(assumptions.ASSUMPTION_REGISTRY)),
    )


def effective_td_rate(profile: IssuerProfile, *, degraded_fraction: float) -> float:
    """Recompute the long-run TD implied by a profile. Used to verify that
    calibration round-trips back to the published figure."""
    return profile.baseline_td_rate * (
        1.0 - degraded_fraction + profile.degradation_td_multiplier * degraded_fraction
    )
