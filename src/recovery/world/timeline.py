"""Issuer health over time.

Technical declines cluster. An issuer is mostly healthy, then degrades for a
bounded window, then recovers. Calibration (ADR-0007) solved for a baseline
rate such that baseline plus these episodes reproduces the NPCI published
monthly mean, so this module is what makes that inversion true rather than
merely asserted.

This is also what makes retry *timing* a real decision. Without clustering,
every moment is equally good and the timing thesis is untestable by
construction.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from pydantic import BaseModel, ConfigDict

from recovery.calibration.models import IssuerProfile


class DegradationEpisode(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bank_name: str
    starts_at: datetime
    ends_at: datetime
    td_multiplier: float

    def covers(self, moment: datetime) -> bool:
        return self.starts_at <= moment < self.ends_at


class IssuerTimeline:
    """Per-issuer degradation schedule for a simulated window.

    Episodes are generated once, up front, for the whole window. That matters
    for the counterfactual framework: when we ask what *would* have happened
    had we retried at 14:00 instead of 09:00, the issuer's state at 14:00 must
    already be determined. Sampling it lazily at query time would make the
    counterfactual depend on the order in which questions are asked.
    """

    def __init__(
        self,
        profiles: tuple[IssuerProfile, ...],
        start: datetime,
        days: int,
        duration_hours_range: tuple[float, float],
        rng: np.random.Generator,
    ) -> None:
        self.start = start
        self.end = start + timedelta(days=days)
        self._baseline: dict[str, float] = {p.bank_name: p.baseline_td_rate for p in profiles}
        self._episodes: dict[str, list[DegradationEpisode]] = {}

        lo, hi = duration_hours_range
        for profile in profiles:
            episodes: list[DegradationEpisode] = []
            n = rng.poisson(profile.degradation_episode_rate_per_day * days)
            for _ in range(int(n)):
                offset_h = float(rng.uniform(0, days * 24))
                dur_h = float(rng.uniform(lo, hi))
                begins = start + timedelta(hours=offset_h)
                episodes.append(
                    DegradationEpisode(
                        bank_name=profile.bank_name,
                        starts_at=begins,
                        ends_at=begins + timedelta(hours=dur_h),
                        td_multiplier=profile.degradation_td_multiplier,
                    )
                )
            episodes.sort(key=lambda e: e.starts_at)
            self._episodes[profile.bank_name] = episodes

    def is_degraded(self, bank_name: str, moment: datetime) -> bool:
        return any(e.covers(moment) for e in self._episodes.get(bank_name, ()))

    def td_rate_at(self, bank_name: str, moment: datetime) -> float:
        """Effective technical decline rate for this issuer at this moment."""
        base = self._baseline.get(bank_name, 0.0)
        for episode in self._episodes.get(bank_name, ()):
            if episode.covers(moment):
                return min(base * episode.td_multiplier, 0.95)
        return base

    def next_healthy_moment(
        self, bank_name: str, after: datetime, horizon_hours: int = 48
    ) -> datetime | None:
        """When this issuer next recovers, if it is currently degraded.

        The generator uses this only to place episodes. The *policy* must
        infer degradation from the failure stream - it has no access to this
        object, which is why timeline lives in `world` and not in a shared
        utility module.
        """
        for episode in self._episodes.get(bank_name, ()):
            if episode.covers(after):
                if episode.ends_at <= after + timedelta(hours=horizon_hours):
                    return episode.ends_at
                return None
        return after

    def episode_count(self, bank_name: str) -> int:
        return len(self._episodes.get(bank_name, ()))

    def total_episodes(self) -> int:
        return sum(len(v) for v in self._episodes.values())
