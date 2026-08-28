"""Calibration data structures.

`NpciSnapshot` is what we ingest. `IssuerProfile` and `WorldParameters` are
what the Phase 3 generator consumes. The separation matters: the snapshot is
evidence, the parameters are derived, and the derivation is testable.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

from recovery.calibration import priors
from recovery.domain.enums import FailureReason


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class IssuerStatistic(Frozen):
    """One bank's published figures for one month.

    `approved_rate` is optional but strongly preferred: when present it lets
    us validate approved + BD + TD = 1.0 per row, which is a property of the
    source rather than an assumption of ours.
    """

    bank_name: str
    td_rate: float = Field(ge=0.0, le=1.0)
    bd_rate: float = Field(ge=0.0, le=1.0)
    volume_millions: float = Field(gt=0.0)
    approved_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    @property
    def total_decline_rate(self) -> float:
        return self.td_rate + self.bd_rate

    @property
    def success_rate(self) -> float:
        return 1.0 - self.total_decline_rate

    @model_validator(mode="after")
    def _declines_cannot_exceed_volume(self) -> IssuerStatistic:
        if self.total_decline_rate >= 1.0:
            raise ValueError(
                f"{self.bank_name}: TD + BD = {self.total_decline_rate:.4f} "
                "leaves no successful transactions"
            )
        return self

    @model_validator(mode="after")
    def _components_sum_to_one(self) -> IssuerStatistic:
        """approved + BD + TD must account for every transaction.

        This is the strongest single check available, because it is an
        internal property of NPCI's own table. It catches a percent/rate
        confusion, a mis-mapped column, and a bad row - all at once.
        """
        if self.approved_rate is None:
            return self
        total = self.approved_rate + self.bd_rate + self.td_rate
        if abs(total - 1.0) > priors.APPROVED_BD_TD_SUM_TOLERANCE:
            raise ValueError(
                f"{self.bank_name}: approved + BD + TD = {total:.5f}, expected "
                f"1.0. Check column mapping and whether values are rates or "
                f"percentages."
            )
        return self


class Provenance(Frozen):
    """Where a snapshot came from. Required, not optional.

    A calibration whose source cannot be stated is indistinguishable from
    invented numbers, so the schema refuses to represent one.
    """

    source_name: str
    source_url: str
    reporting_period: str = Field(description="e.g. '2026-07'")
    retrieved_on: date
    retrieved_by: str
    notes: str = ""


class NpciSnapshot(Frozen):
    """A dated set of per-issuer statistics with its provenance."""

    provenance: Provenance
    issuers: tuple[IssuerStatistic, ...] = Field(min_length=1)

    @property
    def total_volume_millions(self) -> float:
        return sum(i.volume_millions for i in self.issuers)

    @property
    def volume_weighted_td(self) -> float:
        total = self.total_volume_millions
        return sum(i.td_rate * i.volume_millions for i in self.issuers) / total

    @property
    def volume_weighted_bd(self) -> float:
        total = self.total_volume_millions
        return sum(i.bd_rate * i.volume_millions for i in self.issuers) / total

    @property
    def volume_weighted_success(self) -> float:
        return 1.0 - self.volume_weighted_td - self.volume_weighted_bd

    @model_validator(mode="after")
    def _bank_names_unique(self) -> NpciSnapshot:
        names = [i.bank_name for i in self.issuers]
        if len(names) != len(set(names)):
            dupes = {n for n in names if names.count(n) > 1}
            raise ValueError(f"duplicate bank names in snapshot: {sorted(dupes)}")
        return self


class IssuerProfile(Frozen):
    """Per-issuer parameters the generator samples from.

    `baseline_td_rate` is *below* the published rate: the published figure is
    a monthly average that already includes degradation episodes, so applying
    it uniformly would double-count them. `calibrate` solves for the baseline
    such that baseline plus episodes reproduces the published mean.
    """

    bank_name: str
    volume_share: float = Field(gt=0.0, le=1.0)
    published_td_rate: float = Field(ge=0.0, le=1.0)
    published_bd_rate: float = Field(ge=0.0, le=1.0)
    baseline_td_rate: float = Field(ge=0.0, le=1.0)
    degradation_episode_rate_per_day: float = Field(ge=0.0)
    degradation_td_multiplier: float = Field(ge=1.0)

    @model_validator(mode="after")
    def _baseline_below_published(self) -> IssuerProfile:
        if self.baseline_td_rate > self.published_td_rate + 1e-9:
            raise ValueError(
                f"{self.bank_name}: baseline TD {self.baseline_td_rate:.5f} exceeds "
                f"published {self.published_td_rate:.5f}; degradation episodes can "
                "only raise the effective rate, never lower it"
            )
        return self


class WorldParameters(Frozen):
    """The complete, frozen parameter set handed to the Phase 3 generator."""

    provenance: Provenance
    issuers: tuple[IssuerProfile, ...] = Field(min_length=1)
    td_reason_mix: dict[FailureReason, float]
    bd_reason_mix: dict[FailureReason, float]
    mandate_bd_uplift_factor: float = Field(gt=0.0)
    salary_window_bd_multiplier: float = Field(gt=0.0)
    assumption_names: tuple[str, ...]

    @model_validator(mode="after")
    def _shares_and_mixes_sum_to_one(self) -> WorldParameters:
        share_total = sum(i.volume_share for i in self.issuers)
        if abs(share_total - 1.0) > 1e-6:
            raise ValueError(f"issuer volume shares sum to {share_total}, expected 1.0")

        for label, mix in (
            ("td_reason_mix", self.td_reason_mix),
            ("bd_reason_mix", self.bd_reason_mix),
        ):
            total = sum(mix.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"{label} sums to {total}, expected 1.0")
        return self
