"""Calibration tests.

The important one is `TestBaselineInversion::test_round_trips_to_published`:
if that breaks, the simulator no longer reproduces published decline rates
and every downstream rupee figure loses its footing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recovery.calibration import assumptions, priors
from recovery.calibration.calibrate import (
    calibrate,
    degraded_time_fraction,
    effective_td_rate,
    solve_baseline_td,
)
from recovery.calibration.models import (
    IssuerProfile,
    IssuerStatistic,
    NpciSnapshot,
    Provenance,
)
from recovery.calibration.npci import CalibrationError, load_snapshot

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "external" / "npci"
FIXTURE_CSV = FIXTURE_DIR / "fixture_small.csv"
FIXTURE_PROVENANCE = FIXTURE_DIR / "fixture_small.provenance.yaml"


@pytest.fixture
def snapshot() -> NpciSnapshot:
    return load_snapshot(FIXTURE_CSV, FIXTURE_PROVENANCE)


class TestLoading:
    def test_loads_all_issuers(self, snapshot: NpciSnapshot) -> None:
        assert len(snapshot.issuers) == 4

    def test_provenance_is_populated(self, snapshot: NpciSnapshot) -> None:
        assert snapshot.provenance.reporting_period == "fixture"
        assert snapshot.provenance.retrieved_on.year == 2026

    def test_missing_csv_names_the_remedy(self, tmp_path: Path) -> None:
        with pytest.raises(CalibrationError, match="README"):
            load_snapshot(tmp_path / "absent.csv", FIXTURE_PROVENANCE)

    def test_missing_provenance_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(CalibrationError, match="provenance"):
            load_snapshot(FIXTURE_CSV, tmp_path / "absent.yaml")

    def test_missing_column_is_reported(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        bad.write_text("bank_name,td_rate\nX,0.01\n", encoding="utf-8")
        with pytest.raises(CalibrationError, match="missing columns"):
            load_snapshot(bad, FIXTURE_PROVENANCE)

    def test_duplicate_banks_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "dupe.csv"
        bad.write_text(
            "bank_name,td_rate,bd_rate,total_volume_mn\nA,0.008,0.04,100\nA,0.009,0.04,100\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate bank names"):
            load_snapshot(bad, FIXTURE_PROVENANCE)


class TestPublishedBounds:
    """The loader must reject snapshots that contradict published aggregates.

    This is the guard against the percent/rate confusion, which would
    otherwise be a silent 100x error in every downstream number.
    """

    def test_percent_mistaken_for_rate_is_caught(self, tmp_path: Path) -> None:
        bad = tmp_path / "percentish.csv"
        # 0.8 and 4.0 as *rates* means 80% and 400% decline. Nonsense, but the
        # kind of nonsense that arrives from a copy-paste.
        bad.write_text(
            "bank_name,td_rate,bd_rate,total_volume_mn\nA,0.8,4.0,100\n",
            encoding="utf-8",
        )
        with pytest.raises((CalibrationError, ValueError)):
            load_snapshot(bad, FIXTURE_PROVENANCE)

    def test_implausibly_clean_world_rejected(self, tmp_path: Path) -> None:
        """A world where nothing fails is not a calibration, it is a bug."""
        bad = tmp_path / "utopia.csv"
        bad.write_text(
            "bank_name,td_rate,bd_rate,total_volume_mn\nA,0.00001,0.00001,100\n",
            encoding="utf-8",
        )
        with pytest.raises(CalibrationError, match="outside the plausible"):
            load_snapshot(bad, FIXTURE_PROVENANCE)

    def test_fixture_sits_inside_published_bounds(self, snapshot: NpciSnapshot) -> None:
        lo, hi = priors.SYSTEM_TD_RATE_RANGE
        assert lo <= snapshot.volume_weighted_td <= hi
        slo, shi = priors.REMITTER_APPROVAL_RATE_RANGE
        assert slo <= snapshot.volume_weighted_success <= shi


class TestBaselineInversion:
    def test_degraded_fraction_matches_hand_calculation(self) -> None:
        # 0.08 episodes/day * 3.25h mean / 24h = 0.010833...
        fraction = degraded_time_fraction(0.08, (0.5, 6.0))
        assert fraction == pytest.approx(0.08 * 3.25 / 24.0)

    def test_baseline_is_below_published(self) -> None:
        baseline = solve_baseline_td(0.010, degraded_fraction=0.02, multiplier=10.0)
        assert baseline < 0.010

    def test_round_trips_to_published(self) -> None:
        """Baseline plus episodes must reproduce the published mean.

        This is the property the whole calibration rests on.
        """
        published = 0.0082
        fraction = 0.015
        multiplier = 12.0
        baseline = solve_baseline_td(published, fraction, multiplier)
        recovered = baseline * (1 - fraction + multiplier * fraction)
        assert recovered == pytest.approx(published)

    def test_no_degradation_leaves_rate_unchanged(self) -> None:
        published = 0.0082
        assert solve_baseline_td(published, 0.0, 10.0) == pytest.approx(published)

    def test_profile_rejects_baseline_above_published(self) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            IssuerProfile(
                bank_name="X",
                volume_share=1.0,
                published_td_rate=0.008,
                published_bd_rate=0.04,
                baseline_td_rate=0.020,
                degradation_episode_rate_per_day=0.08,
                degradation_td_multiplier=10.0,
            )


class TestCalibrate:
    def test_volume_shares_sum_to_one(self, snapshot: NpciSnapshot) -> None:
        params = calibrate(snapshot)
        assert sum(i.volume_share for i in params.issuers) == pytest.approx(1.0)

    def test_every_issuer_round_trips(self, snapshot: NpciSnapshot) -> None:
        params = calibrate(snapshot)
        fraction = degraded_time_fraction(
            assumptions.DEGRADATION_EPISODE_RATE_PER_DAY,
            assumptions.DEGRADATION_DURATION_HOURS_RANGE,
        )
        for profile in params.issuers:
            assert effective_td_rate(profile, degraded_fraction=fraction) == pytest.approx(
                profile.published_td_rate
            )

    def test_provenance_survives_into_parameters(self, snapshot: NpciSnapshot) -> None:
        """Parameters must carry their source. A parameter set that cannot
        say where it came from is not usable as evidence."""
        params = calibrate(snapshot)
        assert params.provenance == snapshot.provenance

    def test_assumption_overrides_change_baseline(self, snapshot: NpciSnapshot) -> None:
        """Phase 9's sweep depends on these overrides actually taking effect."""
        base = calibrate(snapshot)
        swept = calibrate(snapshot, td_multiplier=40.0)
        assert swept.issuers[0].baseline_td_rate < base.issuers[0].baseline_td_rate
        assert swept.issuers[0].published_td_rate == base.issuers[0].published_td_rate

    def test_tier1_cannot_be_overridden(self, snapshot: NpciSnapshot) -> None:
        """calibrate() exposes no keyword that alters published figures."""
        import inspect

        params = set(inspect.signature(calibrate).parameters)
        forbidden = {"td_rate", "bd_rate", "published_td_rate", "volume_share"}
        assert not (params & forbidden)

    def test_registry_covers_every_assumption(self) -> None:
        """An assumption not in the registry would be invisible to the sweep."""
        module_names = {
            name for name in vars(assumptions) if name.isupper() and name != "ASSUMPTION_REGISTRY"
        }
        assert module_names == set(assumptions.ASSUMPTION_REGISTRY)


class TestSnapshotAggregates:
    def test_weighted_td_respects_volume(self) -> None:
        """A tiny bank with terrible TD must not dominate the aggregate."""
        snapshot = NpciSnapshot(
            provenance=_fixture_provenance(),
            issuers=(
                IssuerStatistic(
                    bank_name="Big", td_rate=0.005, bd_rate=0.04, volume_millions=9900.0
                ),
                IssuerStatistic(bank_name="Tiny", td_rate=0.50, bd_rate=0.04, volume_millions=1.0),
            ),
        )
        assert snapshot.volume_weighted_td < 0.006

    def test_total_decline_over_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="no successful transactions"):
            IssuerStatistic(bank_name="X", td_rate=0.6, bd_rate=0.5, volume_millions=10.0)


def _fixture_provenance() -> Provenance:
    from datetime import date

    return Provenance(
        source_name="SYNTHETIC FIXTURE",
        source_url="https://example.invalid/fixture",
        reporting_period="fixture",
        retrieved_on=date(2026, 8, 26),
        retrieved_by="test suite",
    )


class TestRealDataCompatibility:
    """Regression guard for INC-006.

    Real NPCI figures for 2026-07 (top 20 remitter banks by volume) were
    rejected by the original bounds. These assertions pin the corrected
    bands to the observed values so the same mistake cannot recur silently.
    """

    OBSERVED_TD = 0.00376
    OBSERVED_BD = 0.10884
    OBSERVED_APPROVAL = 0.88737

    def test_observed_td_is_accepted(self) -> None:
        lo, hi = priors.SYSTEM_TD_RATE_RANGE
        assert lo <= self.OBSERVED_TD <= hi

    def test_observed_approval_is_accepted(self) -> None:
        lo, hi = priors.REMITTER_APPROVAL_RATE_RANGE
        assert lo <= self.OBSERVED_APPROVAL <= hi

    def test_merchant_band_is_not_used_for_remitter_data(self) -> None:
        """The two metrics have different denominators. Observed remitter
        approval falls outside the merchant band, which is exactly why
        conflating them broke calibration."""
        mlo, _ = priors.MERCHANT_P2M_SUCCESS_RANGE
        assert mlo > self.OBSERVED_APPROVAL

    def test_components_sum_to_one(self) -> None:
        total = self.OBSERVED_TD + self.OBSERVED_BD + self.OBSERVED_APPROVAL
        assert abs(total - 1.0) < 0.001

    def test_row_with_bad_component_sum_is_rejected(self) -> None:
        from recovery.calibration.models import IssuerStatistic

        with pytest.raises(ValueError, match="approved \\+ BD \\+ TD"):
            IssuerStatistic(
                bank_name="Mismatched",
                td_rate=0.0068,
                bd_rate=0.0902,
                volume_millions=100.0,
                approved_rate=0.50,
            )

    def test_real_row_passes(self) -> None:
        from recovery.calibration.models import IssuerStatistic

        sbi = IssuerStatistic(
            bank_name="State Bank of India",
            td_rate=0.0068,
            bd_rate=0.0902,
            volume_millions=6622.02,
            approved_rate=0.9029,
        )
        assert sbi.approved_rate == 0.9029
