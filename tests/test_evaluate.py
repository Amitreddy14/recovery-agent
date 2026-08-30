"""Evaluation-layer tests.

These cover the code that validates everything else. A silent estimator bug
here would corrupt the argument the whole synthetic-world approach rests on,
and would look fine while doing it — the estimate would simply be wrong, with
no error raised anywhere.

Expected values are computed by hand in the docstrings rather than copied
from the implementation, so the assertions check arithmetic rather than
restate it.
"""

from __future__ import annotations

import pytest

from recovery.domain.enums import ActionType, FailureReason
from recovery.evaluate.ope import (
    MIN_PROPENSITY,
    doubly_robust,
    effective_sample_size,
    fit_reward_model,
    ips,
    snips,
    true_policy_value,
    validate_estimators,
)
from recovery.evaluate.sweep import (
    SweepPoint,
    SweepResult,
    SweepSummary,
    default_grid,
    perturb,
    sample_grid,
)

A = ActionType.RETRY_NOW
B = ActionType.SEND_PAYMENT_LINK
C = ActionType.NO_ACTION

# The worked example used throughout:
#
#   rewards      100    200    300
#   logged        A      B      A
#   propensity   0.5   0.25    0.5
#   target        A      A      A
#
#   weights = [1/0.5, 0 (disagrees), 1/0.5] = [2, 0, 2]
#
#   IPS   = (2*100 + 0*200 + 2*300) / 3      = 800/3 = 266.667
#   SNIPS = (2*100 + 0*200 + 2*300) / (2+0+2) = 800/4 = 200.0
#   ESS   = (2+0+2)^2 / (4+0+4)               = 16/8  = 2.0
REWARDS = [100.0, 200.0, 300.0]
LOGGED = [A, B, A]
PROPS = [0.5, 0.25, 0.5]
TARGET = [A, A, A]


class TestIPS:
    def test_matches_hand_calculation(self) -> None:
        assert ips(REWARDS, LOGGED, PROPS, TARGET) == pytest.approx(800 / 3)

    def test_disagreeing_cases_contribute_nothing(self) -> None:
        """Where the target would not have taken the logged action we have no
        evidence about its reward, and inventing some is precisely what the
        estimator must not do."""
        louder = [100.0, 10_000_000.0, 300.0]
        assert ips(louder, LOGGED, PROPS, TARGET) == pytest.approx(
            ips(REWARDS, LOGGED, PROPS, TARGET)
        )

    def test_full_agreement_at_unit_propensity_is_the_plain_mean(self) -> None:
        assert ips(REWARDS, TARGET, [1.0, 1.0, 1.0], TARGET) == pytest.approx(200.0)

    def test_rare_actions_are_upweighted(self) -> None:
        """A case logged at propensity 0.1 stands in for ten cases."""
        assert ips([100.0], [A], [0.1], [A]) == pytest.approx(1000.0)

    def test_propensity_is_clipped(self) -> None:
        """Without a floor, one mis-logged near-zero propensity produces a
        weight in the thousands and the estimate becomes that single case."""
        clipped = ips([100.0], [A], [1e-12], [A])
        assert clipped == pytest.approx(100.0 / MIN_PROPENSITY)

    def test_no_agreement_yields_zero(self) -> None:
        assert ips(REWARDS, [B, B, B], PROPS, TARGET) == 0.0


class TestSNIPS:
    def test_matches_hand_calculation(self) -> None:
        assert snips(REWARDS, LOGGED, PROPS, TARGET) == pytest.approx(200.0)

    def test_differs_from_ips_under_partial_agreement(self) -> None:
        """IPS divides by n, SNIPS by the weight mass it actually observed.
        They coincide only when the two are equal."""
        assert snips(REWARDS, LOGGED, PROPS, TARGET) != pytest.approx(
            ips(REWARDS, LOGGED, PROPS, TARGET)
        )

    def test_no_agreement_yields_zero_not_a_division_error(self) -> None:
        assert snips(REWARDS, [B, B, B], PROPS, TARGET) == 0.0

    def test_is_scale_invariant_in_weights(self) -> None:
        """Halving every propensity doubles every weight, which SNIPS
        normalises away. That invariance is the whole point of it."""
        halved = [p / 2 for p in PROPS]
        assert snips(REWARDS, LOGGED, halved, TARGET) == pytest.approx(
            snips(REWARDS, LOGGED, PROPS, TARGET)
        )


class TestDoublyRobust:
    def test_matches_hand_calculation(self) -> None:
        model = {A: 200.0, B: 200.0}
        assert doubly_robust(REWARDS, LOGGED, PROPS, TARGET, model) == pytest.approx(200.0)

    def test_reduces_to_ips_with_a_zero_reward_model(self) -> None:
        """With no model, the residual term is the reward itself and DR
        collapses to IPS. A DR implementation that fails this is not DR."""
        zero: dict[ActionType, float] = {}
        assert doubly_robust(REWARDS, LOGGED, PROPS, TARGET, zero) == pytest.approx(
            ips(REWARDS, LOGGED, PROPS, TARGET)
        )

    def test_falls_back_to_the_model_with_no_agreement(self) -> None:
        """Where there is no overlap the IPS correction vanishes and DR
        returns the model's prediction — which is the property that makes it
        usable on logs where IPS returns zero."""
        model = {A: 175.0, B: 50.0}
        assert doubly_robust(REWARDS, [B, B, B], PROPS, TARGET, model) == pytest.approx(175.0)

    def test_unknown_action_in_model_defaults_to_zero(self) -> None:
        result = doubly_robust(REWARDS, LOGGED, PROPS, TARGET, {A: 100.0})
        assert isinstance(result, float)


class TestRewardModel:
    def test_averages_per_action(self) -> None:
        model = fit_reward_model(REWARDS, LOGGED)
        assert model[A] == pytest.approx(200.0)
        assert model[B] == pytest.approx(200.0)

    def test_only_includes_observed_actions(self) -> None:
        """An action never logged has no evidence and must not acquire a
        fabricated mean."""
        assert C not in fit_reward_model(REWARDS, LOGGED)


class TestEffectiveSampleSize:
    def test_matches_hand_calculation(self) -> None:
        assert effective_sample_size(LOGGED, PROPS, TARGET) == pytest.approx(2.0)

    def test_equals_n_under_uniform_full_agreement(self) -> None:
        assert effective_sample_size(TARGET, [0.5, 0.5, 0.5], TARGET) == pytest.approx(3.0)

    def test_collapses_when_one_case_dominates(self) -> None:
        """This is the diagnostic the number exists for: a low ESS means the
        estimate rests on a handful of heavily-weighted cases, so a value that
        happens to land close is not evidence of accuracy."""
        skewed = effective_sample_size([A, A], [0.001, 1.0], [A, A])
        assert skewed < 1.1

    def test_zero_when_nothing_agrees(self) -> None:
        assert effective_sample_size([B, B, B], PROPS, TARGET) == 0.0


class TestValidationIsNotCircular:
    def test_estimators_receive_no_ground_truth(self) -> None:
        """Structural check. If an estimator could see the truth it is scored
        against, every reported error would be meaningless."""
        import inspect

        for fn in (ips, snips):
            params = set(inspect.signature(fn).parameters)
            assert not (params & {"outcomes", "truth", "potential_outcomes"})

    def test_truth_reads_the_counterfactual(self) -> None:
        """Available only because this is a simulated world — and the reason
        the whole exercise is possible."""
        from recovery.world.oracle.outcomes import PotentialOutcomes

        outcome = PotentialOutcomes(
            case_id="c1",
            recovery_prob={A: 0.9, B: 0.1},
            recovered={A: True, B: False},
            mandate_cancelled={A: False, B: False},
            uniform_draw=0.5,
            cancel_draw=0.5,
        )
        assert true_policy_value([outcome], [50000], [A]) == pytest.approx(50000.0)
        assert true_policy_value([outcome], [50000], [B]) == pytest.approx(0.0)

    def test_validate_returns_all_three_estimators(self) -> None:
        from recovery.world.oracle.outcomes import PotentialOutcomes

        outcomes = [
            PotentialOutcomes(
                case_id=f"c{i}",
                recovery_prob={A: 0.9, B: 0.1},
                recovered={A: True, B: False},
                mandate_cancelled={A: False, B: False},
                uniform_draw=0.5,
                cancel_draw=0.5,
            )
            for i in range(3)
        ]
        results = validate_estimators(
            outcomes=outcomes,
            amounts_paise=[100, 200, 300],
            logged_actions=LOGGED,
            propensities=PROPS,
            target_actions=TARGET,
        )
        assert {r.estimator for r in results} == {"ips", "snips", "doubly_robust"}
        assert all(r.n == 3 for r in results)


class TestPerturb:
    def _point(self, **kw: float) -> SweepPoint:
        base: dict[str, float] = {
            "cancellation_horizon": 12,
            "insufficient_funds_share": 0.42,
            "cancelled_share": 0.22,
            "mandate_bd_uplift": 1.35,
            "salary_window_multiplier": 0.45,
        }
        base.update(kw)
        return SweepPoint(**base)  # type: ignore[arg-type]

    @pytest.fixture
    def params(self):  # type: ignore[no-untyped-def]
        from pathlib import Path

        from recovery.calibration.models import WorldParameters

        path = Path(__file__).resolve().parents[1] / "configs" / "generator" / "world_params.json"
        return WorldParameters.model_validate_json(path.read_text(encoding="utf-8"))

    def test_tier1_values_are_untouched(self, params) -> None:  # type: ignore[no-untyped-def]
        """The sweep moves assumptions, never published data (ADR-0006). If
        this fails, the sweep is perturbing the NPCI snapshot and the
        calibration claim is void."""
        swept = perturb(params, self._point())
        for before, after in zip(params.issuers, swept.issuers, strict=True):
            assert before.published_td_rate == after.published_td_rate
            assert before.published_bd_rate == after.published_bd_rate
            assert before.volume_share == after.volume_share

    def test_named_shares_are_set_exactly(self, params) -> None:  # type: ignore[no-untyped-def]
        swept = perturb(params, self._point(insufficient_funds_share=0.5, cancelled_share=0.3))
        assert swept.bd_reason_mix[FailureReason.INSUFFICIENT_FUNDS] == pytest.approx(0.5)
        assert swept.bd_reason_mix[FailureReason.PAYMENT_CANCELLED_BY_USER] == pytest.approx(0.3)

    def test_mix_still_sums_to_one(self, params) -> None:  # type: ignore[no-untyped-def]
        swept = perturb(params, self._point(insufficient_funds_share=0.5, cancelled_share=0.3))
        assert sum(swept.bd_reason_mix.values()) == pytest.approx(1.0)

    def test_unnamed_reasons_keep_their_relative_proportions(self, params) -> None:  # type: ignore[no-untyped-def]
        """Rescaling everything uniformly would move assumptions the sweep
        point did not name, making the result unattributable."""
        fixed = {FailureReason.INSUFFICIENT_FUNDS, FailureReason.PAYMENT_CANCELLED_BY_USER}
        others = [r for r in params.bd_reason_mix if r not in fixed]
        before_ratio = params.bd_reason_mix[others[0]] / params.bd_reason_mix[others[1]]
        swept = perturb(params, self._point(insufficient_funds_share=0.5, cancelled_share=0.3))
        after_ratio = swept.bd_reason_mix[others[0]] / swept.bd_reason_mix[others[1]]
        assert before_ratio == pytest.approx(after_ratio)

    def test_impossible_point_is_rejected(self, params) -> None:  # type: ignore[no-untyped-def]
        """Shares summing above 1.0 leave no mass for anything else. Silently
        renormalising would produce a world that is not the one requested."""
        with pytest.raises(ValueError, match="no mass"):
            perturb(params, self._point(insufficient_funds_share=0.7, cancelled_share=0.5))

    def test_scalar_assumptions_are_applied(self, params) -> None:  # type: ignore[no-untyped-def]
        swept = perturb(params, self._point(mandate_bd_uplift=2.0, salary_window_multiplier=0.9))
        assert swept.mandate_bd_uplift_factor == pytest.approx(2.0)
        assert swept.salary_window_bd_multiplier == pytest.approx(0.9)

    def test_provenance_survives(self, params) -> None:  # type: ignore[no-untyped-def]
        """A swept world must still be able to say where its empirical half
        came from."""
        assert perturb(params, self._point()).provenance == params.provenance


class TestGrid:
    def test_sampling_is_reproducible(self) -> None:
        """A random subset that cannot be re-run is not evidence."""
        grid = default_grid()
        assert [p.label() for p in sample_grid(grid, 8, seed=3)] == [
            p.label() for p in sample_grid(grid, 8, seed=3)
        ]

    def test_different_seeds_sample_differently(self) -> None:
        grid = default_grid()
        assert [p.label() for p in sample_grid(grid, 8, seed=1)] != [
            p.label() for p in sample_grid(grid, 8, seed=2)
        ]

    def test_requesting_more_than_available_returns_all(self) -> None:
        grid = default_grid()
        assert len(sample_grid(grid, len(grid) * 2, seed=0)) == len(grid)

    def test_grid_spans_the_declared_ranges(self) -> None:
        grid = default_grid()
        assert {p.cancellation_horizon for p in grid} == {3, 12, 24}
        assert len({p.cancelled_share for p in grid}) == 3


class TestSweepSummary:
    def _result(self, uplift: int, risk: int, i: int = 0) -> SweepResult:
        return SweepResult(
            point=SweepPoint(12, 0.42, 0.22, 1.35, 0.45),
            seed=i,
            net_by_policy={"uplift_ev": uplift, "risk_topN": risk},
        )

    def test_win_rate(self) -> None:
        summary = SweepSummary(
            "uplift_ev",
            "risk_topN",
            [self._result(100, 50), self._result(100, 150), self._result(100, 90)],
        )
        assert summary.wins == 2
        assert summary.win_rate == pytest.approx(2 / 3)

    def test_worst_delta_can_be_negative(self) -> None:
        """The sweep must be able to report a loss. A summary that cannot is
        a presentation, not an analysis."""
        summary = SweepSummary(
            "uplift_ev", "risk_topN", [self._result(100, 50), self._result(100, 500)]
        )
        assert summary.worst_delta_paise == -400

    def test_median_of_even_count(self) -> None:
        summary = SweepSummary(
            "uplift_ev",
            "risk_topN",
            [self._result(100, 90), self._result(100, 80)],
        )
        assert summary.median_delta_paise == pytest.approx(15.0)

    def test_losing_points_are_enumerated(self) -> None:
        summary = SweepSummary(
            "uplift_ev", "risk_topN", [self._result(100, 50), self._result(100, 150)]
        )
        assert len(summary.losing_points()) == 1

    def test_no_losses_is_reported_as_a_caveat_not_a_victory(self) -> None:
        """An approach that wins everywhere usually means the grid was too
        narrow, and the description says so rather than implying triumph."""
        summary = SweepSummary("uplift_ev", "risk_topN", [self._result(100, 50)])
        assert "widen" in summary.describe_losses()
