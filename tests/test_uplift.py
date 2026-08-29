"""Uplift tests.

The claims under test:

* Qini's correction term does what it is supposed to, verified on constructed
  input rather than on a fitted model.
* A ranking that puts sleeping dogs first produces a negative curve. This is
  the property that makes Qini the right metric here.
* Uplift targeting beats risk targeting, and the segment breakdown explains
  why rather than merely asserting it.
* The X-learner beats the T-learner under arm imbalance, which is the reason
  it exists.
* No latent trait reaches the feature matrix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from recovery.calibration.models import WorldParameters
from recovery.evaluate.uplift_eval import evaluate_uplift, segment_capture
from recovery.uplift.features import FEATURE_NAMES, FeatureEncoder
from recovery.uplift.learners import TLearner, TreatmentData, XLearner
from recovery.uplift.metrics import qini_curve, uplift_at_k
from recovery.uplift.targeting import TargetingResult
from recovery.world.generate import ObservableBatch, OracleBatch, generate
from recovery.world.latent import LatentCustomer
from recovery.world.oracle.segments import Segment

PARAMS_PATH = Path(__file__).resolve().parents[1] / "configs" / "generator" / "world_params.json"

Batch = tuple[ObservableBatch, OracleBatch]


@pytest.fixture(scope="module")
def params() -> WorldParameters:
    return WorldParameters.model_validate_json(PARAMS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def batch(params: WorldParameters) -> Batch:
    return generate(params, n_cases=40000, seed=42)


@pytest.fixture(scope="module")
def results(batch: Batch) -> list[TargetingResult]:
    observable, oracle = batch
    scored, _ = evaluate_uplift(
        observable.features, observable.logged, observable.realized, oracle.outcomes
    )
    return scored


def _by_name(results: list[TargetingResult], name: str) -> TargetingResult:
    return next(r for r in results if r.name == name)


class TestQiniMetric:
    def test_correction_term_rescales_the_control_count(self) -> None:
        """Groups at a given depth are rarely the same size. Without the
        N_t/N_c correction, whichever arm the logging policy favoured looks
        better purely for being larger."""
        # 3 treated (2 recovered), 1 control (1 recovered), all at the top.
        scores = np.array([4.0, 3.0, 2.0, 1.0])
        treated = np.array([True, True, True, False])
        outcomes = np.array([1.0, 1.0, 0.0, 1.0])
        curve = qini_curve(scores, treated, outcomes)
        # Raw difference would be 2 - 1 = 1. Corrected: 2 - 1 * (3/1) = -1.
        assert curve.values[-1] == pytest.approx(-1.0)

    def test_perfect_ranking_beats_reversed(self) -> None:
        rng = np.random.default_rng(0)
        n = 2000
        true_uplift = rng.normal(0, 1, n)
        treated = rng.random(n) < 0.5
        base = 0.3
        prob = np.clip(base + treated * true_uplift * 0.1, 0.01, 0.99)
        outcomes = (rng.random(n) < prob).astype(float)

        good = qini_curve(true_uplift, treated, outcomes)
        bad = qini_curve(-true_uplift, treated, outcomes)
        assert good.coefficient > bad.coefficient

    def test_sleeping_dog_ranking_goes_negative(self) -> None:
        """A curve dipping below zero is a direct measurement of value
        destroyed by contacting people who would have paid anyway."""
        rng = np.random.default_rng(1)
        n = 1500
        harm = -np.abs(rng.normal(0, 1, n))  # everyone is harmed by contact
        treated = rng.random(n) < 0.5
        prob = np.clip(0.5 + treated * harm * 0.25, 0.01, 0.99)
        outcomes = (rng.random(n) < prob).astype(float)
        curve = qini_curve(-harm, treated, outcomes)  # rank most-harmed first
        assert curve.destroys_value

    def test_uplift_at_k_uses_only_the_top_slice(self) -> None:
        scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        treated = np.array([True, False, True, False, True, False])
        outcomes = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        top = uplift_at_k(scores, treated, outcomes, k=0.5)
        assert top == pytest.approx(1.0)

    def test_mismatched_lengths_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            qini_curve(np.array([1.0]), np.array([True, False]), np.array([1.0]))


class TestLearners:
    def test_treatment_data_validates_lengths(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            TreatmentData(x=np.zeros((3, 2)), y=np.zeros(2), propensity=np.ones(3))

    def test_single_class_arm_falls_back_to_base_rate(self) -> None:
        """Fitting a model on an arm with one outcome class would either raise
        or silently predict a constant that looks like a model. Falling back
        to the observed rate is honest and visible."""
        x = np.random.default_rng(0).normal(size=(200, 4))
        control = TreatmentData(x=x, y=np.zeros(200), propensity=np.full(200, 0.5))
        treated = TreatmentData(x=x, y=np.ones(200), propensity=np.full(200, 0.5))
        model = TLearner().fit(control, treated)
        assert model.mu0 is None
        assert model.mu1 is None
        uplift = model.predict_uplift(x)
        assert np.allclose(uplift, 1.0)

    def test_tiny_arm_is_not_fitted(self) -> None:
        x = np.random.default_rng(0).normal(size=(10, 4))
        y = np.array([0, 1] * 5, dtype=float)
        small = TreatmentData(x=x, y=y, propensity=np.full(10, 0.5))
        model = TLearner().fit(small, small)
        assert model.mu0 is None

    def test_x_learner_blends_by_propensity(self) -> None:
        """The blend is the point of the X-learner. Different propensities
        must produce different estimates, or the weighting is inert."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(400, 5))
        y_c = (rng.random(400) < 0.3).astype(float)
        y_t = (rng.random(400) < 0.6).astype(float)
        control = TreatmentData(x=x, y=y_c, propensity=np.full(400, 0.5))
        treated = TreatmentData(x=x, y=y_t, propensity=np.full(400, 0.5))
        model = XLearner().fit(control, treated)

        low = model.predict_uplift(x, np.full(400, 0.1))
        high = model.predict_uplift(x, np.full(400, 0.9))
        assert not np.allclose(low, high)


class TestFeatureIntegrity:
    def test_no_latent_traits_in_feature_names(self) -> None:
        """The import contract already forbids reaching into the world. This
        asserts the column set independently, so a leak would have to defeat
        both."""
        latent_fields = set(LatentCustomer.model_fields)
        assert not (set(FEATURE_NAMES) & latent_fields)
        for forbidden in ("uplift", "recovered", "segment", "recovery_prob"):
            assert not any(forbidden in name for name in FEATURE_NAMES)

    def test_matrix_shape_matches_names(self, batch: Batch) -> None:
        observable, _ = batch
        encoder = FeatureEncoder.fit(observable.features[:2000])
        matrix = encoder.transform(observable.features[:200])
        assert matrix.shape == (200, len(FEATURE_NAMES))

    def test_matrix_is_finite(self, batch: Batch) -> None:
        observable, _ = batch
        encoder = FeatureEncoder.fit(observable.features[:2000])
        assert np.isfinite(encoder.transform(observable.features[:500])).all()


class TestUpliftBeatsRisk:
    """Claim 1 of the README, measured on the randomised holdout."""

    def test_uplift_beats_risk_ranking(self, results: list[TargetingResult]) -> None:
        uplift = _by_name(results, "uplift_x_learner")
        risk = _by_name(results, "risk_ranking")
        assert uplift.coefficient > risk.coefficient

    def test_uplift_beats_random(self, results: list[TargetingResult]) -> None:
        uplift = _by_name(results, "uplift_x_learner")
        random = _by_name(results, "random")
        assert uplift.coefficient > random.coefficient

    def test_oracle_is_the_ceiling(self, results: list[TargetingResult]) -> None:
        """Nothing may beat the true-uplift ranking. If something does, the
        model is reading something it should not be able to see."""
        oracle = _by_name(results, "oracle_uplift")
        for result in results:
            assert result.coefficient <= oracle.coefficient + 1e-9

    def test_x_learner_beats_t_learner(self, results: list[TargetingResult]) -> None:
        """The reason the X-learner exists: the control arm is starved by the
        logging policy, and cross-fitting recovers signal the T-learner loses
        to that arm's noise."""
        x_learner = _by_name(results, "uplift_x_learner")
        t_learner = _by_name(results, "uplift_t_learner")
        assert x_learner.coefficient > t_learner.coefficient

    def test_uplift_recovers_more_at_a_realistic_budget(
        self, results: list[TargetingResult]
    ) -> None:
        """The operational question: with contact capacity for 30% of cases,
        which ranking recovers more?

        Stated at a fixed depth rather than as a comparison of curve minima.
        An earlier version compared `min_value`, which is dominated by the
        first few points where the curve rests on single observations, and it
        flipped between batch sizes (INC-015)."""
        uplift = _by_name(results, "uplift_x_learner")
        risk = _by_name(results, "risk_ranking")
        assert uplift.uplift_at_30 > risk.uplift_at_30


class TestSegmentCapture:
    """Why uplift wins, not just that it does."""

    def test_uplift_captures_more_persuadables(self, batch: Batch) -> None:
        observable, oracle = batch
        capture = segment_capture(
            observable.features, observable.logged, observable.realized, oracle.outcomes
        )
        assert capture["uplift"][Segment.PERSUADABLE] > capture["risk"][Segment.PERSUADABLE]

    def test_risk_ranking_fills_its_budget_with_lost_causes(self, batch: Batch) -> None:
        """The specific failure mode. A lost cause looks maximally risky and
        is maximally unhelpful to contact."""
        observable, oracle = batch
        capture = segment_capture(
            observable.features, observable.logged, observable.realized, oracle.outcomes
        )
        risk_total = sum(capture["risk"].values())
        assert capture["risk"][Segment.LOST_CAUSE] / risk_total > 0.30
        uplift_total = sum(capture["uplift"].values())
        assert capture["uplift"][Segment.LOST_CAUSE] / uplift_total < 0.20
