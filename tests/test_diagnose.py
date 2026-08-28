"""Diagnosis tests.

The central claims under test:

* The taxonomy separates FUNDING from INTENT. Conflating them is the naive
  policy's costliest error.
* The Bayesian detector does not fire on thin evidence, and the fixed rule
  does. This is the whole argument for modelling uncertainty.
* Nothing in `diagnose` can reach the world.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from recovery.calibration.models import WorldParameters
from recovery.diagnose.engine import DiagnosisEngine
from recovery.diagnose.issuer_health import (
    IssuerHealthModel,
    ThresholdHealthModel,
)
from recovery.diagnose.taxonomy import (
    Recoverability,
    classify,
    is_silently_recoverable,
    is_terminal,
)
from recovery.domain.enums import (
    CaseType,
    DeclineClass,
    FailureReason,
    PaymentMethod,
)
from recovery.domain.observations import CaseFeatures
from recovery.evaluate.diagnosis_eval import (
    DetectorScore,
    evaluate_health_models,
    score_detector,
)
from recovery.world.generate import ObservableBatch, OracleBatch, generate

PARAMS_PATH = Path(__file__).resolve().parents[1] / "configs" / "generator" / "world_params.json"

Batch = tuple[ObservableBatch, OracleBatch]
"""Return type of the module-scoped `batch` fixture.

Named so every test can annotate its parameter directly. An earlier version
suppressed the missing annotation instead, which placed the ignore comment on
the parameter line of a wrapped signature while mypy reports the error on the
`def` line — producing 27 errors at once, half "missing annotation" and half
"unused ignore" (INC-011).
"""


@pytest.fixture(scope="module")
def params() -> WorldParameters:
    return WorldParameters.model_validate_json(PARAMS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def batch(params: WorldParameters) -> Batch:
    return generate(params, n_cases=6000, seed=42)


def _features(**kw: Any) -> CaseFeatures:
    base: dict[str, Any] = {
        "case_id": "c1",
        "case_type": CaseType.PAYMENT_FAILURE,
        "created_at": datetime(2026, 7, 5, 12, tzinfo=UTC),
        "amount_paise": 199900,
        "method": PaymentMethod.UPI,
        "issuer": "State Bank of India",
        "reason": FailureReason.BANK_DOWNTIME,
        "decline_class": DeclineClass.TECHNICAL,
        "customer_id": "cust_1",
        "tenure_days": 400,
        "prior_payment_count": 20,
        "prior_failure_count": 3,
        "prior_recovery_count": 2,
        "contacts_last_30d": 0,
        "dnd_registered": False,
        "hour_of_day": 12,
        "day_of_month": 5,
        "days_since_salary": 4,
        "issuer_failures_last_hour": 2,
        "issuer_volume_last_hour": 300,
    }
    base.update(kw)
    return CaseFeatures(**base)


class TestTaxonomy:
    def test_funding_and_intent_are_separated(self) -> None:
        """Insufficient funds means 'cannot pay now'. Cancellation means
        'chose not to pay'. Same observable failure, opposite remedies."""
        assert classify(FailureReason.INSUFFICIENT_FUNDS) is Recoverability.FUNDING
        assert classify(FailureReason.PAYMENT_CANCELLED_BY_USER) is Recoverability.INTENT

    def test_intent_is_not_silently_recoverable(self) -> None:
        assert not is_silently_recoverable(FailureReason.PAYMENT_CANCELLED_BY_USER)

    def test_funding_is_silently_recoverable(self) -> None:
        """A retry at the right moment costs no customer contact."""
        assert is_silently_recoverable(FailureReason.INSUFFICIENT_FUNDS)

    def test_revoked_mandate_is_terminal(self) -> None:
        assert is_terminal(FailureReason.MANDATE_REVOKED)
        assert not is_silently_recoverable(FailureReason.MANDATE_REVOKED)

    def test_expired_card_is_not_silently_recoverable(self) -> None:
        """Retrying a dead instrument cannot work regardless of timing."""
        assert classify(FailureReason.CARD_EXPIRED) is Recoverability.INSTRUMENT
        assert not is_silently_recoverable(FailureReason.CARD_EXPIRED)

    def test_unmapped_reason_is_flagged_not_guessed(self) -> None:
        assert classify(FailureReason.OTHER) is Recoverability.UNKNOWN


class TestIssuerHealthUncertainty:
    """The core argument: a fixed threshold is most confident where the
    evidence is weakest."""

    def test_thin_evidence_does_not_trigger_bayesian(
        self,
        batch: Batch,
    ) -> None:
        observable, _ = batch
        model = IssuerHealthModel().fit(observable.features)
        thin = _features(issuer_failures_last_hour=1, issuer_volume_last_hour=20)
        assessment = model.assess(thin)
        assert not assessment.is_degraded
        assert assessment.degradation_probability < 0.7

    def test_thin_evidence_does_trigger_fixed_rule(self) -> None:
        """1 failure in 20 is a 5% rate. The rule fires on a single event."""
        thin = _features(issuer_failures_last_hour=1, issuer_volume_last_hour=20)
        assert ThresholdHealthModel().assess(thin).is_degraded

    def test_strong_evidence_triggers_bayesian(
        self,
        batch: Batch,
    ) -> None:
        observable, _ = batch
        model = IssuerHealthModel().fit(observable.features)
        strong = _features(issuer_failures_last_hour=40, issuer_volume_last_hour=400)
        assert model.assess(strong).is_degraded

    def test_posterior_sharpens_with_volume(
        self,
        batch: Batch,
    ) -> None:
        """Same observed rate, more evidence, higher confidence."""
        observable, _ = batch
        model = IssuerHealthModel().fit(observable.features)
        small = model.assess(_features(issuer_failures_last_hour=2, issuer_volume_last_hour=25))
        large = model.assess(_features(issuer_failures_last_hour=32, issuer_volume_last_hour=400))
        assert large.degradation_probability > small.degradation_probability

    def test_shrinkage_weight_scales_with_volume(
        self,
        batch: Batch,
    ) -> None:
        observable, _ = batch
        model = IssuerHealthModel().fit(observable.features)
        baselines = [
            b
            for issuer in {f.issuer for f in observable.features}
            if (b := model.baseline_for(issuer)) is not None
        ]
        ranked = sorted(baselines, key=lambda b: b.observed_volume)
        assert ranked[0].shrinkage_weight < ranked[-1].shrinkage_weight

    def test_unseen_issuer_falls_back_to_population(
        self,
        batch: Batch,
    ) -> None:
        observable, _ = batch
        model = IssuerHealthModel().fit(observable.features)
        assessment = model.assess(_features(issuer="Nonexistent Bank Ltd"))
        assert assessment.baseline_rate == pytest.approx(model.population_rate)

    def test_fit_is_required_before_assess(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            IssuerHealthModel().assess(_features())


class TestDetectorBeatsRuleOnCost:
    def test_bayesian_has_lower_expected_cost(
        self,
        batch: Batch,
    ) -> None:
        observable, oracle = batch
        scores, naive = evaluate_health_models(observable.features, oracle.outcomes)
        best = min(scores, key=lambda s: s.weighted_cost)
        assert best.weighted_cost < naive.weighted_cost

    def test_rule_over_fires_on_low_volume(
        self,
        batch: Batch,
    ) -> None:
        """The predicted failure mode, measured directly."""
        observable, oracle = batch
        truth = {o.case_id: o.issuer_degraded_at_failure for o in oracle.outcomes}
        low = [f for f in observable.features if f.issuer_volume_last_hour < 100]
        assert low

        bayes = IssuerHealthModel(decision_threshold=0.8).fit(observable.features)
        rule = ThresholdHealthModel()
        truths = [truth[f.case_id] for f in low]

        b = score_detector("b", [bayes.assess(f) for f in low], truths)
        r = score_detector("r", [rule.assess(f) for f in low], truths)
        assert r.positive_rate > b.positive_rate
        assert b.precision > r.precision

    def test_cost_penalises_missed_degradation_more_than_false_alarms(
        self,
    ) -> None:
        """The two errors are not equal and the objective must say so.

        Asserted on constructed scores rather than on a fitted threshold: an
        earlier version compared the F1-optimal and cost-optimal thresholds
        directly, which agreed at n=6000 and diverged at n=8000. That made
        the test a statement about sample size rather than about the
        objective (INC-009).
        """
        balanced = DetectorScore(
            name="fp_heavy",
            n=1000,
            true_positives=50,
            false_positives=40,
            true_negatives=900,
            false_negatives=10,
        )
        fn_heavy = DetectorScore(
            name="fn_heavy",
            n=1000,
            true_positives=50,
            false_positives=10,
            true_negatives=900,
            false_negatives=40,
        )
        assert balanced.f1 == pytest.approx(fn_heavy.f1)
        assert balanced.weighted_cost < fn_heavy.weighted_cost

    def test_cost_optimum_is_interior(
        self,
        batch: Batch,
    ) -> None:
        """Neither firing constantly nor never firing should win. An optimum
        at an endpoint would mean the threshold is not doing real work."""
        observable, oracle = batch
        scores, _ = evaluate_health_models(observable.features, oracle.outcomes)
        best = min(scores, key=lambda s: s.weighted_cost)
        assert best.threshold not in (scores[0].threshold, scores[-1].threshold)


class TestDiagnosisEngine:
    def test_terminal_case_is_not_silently_recoverable(
        self,
        batch: Batch,
    ) -> None:
        observable, _ = batch
        engine = DiagnosisEngine.fit(observable.features)
        d = engine.diagnose(
            _features(
                reason=FailureReason.MANDATE_REVOKED,
                decline_class=DeclineClass.BUSINESS,
            )
        )
        assert d.root_cause == "terminal"
        assert not d.recoverable_without_contact

    def test_inconsistent_record_is_surfaced_not_diagnosed(
        self,
        batch: Batch,
    ) -> None:
        """A reason that contradicts its decline class means upstream data
        corruption. Diagnosing confidently from it would be worse than
        admitting we cannot."""
        observable, _ = batch
        engine = DiagnosisEngine.fit(observable.features)
        d = engine.diagnose(
            _features(
                reason=FailureReason.INSUFFICIENT_FUNDS,
                decline_class=DeclineClass.TECHNICAL,
            )
        )
        assert d.root_cause == "inconsistent_record"
        assert d.confidence < 0.3
        assert any("INCONSISTENT" in e for e in d.evidence)

    def test_every_diagnosis_carries_evidence(
        self,
        batch: Batch,
    ) -> None:
        observable, _ = batch
        engine = DiagnosisEngine.fit(observable.features)
        for feature in observable.features[:500]:
            assert engine.diagnose(feature).evidence

    def test_upcoming_cases_are_diagnosed_as_prevention(
        self,
        batch: Batch,
    ) -> None:
        observable, _ = batch
        engine = DiagnosisEngine.fit(observable.features)
        upcoming = next(f for f in observable.features if f.case_type is CaseType.UPCOMING_AT_RISK)
        d = engine.diagnose(upcoming)
        assert d.root_cause == "upcoming_debit"
        assert not d.issuer_degraded

    def test_degradation_only_considered_for_transient(
        self,
        batch: Batch,
    ) -> None:
        """Issuer health is irrelevant to an expired card. Checking it anyway
        would add a spurious reason to the audit trail."""
        observable, _ = batch
        engine = DiagnosisEngine.fit(observable.features)
        d = engine.diagnose(
            _features(
                reason=FailureReason.CARD_EXPIRED,
                decline_class=DeclineClass.BUSINESS,
                issuer_failures_last_hour=200,
                issuer_volume_last_hour=300,
            )
        )
        assert not d.issuer_degraded
