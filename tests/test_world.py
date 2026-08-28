"""World generator tests.

These fall into three groups:

* **Integrity** - the oracle stays quarantined, on disk and in imports.
* **Determinism** - same seed, same world, everywhere.
* **Causal structure** - the mechanisms the policy is supposed to discover
  actually exist in the data. If `test_degraded_issuer_suppresses_retry`
  fails, the retry-timing thesis is untestable and Phase 5 is wasted work.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from recovery.calibration.models import WorldParameters
from recovery.domain.enums import ActionType, CaseType, FailureReason
from recovery.world.cases import (
    EXPLORATION_EPSILON,
    HOLDOUT_FRACTION,
    LOGGABLE_ACTIONS,
)
from recovery.world.generate import generate, write_batch
from recovery.world.latent import LatentCustomer, sample_latent
from recovery.world.oracle.outcomes import compute_potential_outcomes
from recovery.world.oracle.segments import Segment, classify
from recovery.world.timeline import IssuerTimeline

PARAMS_PATH = Path(__file__).resolve().parents[1] / "configs" / "generator" / "world_params.json"


@pytest.fixture(scope="module")
def params() -> WorldParameters:
    return WorldParameters.model_validate_json(PARAMS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def batch(params: WorldParameters):  # type: ignore[no-untyped-def]
    return generate(params, n_cases=3000, seed=42)


class TestDeterminism:
    def test_same_seed_same_world(self, params: WorldParameters) -> None:
        a, _ = generate(params, n_cases=200, seed=7)
        b, _ = generate(params, n_cases=200, seed=7)
        assert [f.model_dump_json() for f in a.features] == [
            f.model_dump_json() for f in b.features
        ]

    def test_different_seed_different_world(self, params: WorldParameters) -> None:
        a, _ = generate(params, n_cases=200, seed=7)
        b, _ = generate(params, n_cases=200, seed=8)
        assert a.features[0].model_dump_json() != b.features[0].model_dump_json()

    def test_oracle_is_deterministic_too(self, params: WorldParameters) -> None:
        _, a = generate(params, n_cases=200, seed=7)
        _, b = generate(params, n_cases=200, seed=7)
        assert [o.uniform_draw for o in a.outcomes] == [o.uniform_draw for o in b.outcomes]


class TestEvaluationIntegrity:
    def test_features_carry_no_outcome_fields(self, batch) -> None:  # type: ignore[no-untyped-def]
        observable, _ = batch
        leaky = {"recovered", "recovery_prob", "uplift", "segment", "intent", "latent"}
        fields = set(observable.features[0].model_dump())
        assert not (fields & leaky)

    def test_features_carry_no_latent_traits(self, batch) -> None:  # type: ignore[no-untyped-def]
        observable, _ = batch
        fields = set(observable.features[0].model_dump())
        latent_fields = set(LatentCustomer.model_fields)
        assert not (fields & latent_fields)

    def test_oracle_written_to_separate_directory(
        self, params: WorldParameters, tmp_path: Path
    ) -> None:
        observable, oracle = generate(params, n_cases=50, seed=1)
        cases_p, _logged_p, oracle_p = write_batch(observable, oracle, tmp_path)
        assert oracle_p.parent.name == "oracle"
        assert cases_p.parent != oracle_p.parent
        # A glob over the batch directory must not reach the oracle.
        assert oracle_p not in set(tmp_path.glob("*.jsonl"))

    def test_manifest_warns_about_oracle(self, params: WorldParameters, tmp_path: Path) -> None:
        observable, oracle = generate(params, n_cases=20, seed=1)
        write_batch(observable, oracle, tmp_path)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert "oracle" in manifest["warning"].lower()
        assert manifest["calibration_provenance"]


class TestLoggedPropensities:
    """Off-policy evaluation divides by these. A wrong value biases every
    downstream estimate silently rather than raising an error."""

    def test_all_propensities_are_positive(self, batch) -> None:  # type: ignore[no-untyped-def]
        observable, _ = batch
        assert all(d.propensity > 0 for d in observable.logged)

    def test_holdout_propensity_is_exactly_uniform(self, batch) -> None:  # type: ignore[no-untyped-def]
        observable, _ = batch
        expected = 1.0 / len(LOGGABLE_ACTIONS)
        holdout = [d for d in observable.logged if d.is_holdout]
        assert holdout
        assert all(d.propensity == pytest.approx(expected) for d in holdout)

    def test_holdout_share_matches_target(self, batch) -> None:  # type: ignore[no-untyped-def]
        observable, _ = batch
        share = sum(d.is_holdout for d in observable.logged) / len(observable.logged)
        assert share == pytest.approx(HOLDOUT_FRACTION, abs=0.03)

    def test_every_action_is_explored(self, batch) -> None:  # type: ignore[no-untyped-def]
        """An action never taken has no logged data, so its value can never
        be estimated. Full support is a precondition for OPE, not a nicety."""
        observable, _ = batch
        taken = {d.action for d in observable.logged}
        assert set(LOGGABLE_ACTIONS) <= taken

    def test_greedy_propensity_accounts_for_both_paths(self, batch) -> None:  # type: ignore[no-untyped-def]
        """The preferred action is reachable via exploitation *and* via the
        random exploration branch. Ignoring the second path understates its
        propensity and inflates its IPS weight."""
        observable, _ = batch
        n = len(LOGGABLE_ACTIONS)
        expected_max = ((1.0 - EXPLORATION_EPSILON) + EXPLORATION_EPSILON / n) * (
            1.0 - HOLDOUT_FRACTION
        )
        non_holdout = [d.propensity for d in observable.logged if not d.is_holdout]
        assert max(non_holdout) == pytest.approx(expected_max)


class TestSegmentsEmerge:
    """Segments must arise from feature x action interactions, never from a
    label. All four must be present at usable prevalence or the uplift work
    in Phase 5 has nothing to find."""

    def test_all_four_segments_present(self, batch) -> None:  # type: ignore[no-untyped-def]
        _, oracle = batch
        counts = Counter(classify(o) for o in oracle.outcomes)
        for segment in (
            Segment.SURE_THING,
            Segment.LOST_CAUSE,
            Segment.PERSUADABLE,
            Segment.SLEEPING_DOG,
        ):
            assert counts[segment] > 0, f"{segment} absent - uplift is untestable"

    def test_sleeping_dogs_are_material(self, batch) -> None:  # type: ignore[no-untyped-def]
        """The headline claim is that naive contact destroys value. If
        sleeping dogs are rare, that claim is not demonstrable."""
        _, oracle = batch
        counts = Counter(classify(o) for o in oracle.outcomes)
        assert counts[Segment.SLEEPING_DOG] / len(oracle.outcomes) > 0.04

    def test_sure_things_are_pre_failure_only(self, batch) -> None:  # type: ignore[no-untyped-def]
        """Structural property, discovered rather than designed (INC-007):
        once a payment has failed, some action always beats doing nothing, so
        sure things can only exist in the prevention population."""
        observable, oracle = batch
        by_id = {f.case_id: f for f in observable.features}
        for outcome in oracle.outcomes:
            if classify(outcome) is Segment.SURE_THING:
                assert by_id[outcome.case_id].case_type is CaseType.UPCOMING_AT_RISK

    def test_sleeping_dogs_have_negative_contact_uplift(self, batch) -> None:  # type: ignore[no-untyped-def]
        _, oracle = batch
        dogs = [o for o in oracle.outcomes if classify(o) is Segment.SLEEPING_DOG]
        assert dogs
        for dog in dogs:
            assert dog.uplift(ActionType.SEND_PAYMENT_LINK) < 0


class TestCausalMechanisms:
    """The mechanisms the policy is meant to discover must actually be in the
    data. Each of these corresponds to one claim in the README."""

    def _latent(self, **kw: float) -> LatentCustomer:
        base: dict[str, float] = {
            "intent": 0.85,
            "balance_recovery_rate": 0.35,
            "contact_sensitivity": 0.15,
            "link_responsiveness": 0.5,
            "rail_flexibility": 0.5,
        }
        base.update(kw)
        return LatentCustomer(**base)

    def _timeline(self, params: WorldParameters, seed: int = 3) -> IssuerTimeline:
        from recovery.calibration import assumptions

        return IssuerTimeline(
            profiles=params.issuers,
            start=datetime(2026, 7, 1, tzinfo=UTC),
            days=30,
            duration_hours_range=assumptions.DEGRADATION_DURATION_HOURS_RANGE,
            rng=np.random.default_rng(seed),
        )

    def test_degraded_issuer_suppresses_immediate_retry(self, params: WorldParameters) -> None:
        """Claim 3: retry timing against issuer health recovers money with no
        customer contact. Requires that retrying into a degraded issuer is
        materially worse than waiting."""
        timeline = self._timeline(params)
        issuer = params.issuers[0].bank_name
        moment = datetime(2026, 7, 5, 12, tzinfo=UTC)

        healthy = compute_potential_outcomes(
            case_id="c1",
            latent=self._latent(),
            reason=FailureReason.BANK_DOWNTIME,
            issuer=issuer,
            failed_at=moment,
            timeline=timeline,
            is_mandate=False,
            rng=np.random.default_rng(0),
        )
        assert healthy.recovery_prob[ActionType.RETRY_NOW] > 0.5

        class AlwaysDegraded(IssuerTimeline):
            def __init__(self) -> None:
                pass

            def is_degraded(self, bank_name: str, moment: datetime) -> bool:
                return True

        degraded = compute_potential_outcomes(
            case_id="c2",
            latent=self._latent(),
            reason=FailureReason.BANK_DOWNTIME,
            issuer=issuer,
            failed_at=moment,
            timeline=AlwaysDegraded(),
            is_mandate=False,
            rng=np.random.default_rng(0),
        )
        assert (
            degraded.recovery_prob[ActionType.RETRY_NOW]
            < healthy.recovery_prob[ActionType.RETRY_NOW] / 2
        )

    def test_insufficient_funds_favours_delayed_retry(self, params: WorldParameters) -> None:
        """The balance hazard must make waiting genuinely better, otherwise
        'retry at the right moment' is indistinguishable from 'retry'."""
        timeline = self._timeline(params)
        outcome = compute_potential_outcomes(
            case_id="c3",
            latent=self._latent(balance_recovery_rate=0.9),
            reason=FailureReason.INSUFFICIENT_FUNDS,
            issuer=params.issuers[0].bank_name,
            failed_at=datetime(2026, 7, 5, 12, tzinfo=UTC),
            timeline=timeline,
            is_mandate=False,
            rng=np.random.default_rng(0),
        )
        assert (
            outcome.recovery_prob[ActionType.RETRY_SCHEDULED]
            > outcome.recovery_prob[ActionType.RETRY_NOW]
        )

    def test_cancelled_by_user_is_near_unrecoverable(self, params: WorldParameters) -> None:
        """A cancellation is evidence about intent, not a technical fault.
        Treating it like downtime is the naive policy's costliest error."""
        timeline = self._timeline(params)
        common: dict[str, Any] = {
            "latent": self._latent(),
            "issuer": params.issuers[0].bank_name,
            "failed_at": datetime(2026, 7, 5, 12, tzinfo=UTC),
            "timeline": timeline,
            "is_mandate": False,
            "rng": np.random.default_rng(0),
        }
        cancelled = compute_potential_outcomes(
            case_id="c4", reason=FailureReason.PAYMENT_CANCELLED_BY_USER, **common
        )
        downtime = compute_potential_outcomes(
            case_id="c5", reason=FailureReason.BANK_DOWNTIME, **common
        )
        assert (
            cancelled.recovery_prob[ActionType.RETRY_NOW]
            < downtime.recovery_prob[ActionType.RETRY_NOW] / 3
        )

    def test_revoked_mandate_is_unrecoverable_by_any_action(self, params: WorldParameters) -> None:
        outcome = compute_potential_outcomes(
            case_id="c6",
            latent=self._latent(intent=1.0),
            reason=FailureReason.MANDATE_REVOKED,
            issuer=params.issuers[0].bank_name,
            failed_at=datetime(2026, 7, 5, 12, tzinfo=UTC),
            timeline=self._timeline(params),
            is_mandate=True,
            rng=np.random.default_rng(0),
        )
        assert all(p == 0.0 for p in outcome.recovery_prob.values())

    def test_contact_sensitivity_creates_negative_uplift(self, params: WorldParameters) -> None:
        """Claim 1: uplift beats risk targeting. Requires that contact can
        make things worse, not merely fail to help."""
        timeline = self._timeline(params)
        common: dict[str, Any] = {
            "reason": FailureReason.BANK_DOWNTIME,
            "issuer": params.issuers[0].bank_name,
            "failed_at": datetime(2026, 7, 5, 12, tzinfo=UTC),
            "timeline": timeline,
            "is_mandate": False,
            "rng": np.random.default_rng(0),
        }
        sensitive = compute_potential_outcomes(
            case_id="c7",
            latent=self._latent(contact_sensitivity=0.95, link_responsiveness=0.1),
            **common,
        )
        assert sensitive.uplift(ActionType.SEND_PAYMENT_LINK) < 0

    def test_coupled_draw_preserves_outcome_monotonicity(self, params: WorldParameters) -> None:
        """A single uniform per case means a customer who recovers under a
        weaker action also recovers under a stronger one. Independent draws
        would manufacture uplift out of noise."""
        _, oracle = generate(params, n_cases=500, seed=11)
        for outcome in oracle.outcomes:
            ranked = sorted(outcome.recovery_prob.items(), key=lambda kv: kv[1])
            seen_recovery = False
            for _action, prob in ranked:
                recovered = outcome.uniform_draw < prob
                if seen_recovery:
                    assert recovered
                seen_recovery = seen_recovery or recovered


class TestLatentSampling:
    def test_intent_is_bimodal(self) -> None:
        """A unimodal intent distribution would understate both lost causes
        and sure things, flattening the segment structure."""
        rng = np.random.default_rng(0)
        intents = [sample_latent(rng).intent for _ in range(4000)]
        low = sum(i < 0.35 for i in intents) / len(intents)
        high = sum(i > 0.70 for i in intents) / len(intents)
        assert low > 0.10
        assert high > 0.40

    def test_history_is_a_noisy_proxy(self) -> None:
        """Observable history must not perfectly reveal intent, or targeting
        becomes trivial in a way no production system enjoys."""
        from recovery.world.latent import observable_history

        rng = np.random.default_rng(0)
        latent = LatentCustomer(
            intent=0.8,
            balance_recovery_rate=0.3,
            contact_sensitivity=0.2,
            link_responsiveness=0.5,
            rail_flexibility=0.5,
        )
        rates = []
        for _ in range(200):
            payments, failures, _ = observable_history(latent, 400, rng)
            total = payments + failures
            rates.append(failures / total if total else 0.0)
        assert np.std(rates) > 0.01
