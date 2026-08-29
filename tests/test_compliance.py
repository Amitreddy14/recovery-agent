"""Compliance and decision tests.

The claims under test:

* Every gate fires when it should, and denials are recorded with the rule id.
* Deferral is distinct from denial.
* Expected value, not ranking, is what declines to act on a sleeping dog.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from recovery.compliance.engine import CaseContext, ComplianceEngine, load_policy
from recovery.domain.actions import Action, ActionScore, Diagnosis, GateResult
from recovery.domain.entities import Mandate, RecoveryCase
from recovery.domain.enums import (
    ActionType,
    CaseState,
    CaseType,
    Channel,
    GateVerdict,
    MandateCategory,
)
from recovery.policy.decision import DecisionEngine
from recovery.policy.economics import Economics, score_action

POLICY_PATH = Path(__file__).resolve().parents[1] / "configs" / "compliance" / "policy.yaml"
NOON_IST = datetime(2026, 7, 6, 6, 30, tzinfo=UTC)  # 12:00 IST
MIDNIGHT_IST = datetime(2026, 7, 6, 20, 0, tzinfo=UTC)  # 01:30 IST next day


@pytest.fixture(scope="module")
def engine() -> ComplianceEngine:
    return ComplianceEngine(load_policy(POLICY_PATH))


@pytest.fixture(scope="module")
def economics() -> Economics:
    policy = load_policy(POLICY_PATH)
    return Economics.from_policy(policy.costs_paise, policy.sleeping_dog_penalty)


def _case(**kw: Any) -> RecoveryCase:
    base: dict[str, Any] = {
        "case_id": "case_1",
        "case_type": CaseType.MANDATE_FAILURE,
        "merchant_id": "mrc_1",
        "customer_id": "cst_1",
        "amount_at_risk_paise": 49900,
        "created_at": NOON_IST,
        "state": CaseState.SCORED,
    }
    base.update(kw)
    return RecoveryCase(**base)


def _mandate(**kw: Any) -> Mandate:
    base: dict[str, Any] = {
        "mandate_id": "mdt_1",
        "customer_id": "cst_1",
        "merchant_id": "mrc_1",
        "category": MandateCategory.SUBSCRIPTION,
        "max_amount_paise": 99800,
        "debit_amount_paise": 49900,
        "registered_at": NOON_IST - timedelta(days=90),
        "valid_until": NOON_IST + timedelta(days=365),
        "next_debit_at": NOON_IST + timedelta(days=1),
    }
    base.update(kw)
    return Mandate(**base)


def _ctx(**kw: Any) -> CaseContext:
    base: dict[str, Any] = {
        "case": _case(),
        "mandate": _mandate(),
        "now": NOON_IST,
    }
    base.update(kw)
    return CaseContext(**base)


def _verdict(engine: ComplianceEngine, action: Action, ctx: CaseContext, rule: str) -> GateResult:
    review = engine.review(action, ctx)
    return next(r for r in review.results if r.rule_id == rule)


LINK = Action(action_type=ActionType.SEND_PAYMENT_LINK, channel=Channel.SMS)
NUDGE = Action(action_type=ActionType.PRE_DEBIT_NUDGE, channel=Channel.SMS)
RETRY = Action(action_type=ActionType.RETRY_NOW)


class TestHardStops:
    def test_opt_out_blocks_everything_but_inaction(self, engine: ComplianceEngine) -> None:
        ctx = _ctx(case=_case(customer_opted_out=True))
        assert not engine.review(LINK, ctx).allowed
        assert "RBI_OPT_OUT_HONOURED" in engine.review(LINK, ctx).blocking_rules

    def test_dispute_freezes_recovery(self, engine: ComplianceEngine) -> None:
        ctx = _ctx(case=_case(dispute_raised=True))
        assert "DISPUTE_RAISED" in engine.review(RETRY, ctx).blocking_rules

    def test_revoked_mandate_cannot_be_debited(self, engine: ComplianceEngine) -> None:
        ctx = _ctx(mandate=_mandate(active=False))
        assert "MANDATE_REVOKED" in engine.review(RETRY, ctx).blocking_rules

    def test_expired_mandate_cannot_be_debited(self, engine: ComplianceEngine) -> None:
        ctx = _ctx(mandate=_mandate(valid_until=NOON_IST - timedelta(days=1)))
        assert "MANDATE_REVOKED" in engine.review(RETRY, ctx).blocking_rules


class TestRbiEmandate:
    def test_above_afa_ceiling_is_blocked_not_deferred(self, engine: ComplianceEngine) -> None:
        """Waiting does not make an unauthenticated debit permissible."""
        ctx = _ctx(mandate=_mandate(debit_amount_paise=2_000_000, max_amount_paise=4_000_000))
        result = _verdict(engine, RETRY, ctx, "RBI_AFA_THRESHOLD")
        assert result.verdict is GateVerdict.BLOCK

    def test_elevated_category_permits_larger_debit(self, engine: ComplianceEngine) -> None:
        """Insurance premiums carry a Rs 1,00,000 ceiling, not Rs 15,000."""
        ctx = _ctx(
            mandate=_mandate(
                category=MandateCategory.INSURANCE_PREMIUM,
                debit_amount_paise=2_000_000,
                max_amount_paise=4_000_000,
            )
        )
        result = _verdict(engine, RETRY, ctx, "RBI_AFA_THRESHOLD")
        assert result.verdict is GateVerdict.ALLOW

    def test_scheduled_debit_without_notice_is_deferred(self, engine: ComplianceEngine) -> None:
        action = Action(
            action_type=ActionType.RETRY_SCHEDULED,
            scheduled_for=NOON_IST + timedelta(hours=2),
        )
        result = _verdict(engine, action, _ctx(), "RBI_PRE_DEBIT_NOTIFICATION")
        assert result.verdict is GateVerdict.DEFER
        assert result.defer_until == NOON_IST + timedelta(hours=24)

    def test_scheduled_debit_with_notice_is_allowed(self, engine: ComplianceEngine) -> None:
        action = Action(
            action_type=ActionType.RETRY_SCHEDULED,
            scheduled_for=NOON_IST + timedelta(hours=26),
        )
        ctx = _ctx(pre_debit_notice_sent_at=NOON_IST)
        result = _verdict(engine, action, ctx, "RBI_PRE_DEBIT_NOTIFICATION")
        assert result.verdict is GateVerdict.ALLOW


class TestConduct:
    def test_contact_outside_hours_is_deferred_not_blocked(self, engine: ComplianceEngine) -> None:
        """An untimely message is early, not forbidden. Blocking would discard
        recoverable cases for a reason that expires in a few hours."""
        result = _verdict(engine, LINK, _ctx(now=MIDNIGHT_IST), "CONTACT_HOURS")
        assert result.verdict is GateVerdict.DEFER
        assert result.defer_until is not None

    def test_retries_are_not_bound_by_contact_hours(self, engine: ComplianceEngine) -> None:
        """A silent retry disturbs nobody at 2am."""
        result = _verdict(engine, RETRY, _ctx(now=MIDNIGHT_IST), "CONTACT_HOURS")
        assert result.verdict is GateVerdict.ALLOW

    def test_dnd_blocks_payment_link(self, engine: ComplianceEngine) -> None:
        assert "DND_REGISTRY" in engine.review(LINK, _ctx(dnd_registered=True)).blocking_rules

    def test_dnd_exempts_pre_debit_notification(self, engine: ComplianceEngine) -> None:
        """A customer cannot opt out of being told money is about to leave
        their account — the notification is transactional, not promotional."""
        result = _verdict(engine, NUDGE, _ctx(dnd_registered=True), "DND_REGISTRY")
        assert result.verdict is GateVerdict.ALLOW

    def test_contact_cap_blocks_further_messages(self, engine: ComplianceEngine) -> None:
        ctx = _ctx(case=_case(contacts_used=2))
        assert "CONTACT_FREQUENCY_CAP" in engine.review(LINK, ctx).blocking_rules

    def test_attempt_budget_blocks_further_retries(self, engine: ComplianceEngine) -> None:
        ctx = _ctx(case=_case(attempts_used=3))
        assert "ATTEMPT_BUDGET" in engine.review(RETRY, ctx).blocking_rules


class TestAuditTrail:
    def test_all_rules_run_even_after_a_denial(self, engine: ComplianceEngine) -> None:
        """Short-circuiting on the first block would leave the trail unable to
        answer 'what else was wrong with this?'."""
        ctx = _ctx(case=_case(customer_opted_out=True, dispute_raised=True))
        review = engine.review(LINK, ctx)
        assert {"RBI_OPT_OUT_HONOURED", "DISPUTE_RAISED"} <= set(review.blocking_rules)

    def test_every_result_carries_a_reason(self, engine: ComplianceEngine) -> None:
        for result in engine.review(LINK, _ctx()).results:
            assert result.reason

    def test_defer_is_not_allow(self, engine: ComplianceEngine) -> None:
        assert not engine.review(LINK, _ctx(now=MIDNIGHT_IST)).allowed


class TestEconomics:
    def test_cancellation_dominates_a_small_recovery(self, economics: Economics) -> None:
        """The sleeping-dog fix, stated numerically. A Rs 499 subscription
        carries ~Rs 6,000 of remaining value; a 5% cancellation risk costs
        more than the entire payment is worth."""
        mandate = _mandate()
        score = score_action(
            action=ActionType.SEND_PAYMENT_LINK,
            p_recovery=0.55,
            uplift=0.05,
            p_cancel=0.05,
            amount_paise=49900,
            mandate=mandate,
            economics=economics,
        )
        assert score.expected_value_paise < 0

    def test_zero_cancellation_risk_leaves_value_positive(self, economics: Economics) -> None:
        score = score_action(
            action=ActionType.SEND_PAYMENT_LINK,
            p_recovery=0.55,
            uplift=0.05,
            p_cancel=0.0,
            amount_paise=49900,
            mandate=_mandate(),
            economics=economics,
        )
        assert score.expected_value_paise > 0

    def test_non_mandate_case_has_no_cancellation_exposure(self, economics: Economics) -> None:
        assert economics.remaining_mandate_value_paise(None) == 0

    def test_remaining_value_scales_with_horizon(self, economics: Economics) -> None:
        assert economics.remaining_mandate_value_paise(_mandate()) == 49900 * 12


class TestDecisionEngine:
    def _scores(self, contact_ev: float) -> list[ActionScore]:
        return [
            ActionScore(
                action_type=ActionType.NO_ACTION,
                p_recovery=0.5,
                uplift=0.0,
                cost_paise=0,
                expected_value_paise=24950.0,
            ),
            ActionScore(
                action_type=ActionType.SEND_PAYMENT_LINK,
                p_recovery=0.55,
                uplift=0.05,
                cost_paise=35,
                expected_value_paise=contact_ev,
            ),
        ]

    def _diagnosis(self) -> Diagnosis:
        return Diagnosis(
            case_id="case_1",
            root_cause="funding",
            confidence=0.88,
            recoverable_without_contact=True,
        )

    def test_declines_to_act_when_value_is_negative(
        self, engine: ComplianceEngine, economics: Economics
    ) -> None:
        """This is what a ranking cannot do. Ranking orders cases; it has no
        way to decline to spend the budget at all."""
        decision = DecisionEngine(engine, economics).decide(
            scores=self._scores(contact_ev=-500.0),
            context=_ctx(),
            diagnosis=self._diagnosis(),
        )
        assert decision.decision.action.action_type is ActionType.NO_ACTION

    def test_acts_when_value_is_positive(
        self, engine: ComplianceEngine, economics: Economics
    ) -> None:
        decision = DecisionEngine(engine, economics).decide(
            scores=self._scores(contact_ev=30000.0),
            context=_ctx(),
            diagnosis=self._diagnosis(),
        )
        assert decision.decision.action.action_type is ActionType.SEND_PAYMENT_LINK

    def test_gate_overrides_a_profitable_action(
        self, engine: ComplianceEngine, economics: Economics
    ) -> None:
        """Compliance is not a tie-breaker. A blocked action does not run
        however valuable it looks."""
        decision = DecisionEngine(engine, economics).decide(
            scores=self._scores(contact_ev=999999.0),
            context=_ctx(dnd_registered=True),
            diagnosis=self._diagnosis(),
        )
        assert decision.decision.action.action_type is ActionType.NO_ACTION
        assert "DND_REGISTRY" in decision.blocked_actions[ActionType.SEND_PAYMENT_LINK]

    def test_rationale_names_the_blocking_rule(
        self, engine: ComplianceEngine, economics: Economics
    ) -> None:
        decision = DecisionEngine(engine, economics).decide(
            scores=self._scores(contact_ev=999999.0),
            context=_ctx(dnd_registered=True),
            diagnosis=self._diagnosis(),
        )
        assert "DND_REGISTRY" in decision.decision.rationale

    def test_score_table_retains_rejected_actions(
        self, engine: ComplianceEngine, economics: Economics
    ) -> None:
        """An audit needs to see what would have been chosen absent the rule."""
        decision = DecisionEngine(engine, economics).decide(
            scores=self._scores(contact_ev=999999.0),
            context=_ctx(dnd_registered=True),
            diagnosis=self._diagnosis(),
        )
        assert any(s.action_type is ActionType.SEND_PAYMENT_LINK for s in decision.decision.scores)
