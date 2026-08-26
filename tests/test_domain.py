"""Domain invariants.

These tests exist to catch the failure modes that would quietly corrupt the
money numbers: mutated state, float arithmetic on rupees, missing
propensities, and mis-set AFA ceilings.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from recovery.domain import (
    Action,
    ActionScore,
    ActionType,
    CaseState,
    CaseType,
    Channel,
    ComplianceReview,
    Customer,
    Decision,
    GateResult,
    GateVerdict,
    Mandate,
    MandateCategory,
    Outcome,
    PaymentMethod,
    RecoveryCase,
)

NOW = datetime(2026, 8, 26, 10, 0, tzinfo=UTC)


def _mandate(category: MandateCategory, amount_paise: int) -> Mandate:
    return Mandate(
        mandate_id="mdt_1",
        customer_id="cst_1",
        merchant_id="mrc_1",
        category=category,
        max_amount_paise=amount_paise * 2,
        debit_amount_paise=amount_paise,
        registered_at=NOW - timedelta(days=90),
        valid_until=NOW + timedelta(days=365),
        next_debit_at=NOW + timedelta(days=1),
    )


class TestAfaCeiling:
    """RBI E-mandate Framework 2026 thresholds."""

    @pytest.mark.parametrize(
        "category",
        [
            MandateCategory.INSURANCE_PREMIUM,
            MandateCategory.MUTUAL_FUND,
            MandateCategory.CREDIT_CARD_BILL,
        ],
    )
    def test_elevated_categories_get_one_lakh(self, category: MandateCategory) -> None:
        assert _mandate(category, 5_000_00).afa_free_ceiling_paise == 100_000_00

    @pytest.mark.parametrize(
        "category",
        [
            MandateCategory.SUBSCRIPTION,
            MandateCategory.UTILITY,
            MandateCategory.LOAN_EMI,
            MandateCategory.OTHER,
        ],
    )
    def test_other_categories_capped_at_fifteen_thousand(self, category: MandateCategory) -> None:
        assert _mandate(category, 5_000_00).afa_free_ceiling_paise == 15_000_00

    def test_loan_emi_is_not_elevated(self) -> None:
        """The Rs 1 lakh exception covers insurance, mutual funds and credit
        card bills only. Loan EMIs are a common misreading."""
        assert _mandate(MandateCategory.LOAN_EMI, 50_000_00).requires_afa is True

    def test_requires_afa_boundary_is_exclusive(self) -> None:
        assert _mandate(MandateCategory.SUBSCRIPTION, 15_000_00).requires_afa is False
        assert _mandate(MandateCategory.SUBSCRIPTION, 15_000_01).requires_afa is True


class TestImmutability:
    def test_case_cannot_be_mutated(self) -> None:
        case = _case()
        with pytest.raises(ValidationError):
            case.attempts_used = 99  # type: ignore[misc]

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Customer(
                customer_id="cst_1",
                tenure_days=10,
                prior_payment_count=1,
                prior_failure_count=0,
                prior_recovery_count=0,
                issuer="HDFC",
                preferred_method=PaymentMethod.UPI,
                sneaky_leak="ground truth",  # type: ignore[call-arg]
            )


def _case(**overrides: Any) -> RecoveryCase:
    base: dict[str, Any] = {
        "case_id": "case_1",
        "case_type": CaseType.PAYMENT_FAILURE,
        "merchant_id": "mrc_1",
        "customer_id": "cst_1",
        "amount_at_risk_paise": 2_499_00,
        "created_at": NOW,
    }
    base.update(overrides)
    return RecoveryCase(**base)


class TestBudgets:
    def test_remaining_never_negative(self) -> None:
        case = _case(max_attempts=1, attempts_used=5)
        assert case.attempts_remaining == 0

    def test_terminal_states(self) -> None:
        assert _case(state=CaseState.RECOVERED).is_terminal
        assert _case(state=CaseState.ABANDONED).is_terminal
        assert not _case(state=CaseState.DECIDED).is_terminal


class TestActionSemantics:
    def test_retries_consume_attempts_not_contacts(self) -> None:
        action = Action(action_type=ActionType.RETRY_NOW)
        assert action.consumes_attempt
        assert not action.consumes_contact

    def test_payment_link_consumes_contact(self) -> None:
        action = Action(action_type=ActionType.SEND_PAYMENT_LINK, channel=Channel.SMS)
        assert action.consumes_contact
        assert not action.consumes_attempt

    def test_no_action_is_free(self) -> None:
        action = Action(action_type=ActionType.NO_ACTION)
        assert not action.consumes_attempt
        assert not action.consumes_contact


class TestDecisionRequiresPropensity:
    """Off-policy evaluation is impossible without logged propensities, so
    the schema refuses to represent a decision that lacks one."""

    def test_propensity_is_mandatory(self) -> None:
        with pytest.raises(ValidationError):
            Decision(  # type: ignore[call-arg]
                case_id="case_1",
                decided_at=NOW,
                action=Action(action_type=ActionType.NO_ACTION),
                scores=(),
                policy_name="test",
                model_version="v0",
                rationale="none",
            )

    def test_zero_propensity_rejected(self) -> None:
        """A zero propensity would divide by zero in an IPS estimator."""
        with pytest.raises(ValidationError):
            _decision(propensity=0.0)

    def test_valid_decision_round_trips(self) -> None:
        decision = _decision(propensity=0.25)
        assert Decision.model_validate_json(decision.model_dump_json()) == decision


def _decision(propensity: float) -> Decision:
    return Decision(
        case_id="case_1",
        decided_at=NOW,
        action=Action(action_type=ActionType.RETRY_SCHEDULED, scheduled_for=NOW),
        scores=(
            ActionScore(
                action_type=ActionType.RETRY_SCHEDULED,
                p_recovery=0.42,
                uplift=0.18,
                cost_paise=50,
                expected_value_paise=44_932.0,
            ),
        ),
        policy_name="uplift_ev",
        model_version="v0",
        rationale="issuer degraded; deferred retry dominates",
        propensity=propensity,
    )


class TestComplianceReview:
    def test_all_allow_passes(self) -> None:
        review = ComplianceReview(
            case_id="case_1",
            reviewed_at=NOW,
            results=(
                GateResult(rule_id="A", verdict=GateVerdict.ALLOW, reason="ok"),
                GateResult(rule_id="B", verdict=GateVerdict.ALLOW, reason="ok"),
            ),
        )
        assert review.allowed
        assert review.blocking_rules == ()

    def test_defer_is_not_allow(self) -> None:
        """A deferred action has not been approved for execution now."""
        review = ComplianceReview(
            case_id="case_1",
            reviewed_at=NOW,
            results=(
                GateResult(
                    rule_id="CONTACT_HOURS",
                    verdict=GateVerdict.DEFER,
                    reason="outside 09:00-19:00 IST",
                    defer_until=NOW + timedelta(hours=3),
                ),
            ),
        )
        assert not review.allowed
        assert review.blocking_rules == ()

    def test_blocking_rules_reported(self) -> None:
        review = ComplianceReview(
            case_id="case_1",
            reviewed_at=NOW,
            results=(
                GateResult(
                    rule_id="MANDATE_REVOKED",
                    verdict=GateVerdict.BLOCK,
                    reason="mandate revoked 2026-08-01",
                ),
            ),
        )
        assert not review.allowed
        assert review.blocking_rules == ("MANDATE_REVOKED",)


class TestOutcomeArithmetic:
    def test_net_is_integer_paise(self) -> None:
        outcome = Outcome(
            case_id="case_1",
            observed_at=NOW,
            recovered=True,
            recovered_amount_paise=2_499_00,
            intervention_cost_paise=85,
        )
        assert outcome.net_paise == 249_815
        assert isinstance(outcome.net_paise, int)

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Outcome(
                case_id="case_1",
                observed_at=NOW,
                recovered=False,
                intervention_cost_paise=-1,
            )
