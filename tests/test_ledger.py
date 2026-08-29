"""Ledger tests.

The claims, in order of consequence if false:

* The store cannot be edited or deleted from, including by raw SQL.
* Replayed state matches what actually happened.
* The reported recovery total equals the ledger sum, exactly.
* Denials are recorded, not merely enforced.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from recovery.domain.actions import (
    Action,
    ActionScore,
    ComplianceReview,
    Decision,
    Diagnosis,
    ExecutionRecord,
    GateResult,
)
from recovery.domain.enums import (
    ActionType,
    CaseState,
    CaseType,
    Channel,
    DeclineClass,
    FailureReason,
    GateVerdict,
    PaymentMethod,
)
from recovery.domain.observations import CaseFeatures
from recovery.ledger.events import Event, EventType
from recovery.ledger.recorder import AuditRecorder
from recovery.ledger.replay import (
    deferral_report,
    denial_report,
    reconcile,
    replay,
)
from recovery.ledger.store import EventStore

NOW = datetime(2026, 7, 6, 9, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> EventStore:
    return EventStore(tmp_path / "audit.sqlite3")


@pytest.fixture
def recorder(store: EventStore) -> AuditRecorder:
    return AuditRecorder(store)


def _features(case_id: str = "case_1", amount_paise: int = 49900) -> CaseFeatures:
    return CaseFeatures(
        case_id=case_id,
        case_type=CaseType.MANDATE_FAILURE,
        created_at=NOW,
        amount_paise=amount_paise,
        method=PaymentMethod.EMANDATE,
        issuer="State Bank of India",
        reason=FailureReason.INSUFFICIENT_FUNDS,
        decline_class=DeclineClass.BUSINESS,
        customer_id="cst_1",
        tenure_days=400,
        prior_payment_count=20,
        prior_failure_count=3,
        prior_recovery_count=2,
        contacts_last_30d=0,
        dnd_registered=False,
        hour_of_day=9,
        day_of_month=6,
        days_since_salary=5,
        issuer_failures_last_hour=2,
        issuer_volume_last_hour=300,
    )


def _diagnosis(case_id: str = "case_1") -> Diagnosis:
    return Diagnosis(
        case_id=case_id,
        root_cause="funding",
        confidence=0.88,
        recoverable_without_contact=True,
        evidence=("reason=insufficient_funds -> funding", "5d since salary"),
    )


def _decision(case_id: str = "case_1", action: ActionType = ActionType.RETRY_SCHEDULED) -> Decision:
    return Decision(
        case_id=case_id,
        decided_at=NOW,
        action=Action(action_type=action, channel=Channel.NONE),
        scores=(
            ActionScore(
                action_type=ActionType.NO_ACTION,
                p_recovery=0.18,
                uplift=0.0,
                cost_paise=0,
                expected_value_paise=8982.0,
            ),
            ActionScore(
                action_type=action,
                p_recovery=0.44,
                uplift=0.26,
                cost_paise=50,
                expected_value_paise=21906.0,
            ),
        ),
        policy_name="uplift_ev",
        model_version="v0.9",
        rationale="funding; retry_scheduled EV 21906p vs inaction 8982p",
        propensity=1.0,
    )


def _review(
    case_id: str, *, blocked: str | None = None, deferred: str | None = None
) -> ComplianceReview:
    results = [GateResult(rule_id="ATTEMPT_BUDGET", verdict=GateVerdict.ALLOW, reason="0/3")]
    if blocked:
        results.append(GateResult(rule_id=blocked, verdict=GateVerdict.BLOCK, reason="blocked"))
    if deferred:
        results.append(
            GateResult(
                rule_id=deferred,
                verdict=GateVerdict.DEFER,
                reason="not yet",
                defer_until=NOW + timedelta(hours=3),
            )
        )
    return ComplianceReview(case_id=case_id, reviewed_at=NOW, results=tuple(results))


def _execution(case_id: str = "case_1", succeeded: bool = True) -> ExecutionRecord:
    return ExecutionRecord(
        case_id=case_id,
        idempotency_key="rec_abc123",
        action_type=ActionType.RETRY_SCHEDULED,
        executed_at=NOW + timedelta(minutes=1),
        api_endpoint="orders.create",
        request_digest="abc123",
        succeeded=succeeded,
        response_code="created" if succeeded else None,
        error_reason=None if succeeded else "GATEWAY_ERROR: timeout",
    )


class TestAppendOnly:
    def test_update_is_rejected_at_the_database(self, store: EventStore) -> None:
        """Enforced by trigger, not only by the absence of a method. A future
        contributor reaching for raw SQL should hit a wall."""
        store.append(Event(case_id="c", event_type=EventType.CASE_INGESTED, occurred_at=NOW))
        conn = sqlite3.connect(store.path)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE events SET case_id = 'tampered'")
        conn.close()

    def test_delete_is_rejected_at_the_database(self, store: EventStore) -> None:
        store.append(Event(case_id="c", event_type=EventType.CASE_INGESTED, occurred_at=NOW))
        conn = sqlite3.connect(store.path)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM events")
        conn.close()

    def test_store_exposes_no_mutation_methods(self) -> None:
        forbidden = {"update", "delete", "remove", "edit", "amend"}
        assert not {m for m in dir(EventStore) if m in forbidden}

    def test_sequences_are_monotonic_per_case(self, store: EventStore) -> None:
        for _ in range(5):
            store.append(Event(case_id="c", event_type=EventType.CASE_INGESTED, occurred_at=NOW))
        assert store.verify_sequences() == []
        assert len(store.for_case("c")) == 5

    def test_cases_do_not_share_a_sequence(self, store: EventStore) -> None:
        store.append(Event(case_id="a", event_type=EventType.CASE_INGESTED, occurred_at=NOW))
        store.append(Event(case_id="b", event_type=EventType.CASE_INGESTED, occurred_at=NOW))
        store.append(Event(case_id="a", event_type=EventType.CASE_DIAGNOSED, occurred_at=NOW))
        assert len(store.for_case("a")) == 2
        assert len(store.for_case("b")) == 1


class TestReplay:
    def test_full_trail_reconstructs_a_recovery(
        self, store: EventStore, recorder: AuditRecorder
    ) -> None:
        recorder.ingested(_features())
        recorder.diagnosed(_diagnosis(), NOW)
        recorder.scored(_decision())
        recorder.reviewed(_review("case_1"), ActionType.RETRY_SCHEDULED)
        recorder.decided(_decision())
        recorder.executed(_execution(), provider_reference="order_1")
        recorder.outcome("case_1", recovered=True, amount_paise=49900, at=NOW)

        history = replay("case_1", store.for_case("case_1"))
        assert history.state is CaseState.RECOVERED
        assert history.recovered_paise == 49900
        assert history.attempts_used == 1
        assert history.contacts_used == 0
        assert history.root_cause == "funding"

    def test_failed_execution_does_not_count_as_an_attempt_taken(
        self, store: EventStore, recorder: AuditRecorder
    ) -> None:
        recorder.ingested(_features())
        recorder.executed(_execution(succeeded=False))
        history = replay("case_1", store.for_case("case_1"))
        assert history.actions_taken == []
        assert history.state is CaseState.FAILED

    def test_contact_and_attempt_budgets_are_distinguished(
        self, store: EventStore, recorder: AuditRecorder
    ) -> None:
        """A silent retry costs an attempt; a message costs a contact. The
        asymmetry is why well-timed retries are the cheapest money."""
        recorder.ingested(_features())
        recorder.executed(_execution())
        link = _execution()
        recorder.executed(link.model_copy(update={"action_type": ActionType.SEND_PAYMENT_LINK}))
        history = replay("case_1", store.for_case("case_1"))
        assert history.attempts_used == 1
        assert history.contacts_used == 1

    def test_narrative_is_ordered_and_human_readable(
        self, store: EventStore, recorder: AuditRecorder
    ) -> None:
        recorder.ingested(_features())
        recorder.diagnosed(_diagnosis(), NOW)
        recorder.outcome("case_1", recovered=True, amount_paise=49900, at=NOW)
        lines = replay("case_1", store.for_case("case_1")).narrative()
        assert len(lines) == 3
        assert "ingested" in lines[0]
        assert "diagnosed funding" in lines[1]
        assert "RECOVERED" in lines[2]

    def test_replay_is_deterministic(self, store: EventStore, recorder: AuditRecorder) -> None:
        recorder.ingested(_features())
        recorder.executed(_execution())
        recorder.outcome("case_1", recovered=True, amount_paise=49900, at=NOW)
        events = store.for_case("case_1")
        assert replay("case_1", events).recovered_paise == replay("case_1", events).recovered_paise


class TestDenialsAreRecorded:
    def test_blocked_action_names_the_rule(
        self, store: EventStore, recorder: AuditRecorder
    ) -> None:
        recorder.ingested(_features())
        recorder.reviewed(_review("case_1", blocked="DND_REGISTRY"), ActionType.SEND_PAYMENT_LINK)
        history = replay("case_1", store.for_case("case_1"))
        assert (ActionType.SEND_PAYMENT_LINK, "DND_REGISTRY") in history.blocked_by
        assert history.state is CaseState.GATE_BLOCKED

    def test_deferral_is_recorded_separately_from_denial(
        self, store: EventStore, recorder: AuditRecorder
    ) -> None:
        """A deferred action is early, not forbidden. Collapsing the two
        would misreport the reason to a regulator."""
        recorder.ingested(_features())
        recorder.reviewed(_review("case_1", deferred="CONTACT_HOURS"), ActionType.SEND_PAYMENT_LINK)
        history = replay("case_1", store.for_case("case_1"))
        assert history.blocked_by == []
        assert (ActionType.SEND_PAYMENT_LINK, "CONTACT_HOURS") in history.deferred_by

    def test_denial_report_aggregates_across_cases(
        self, store: EventStore, recorder: AuditRecorder
    ) -> None:
        for i in range(3):
            case_id = f"case_{i}"
            recorder.ingested(_features(case_id))
            recorder.reviewed(
                _review(case_id, blocked="DND_REGISTRY"), ActionType.SEND_PAYMENT_LINK
            )
        histories = [replay(c, store.for_case(c)) for c in store.case_ids()]
        assert denial_report(histories)["DND_REGISTRY"] == 3
        assert deferral_report(histories) == {}

    def test_every_gate_result_is_persisted_not_just_the_blocking_one(
        self, store: EventStore, recorder: AuditRecorder
    ) -> None:
        """An auditor asking 'what else was checked' needs the passes too."""
        recorder.reviewed(_review("case_1", blocked="DND_REGISTRY"), ActionType.SEND_PAYMENT_LINK)
        event = store.for_case("case_1")[0]
        assert len(event.payload["results"]) == 2


class TestReconciliation:
    def test_matching_totals_reconcile_exactly(
        self, store: EventStore, recorder: AuditRecorder
    ) -> None:
        for i in range(4):
            case_id = f"case_{i}"
            recorder.ingested(_features(case_id))
            recorder.outcome(case_id, recovered=True, amount_paise=49900, at=NOW)
        histories = [replay(c, store.for_case(c)) for c in store.case_ids()]
        result = reconcile(4 * 49900, histories)
        assert result.reconciles
        assert result.difference_paise == 0
        assert result.cases_recovered == 4

    def test_one_paise_of_drift_fails(self, store: EventStore, recorder: AuditRecorder) -> None:
        """Exact equality, not a tolerance. Integer paise (ADR-0003) exists so
        this test can be this strict; a tolerance would permit the drift the
        check is for."""
        recorder.ingested(_features())
        recorder.outcome("case_1", recovered=True, amount_paise=49900, at=NOW)
        histories = [replay("case_1", store.for_case("case_1"))]
        assert not reconcile(49901, histories).reconciles

    def test_unrecovered_cases_contribute_nothing(
        self, store: EventStore, recorder: AuditRecorder
    ) -> None:
        recorder.ingested(_features("case_1"))
        recorder.outcome("case_1", recovered=False, amount_paise=49900, at=NOW)
        history = replay("case_1", store.for_case("case_1"))
        assert history.recovered_paise == 0
        assert reconcile(0, [history]).reconciles

    def test_reconciliation_reports_case_counts(
        self, store: EventStore, recorder: AuditRecorder
    ) -> None:
        recorder.ingested(_features("case_1"))
        recorder.outcome("case_1", recovered=True, amount_paise=10000, at=NOW)
        recorder.ingested(_features("case_2"))
        recorder.outcome("case_2", recovered=False, amount_paise=10000, at=NOW)
        histories = [replay(c, store.for_case(c)) for c in store.case_ids()]
        result = reconcile(10000, histories)
        assert result.cases_in_ledger == 2
        assert result.cases_recovered == 1


class TestEvidenceIsPreserved:
    def test_diagnosis_evidence_survives_into_the_trail(
        self, store: EventStore, recorder: AuditRecorder
    ) -> None:
        """A diagnosis without its evidence is an assertion, and an auditor
        cannot check an assertion."""
        recorder.diagnosed(_diagnosis(), NOW)
        event = store.for_case("case_1")[0]
        assert len(event.payload["evidence"]) == 2

    def test_rejected_actions_are_kept_in_the_score_table(
        self, store: EventStore, recorder: AuditRecorder
    ) -> None:
        recorder.scored(_decision())
        event = store.for_case("case_1")[0]
        actions = {s["action_type"] for s in event.payload["scores"]}
        assert "no_action" in actions

    def test_idempotency_key_is_in_the_execution_record(
        self, store: EventStore, recorder: AuditRecorder
    ) -> None:
        recorder.executed(_execution(), provider_reference="order_1")
        event = store.for_case("case_1")[0]
        assert event.payload["idempotency_key"] == "rec_abc123"
        assert event.payload["provider_reference"] == "order_1"
