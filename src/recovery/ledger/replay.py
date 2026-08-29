"""Replay and reconciliation.

Two things this module makes possible that a status column cannot:

* **Replay.** Case state is recomputed from the event log, so "how did this
  case end up here" has an answer that is derived rather than asserted.
* **Reconciliation.** The reported recovery total is recomputed by summing
  the ledger. If the two disagree, the headline number is wrong — and this is
  where integer paise (ADR-0003) pays off, because the comparison is exact
  rather than approximate.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from recovery.domain.enums import ActionType, CaseState
from recovery.ledger.events import Event, EventType


@dataclass
class CaseHistory:
    """State of one case, derived entirely from its events."""

    case_id: str
    events: list[Event]
    state: CaseState = CaseState.INGESTED
    amount_at_risk_paise: int = 0
    recovered_paise: int = 0
    attempts_used: int = 0
    contacts_used: int = 0
    actions_taken: list[ActionType] = field(default_factory=list)
    blocked_by: list[tuple[ActionType, str]] = field(default_factory=list)
    deferred_by: list[tuple[ActionType, str]] = field(default_factory=list)
    root_cause: str | None = None

    @property
    def was_recovered(self) -> bool:
        return self.state is CaseState.RECOVERED

    @property
    def denial_count(self) -> int:
        return len(self.blocked_by)

    def narrative(self) -> list[str]:
        """The trail as a human reads it: one line per event, in order."""
        return [f"{e.occurred_at.strftime('%Y-%m-%d %H:%M')}  {e.summary()}" for e in self.events]


CONTACTING = {ActionType.SEND_PAYMENT_LINK, ActionType.PRE_DEBIT_NUDGE}
ATTEMPTING = {
    ActionType.RETRY_NOW,
    ActionType.RETRY_SCHEDULED,
    ActionType.RETRY_ALTERNATE_RAIL,
}


def replay(case_id: str, events: Sequence[Event]) -> CaseHistory:
    """Rebuild a case from its events.

    Deliberately a fold over the log rather than a lookup. If this function
    and the live system ever disagree about a case, the log is right and the
    live system has drifted — which is the property that makes the log
    evidence rather than a report.
    """
    history = CaseHistory(case_id=case_id, events=list(events))

    for event in events:
        payload = event.payload

        if event.event_type is EventType.CASE_INGESTED:
            history.amount_at_risk_paise = int(payload.get("amount_paise", 0))
            history.state = CaseState.INGESTED

        elif event.event_type is EventType.CASE_DIAGNOSED:
            history.root_cause = payload.get("root_cause")
            history.state = CaseState.DIAGNOSED

        elif event.event_type is EventType.ACTIONS_SCORED:
            history.state = CaseState.SCORED

        elif event.event_type is EventType.COMPLIANCE_REVIEWED:
            action = payload.get("action")
            if action:
                for rule in payload.get("blocking_rules", []):
                    history.blocked_by.append((ActionType(action), rule))
                for rule in payload.get("deferring_rules", []):
                    history.deferred_by.append((ActionType(action), rule))
            if payload.get("blocking_rules"):
                history.state = CaseState.GATE_BLOCKED
            elif not payload.get("deferring_rules"):
                history.state = CaseState.GATE_PASSED

        elif event.event_type is EventType.DECISION_MADE:
            history.state = CaseState.DECIDED

        elif event.event_type is EventType.EXECUTION_ATTEMPTED:
            action = payload.get("action")
            if action and payload.get("succeeded"):
                action_type = ActionType(action)
                history.actions_taken.append(action_type)
                if action_type in ATTEMPTING:
                    history.attempts_used += 1
                if action_type in CONTACTING:
                    history.contacts_used += 1
            history.state = CaseState.EXECUTING if payload.get("succeeded") else CaseState.FAILED

        elif event.event_type is EventType.OUTCOME_OBSERVED:
            if payload.get("recovered"):
                history.recovered_paise += int(payload.get("amount_paise", 0))
                history.state = CaseState.RECOVERED
            else:
                history.state = CaseState.ABANDONED

        elif event.event_type is EventType.CASE_TERMINATED:
            history.state = CaseState(payload.get("state", CaseState.ABANDONED.value))

    return history


@dataclass
class Reconciliation:
    """Comparison between a reported total and the ledger's own sum."""

    reported_paise: int
    ledger_paise: int
    cases_in_ledger: int
    cases_recovered: int

    @property
    def difference_paise(self) -> int:
        return self.reported_paise - self.ledger_paise

    @property
    def reconciles(self) -> bool:
        """Exact equality, not a tolerance.

        Money is integer paise throughout (ADR-0003) precisely so this can be
        an equality test. A tolerance here would quietly permit the drift the
        check exists to catch.
        """
        return self.difference_paise == 0


def reconcile(reported_paise: int, histories: Sequence[CaseHistory]) -> Reconciliation:
    return Reconciliation(
        reported_paise=reported_paise,
        ledger_paise=sum(h.recovered_paise for h in histories),
        cases_in_ledger=len(histories),
        cases_recovered=sum(1 for h in histories if h.was_recovered),
    )


def denial_report(histories: Sequence[CaseHistory]) -> Counter[str]:
    """How often each compliance rule blocked an action.

    This is as important an artifact as the recovery total. It is the
    evidence that the agent is bounded — a system that never reports a denial
    is either operating in a world with no rules or is not checking.
    """
    counter: Counter[str] = Counter()
    for history in histories:
        for _action, rule in history.blocked_by:
            counter[rule] += 1
    return counter


def deferral_report(histories: Sequence[CaseHistory]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for history in histories:
        for _action, rule in history.deferred_by:
            counter[rule] += 1
    return counter


def action_mix(histories: Sequence[CaseHistory]) -> Counter[ActionType]:
    counter: Counter[ActionType] = Counter()
    for history in histories:
        for action in history.actions_taken:
            counter[action] += 1
    return counter
