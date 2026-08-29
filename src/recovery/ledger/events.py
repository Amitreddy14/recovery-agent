"""Audit events.

The ledger is append-only and case state is *derived* from it, never stored
and mutated. That choice is the difference between an audit trail and a
status column.

A mutable `case.state` field answers "where is this case now". It cannot
answer "what did the agent consider, what did it reject, and on what
evidence" — and those are the questions an auditor actually asks. Worse, a
mutable field can be corrected after the fact with no trace, which makes it
worthless as evidence precisely when evidence matters.

Every event here is immutable and carries its own timestamp and sequence
position. Nothing is ever updated or deleted. A case that went wrong leaves
the same record as one that went right.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from recovery.domain.enums import ActionType, CaseState, GateVerdict


class EventType(StrEnum):
    CASE_INGESTED = "case_ingested"
    CASE_DIAGNOSED = "case_diagnosed"
    ACTIONS_SCORED = "actions_scored"
    COMPLIANCE_REVIEWED = "compliance_reviewed"
    DECISION_MADE = "decision_made"
    EXECUTION_ATTEMPTED = "execution_attempted"
    OUTCOME_OBSERVED = "outcome_observed"
    CASE_TERMINATED = "case_terminated"


class Event(BaseModel):
    """One immutable fact about one case.

    `payload` is deliberately loose. A typed field per event variant would be
    cleaner today and would break every historical record the first time an
    event gains a field. An audit log has to be readable years after the code
    that wrote it changed, so the envelope is fixed and the payload is not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    event_type: EventType
    occurred_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    sequence: int | None = None
    """Assigned by the store on append. None until persisted."""

    def summary(self) -> str:
        """One line for a human reading the trail."""
        return _SUMMARISERS.get(self.event_type, _default_summary)(self)


def _default_summary(event: Event) -> str:
    return event.event_type.value


def _ingested(event: Event) -> str:
    p = event.payload
    return (
        f"ingested {p.get('case_type', '?')} "
        f"Rs {int(p.get('amount_paise', 0)) / 100:,.2f} "
        f"reason={p.get('reason', '?')} issuer={p.get('issuer', '?')}"
    )


def _diagnosed(event: Event) -> str:
    p = event.payload
    degraded = " ISSUER DEGRADED" if p.get("issuer_degraded") else ""
    return (
        f"diagnosed {p.get('root_cause', '?')} "
        f"(confidence {float(p.get('confidence', 0)):.2f}){degraded}"
    )


def _scored(event: Event) -> str:
    scores = event.payload.get("scores", [])
    best = max(scores, key=lambda s: s["expected_value_paise"], default=None)
    if best is None:
        return "scored 0 actions"
    return (
        f"scored {len(scores)} actions; best {best['action_type']} "
        f"EV Rs {best['expected_value_paise'] / 100:,.2f}"
    )


def _reviewed(event: Event) -> str:
    p = event.payload
    blocked = p.get("blocking_rules", [])
    deferred = p.get("deferring_rules", [])
    if blocked:
        return f"{p.get('action', '?')} BLOCKED by {', '.join(blocked)}"
    if deferred:
        return f"{p.get('action', '?')} DEFERRED by {', '.join(deferred)}"
    return f"{p.get('action', '?')} passed {p.get('rules_checked', 0)} gates"


def _decided(event: Event) -> str:
    p = event.payload
    return f"decided {p.get('action', '?')} — {p.get('rationale', '')}"


def _executed(event: Event) -> str:
    p = event.payload
    status = "ok" if p.get("succeeded") else "FAILED"
    ref = p.get("provider_reference") or "-"
    extra = f" ({p['skipped_reason']})" if p.get("skipped_reason") else ""
    return f"executed {p.get('action', '?')} {status} ref={ref}{extra}"


def _outcome(event: Event) -> str:
    p = event.payload
    if p.get("recovered"):
        return f"RECOVERED Rs {int(p.get('amount_paise', 0)) / 100:,.2f}"
    return f"not recovered ({p.get('note', 'no further action')})"


def _terminated(event: Event) -> str:
    return f"closed as {event.payload.get('state', '?')}"


_SUMMARISERS = {
    EventType.CASE_INGESTED: _ingested,
    EventType.CASE_DIAGNOSED: _diagnosed,
    EventType.ACTIONS_SCORED: _scored,
    EventType.COMPLIANCE_REVIEWED: _reviewed,
    EventType.DECISION_MADE: _decided,
    EventType.EXECUTION_ATTEMPTED: _executed,
    EventType.OUTCOME_OBSERVED: _outcome,
    EventType.CASE_TERMINATED: _terminated,
}


TERMINAL_STATES: frozenset[CaseState] = frozenset(
    {CaseState.RECOVERED, CaseState.ABANDONED, CaseState.FAILED, CaseState.ESCALATED}
)


def gate_rules_by_verdict(results: list[dict[str, Any]], verdict: GateVerdict) -> list[str]:
    return [r["rule_id"] for r in results if r.get("verdict") == verdict.value]


def action_of(event: Event) -> ActionType | None:
    raw = event.payload.get("action")
    return ActionType(raw) if raw else None
