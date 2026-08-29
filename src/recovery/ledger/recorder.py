"""Writing the trail.

One recorder call per stage of the pipeline. The recorder is deliberately the
only path into the store: scattering `store.append(Event(...))` through the
decision code would make it easy to add a stage and forget to record it, and
a trail with a missing stage is worse than none because the gap is invisible.
"""

from __future__ import annotations

from datetime import datetime

from recovery.domain.actions import (
    ComplianceReview,
    Decision,
    Diagnosis,
    ExecutionRecord,
)
from recovery.domain.enums import ActionType, CaseState, GateVerdict
from recovery.domain.observations import CaseFeatures
from recovery.ledger.events import Event, EventType
from recovery.ledger.store import EventStore


class AuditRecorder:
    def __init__(self, store: EventStore) -> None:
        self.store = store

    def ingested(self, features: CaseFeatures) -> Event:
        return self.store.append(
            Event(
                case_id=features.case_id,
                event_type=EventType.CASE_INGESTED,
                occurred_at=features.created_at,
                payload={
                    "case_type": features.case_type.value,
                    "amount_paise": features.amount_paise,
                    "reason": features.reason.value,
                    "issuer": features.issuer,
                    "method": features.method.value,
                    "customer_id": features.customer_id,
                },
            )
        )

    def diagnosed(self, diagnosis: Diagnosis, at: datetime) -> Event:
        return self.store.append(
            Event(
                case_id=diagnosis.case_id,
                event_type=EventType.CASE_DIAGNOSED,
                occurred_at=at,
                payload={
                    "root_cause": diagnosis.root_cause,
                    "confidence": diagnosis.confidence,
                    "issuer_degraded": diagnosis.issuer_degraded,
                    "recoverable_without_contact": diagnosis.recoverable_without_contact,
                    # The evidence is the point. A diagnosis without it is an
                    # assertion, and an auditor cannot check an assertion.
                    "evidence": list(diagnosis.evidence),
                },
            )
        )

    def scored(self, decision: Decision) -> Event:
        return self.store.append(
            Event(
                case_id=decision.case_id,
                event_type=EventType.ACTIONS_SCORED,
                occurred_at=decision.decided_at,
                payload={
                    "model_version": decision.model_version,
                    "policy": decision.policy_name,
                    # Every action scored, including the rejected ones. An
                    # auditor asking "what would you have done otherwise" needs
                    # the alternatives, not just the winner.
                    "scores": [
                        {
                            "action_type": s.action_type.value,
                            "p_recovery": s.p_recovery,
                            "uplift": s.uplift,
                            "cost_paise": s.cost_paise,
                            "expected_value_paise": s.expected_value_paise,
                        }
                        for s in decision.scores
                    ],
                },
            )
        )

    def reviewed(self, review: ComplianceReview, action: ActionType) -> Event:
        results = [
            {
                "rule_id": r.rule_id,
                "verdict": r.verdict.value,
                "reason": r.reason,
                "defer_until": r.defer_until.isoformat() if r.defer_until else None,
            }
            for r in review.results
        ]
        return self.store.append(
            Event(
                case_id=review.case_id,
                event_type=EventType.COMPLIANCE_REVIEWED,
                occurred_at=review.reviewed_at,
                payload={
                    "action": action.value,
                    "rules_checked": len(results),
                    "results": results,
                    "blocking_rules": [
                        r["rule_id"] for r in results if r["verdict"] == GateVerdict.BLOCK.value
                    ],
                    "deferring_rules": [
                        r["rule_id"] for r in results if r["verdict"] == GateVerdict.DEFER.value
                    ],
                },
            )
        )

    def decided(self, decision: Decision) -> Event:
        return self.store.append(
            Event(
                case_id=decision.case_id,
                event_type=EventType.DECISION_MADE,
                occurred_at=decision.decided_at,
                payload={
                    "action": decision.action.action_type.value,
                    "channel": decision.action.channel.value,
                    "scheduled_for": (
                        decision.action.scheduled_for.isoformat()
                        if decision.action.scheduled_for
                        else None
                    ),
                    "rationale": decision.rationale,
                    "policy": decision.policy_name,
                    "model_version": decision.model_version,
                    "propensity": decision.propensity,
                },
            )
        )

    def executed(
        self,
        record: ExecutionRecord,
        *,
        provider_reference: str | None = None,
        skipped_reason: str | None = None,
        payment_link_url: str | None = None,
    ) -> Event:
        return self.store.append(
            Event(
                case_id=record.case_id,
                event_type=EventType.EXECUTION_ATTEMPTED,
                occurred_at=record.executed_at,
                payload={
                    "action": record.action_type.value,
                    "succeeded": record.succeeded,
                    "endpoint": record.api_endpoint,
                    "idempotency_key": record.idempotency_key,
                    "provider_reference": provider_reference,
                    "response_code": record.response_code,
                    "error_reason": record.error_reason,
                    "retry_count": record.retry_count,
                    "skipped_reason": skipped_reason,
                    "payment_link_url": payment_link_url,
                },
            )
        )

    def outcome(
        self,
        case_id: str,
        *,
        recovered: bool,
        amount_paise: int,
        at: datetime,
        note: str = "",
    ) -> Event:
        return self.store.append(
            Event(
                case_id=case_id,
                event_type=EventType.OUTCOME_OBSERVED,
                occurred_at=at,
                payload={
                    "recovered": recovered,
                    "amount_paise": amount_paise if recovered else 0,
                    "note": note,
                },
            )
        )

    def terminated(self, case_id: str, state: CaseState, at: datetime, why: str) -> Event:
        return self.store.append(
            Event(
                case_id=case_id,
                event_type=EventType.CASE_TERMINATED,
                occurred_at=at,
                payload={"state": state.value, "why": why},
            )
        )
