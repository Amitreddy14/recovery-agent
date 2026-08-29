"""The decision engine.

For each case: score every action, gate every action, choose the best action
that survives the gate. If nothing survives, or if nothing beats doing
nothing, the answer is `NO_ACTION` — and that is recorded as a decision with
reasoning, not as an absence.

Ordering matters and is deliberate. Actions are scored *before* gating, and
the full score table is retained even for actions the gate rejected. A
decision log that only shows the surviving option cannot answer "what would
you have done if the rule had not applied", which is exactly what an auditor
asks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from recovery.compliance.engine import CaseContext, ComplianceEngine, channel_for
from recovery.domain.actions import (
    Action,
    ActionScore,
    ComplianceReview,
    Decision,
    Diagnosis,
)
from recovery.domain.entities import Mandate
from recovery.domain.enums import ActionType, GateVerdict

CANDIDATE_ACTIONS: tuple[ActionType, ...] = (
    ActionType.NO_ACTION,
    ActionType.RETRY_NOW,
    ActionType.RETRY_SCHEDULED,
    ActionType.RETRY_ALTERNATE_RAIL,
    ActionType.SEND_PAYMENT_LINK,
    ActionType.PRE_DEBIT_NUDGE,
)

SCHEDULED_DELAY_HOURS = 26.0
"""Matches the world's scheduled-retry offset. The policy does not know this
number is correct — it learns the value of delay from data — but it must
propose a concrete time for the gate to evaluate against."""


@dataclass(frozen=True)
class CaseDecision:
    """A decision plus the compliance review of every action considered."""

    decision: Decision
    reviews: dict[ActionType, ComplianceReview]
    diagnosis: Diagnosis

    @property
    def blocked_actions(self) -> dict[ActionType, tuple[str, ...]]:
        return {
            action: review.blocking_rules
            for action, review in self.reviews.items()
            if review.blocking_rules
        }

    @property
    def deferred_actions(self) -> dict[ActionType, datetime]:
        out: dict[ActionType, datetime] = {}
        for action, review in self.reviews.items():
            deferrals = [
                r.defer_until
                for r in review.results
                if r.verdict is GateVerdict.DEFER and r.defer_until is not None
            ]
            if deferrals:
                out[action] = max(deferrals)
        return out


def build_action(action_type: ActionType, now: datetime) -> Action:
    scheduled = (
        now + timedelta(hours=SCHEDULED_DELAY_HOURS)
        if action_type is ActionType.RETRY_SCHEDULED
        else None
    )
    return Action(
        action_type=action_type,
        channel=channel_for(action_type),
        scheduled_for=scheduled,
    )


class DecisionEngine:
    def __init__(
        self,
        compliance: ComplianceEngine,
        economics: object,
        *,
        policy_name: str = "uplift_ev",
        model_version: str = "v0.7",
    ) -> None:
        self.compliance = compliance
        self.economics = economics
        self.policy_name = policy_name
        self.model_version = model_version

    def decide(
        self,
        *,
        scores: Sequence[ActionScore],
        context: CaseContext,
        diagnosis: Diagnosis,
        mandate: Mandate | None = None,
    ) -> CaseDecision:
        by_action = {s.action_type: s for s in scores}
        baseline = by_action[ActionType.NO_ACTION]

        reviews: dict[ActionType, ComplianceReview] = {}
        for action_type in CANDIDATE_ACTIONS:
            if action_type not in by_action:
                continue
            reviews[action_type] = self.compliance.review(
                build_action(action_type, context.now), context
            )

        # Only actions that clear every gate are eligible. Deferred actions
        # are not eligible *now*; scheduling them is Phase 8's problem, and
        # pretending otherwise here would let the engine claim value it has
        # not yet earned.
        eligible = [
            by_action[a] for a in CANDIDATE_ACTIONS if a in by_action and reviews[a].allowed
        ]
        if not eligible:
            eligible = [baseline]

        best = max(eligible, key=lambda s: s.expected_value_paise)

        # Doing nothing is the default, not a fallback. An action is taken
        # only if it beats inaction on expected value — which is how a
        # sleeping dog gets left alone even when it ranks highly on uplift.
        if best.expected_value_paise <= baseline.expected_value_paise:
            best = baseline

        rationale = self._explain(best, baseline, by_action, reviews, diagnosis)

        decision = Decision(
            case_id=context.case.case_id,
            decided_at=context.now,
            action=build_action(best.action_type, context.now),
            scores=tuple(by_action[a] for a in CANDIDATE_ACTIONS if a in by_action),
            policy_name=self.policy_name,
            model_version=self.model_version,
            rationale=rationale,
            propensity=1.0,
        )
        return CaseDecision(decision=decision, reviews=reviews, diagnosis=diagnosis)

    @staticmethod
    def _explain(
        chosen: ActionScore,
        baseline: ActionScore,
        by_action: dict[ActionType, ActionScore],
        reviews: dict[ActionType, ComplianceReview],
        diagnosis: Diagnosis,
    ) -> str:
        parts = [f"root cause {diagnosis.root_cause}"]
        if diagnosis.issuer_degraded:
            parts.append("issuer degraded")

        if chosen.action_type is ActionType.NO_ACTION:
            better = [
                (a, s)
                for a, s in by_action.items()
                if a is not ActionType.NO_ACTION
                and s.expected_value_paise > baseline.expected_value_paise
            ]
            if not better:
                parts.append("no action has positive incremental value")
            else:
                # Name the *rule*, not the action. "send_payment_link was
                # gated" tells an auditor nothing they cannot see from the
                # decision itself; "DND_REGISTRY" tells them why.
                for action_type, _ in better:
                    review = reviews[action_type]
                    blocking = list(review.blocking_rules)
                    deferring = [
                        r.rule_id for r in review.results if r.verdict is GateVerdict.DEFER
                    ]
                    if blocking:
                        parts.append(
                            f"{action_type.value} scored above inaction, "
                            f"blocked by {', '.join(sorted(blocking))}"
                        )
                    elif deferring:
                        parts.append(
                            f"{action_type.value} scored above inaction, "
                            f"deferred by {', '.join(sorted(deferring))}"
                        )
            return "; ".join(parts)

        delta = chosen.expected_value_paise - baseline.expected_value_paise
        parts.append(
            f"{chosen.action_type.value} EV {chosen.expected_value_paise:.0f}p "
            f"vs inaction {baseline.expected_value_paise:.0f}p "
            f"(+{delta:.0f}p), uplift {chosen.uplift:+.3f}"
        )
        return "; ".join(parts)
