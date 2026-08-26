"""Actions, decisions, compliance verdicts and outcomes.

Together these form the audit trail: for every case you can reconstruct
*what was decided, why, what the gate said, what was executed, and what
happened* — with no gaps.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from recovery.domain.entities import Frozen, Paise
from recovery.domain.enums import ActionType, Channel, GateVerdict

# Actions that touch the customer. Used by the contact-budget gate.
CONTACTING_ACTIONS: frozenset[ActionType] = frozenset(
    {
        ActionType.SEND_PAYMENT_LINK,
        ActionType.PRE_DEBIT_NUDGE,
    }
)

# Actions that consume a payment attempt against the rail.
ATTEMPTING_ACTIONS: frozenset[ActionType] = frozenset(
    {
        ActionType.RETRY_NOW,
        ActionType.RETRY_SCHEDULED,
        ActionType.RETRY_ALTERNATE_RAIL,
    }
)


class Action(Frozen):
    """A concrete, fully-specified action. No free-form fields: the executor
    can only act on what is representable here."""

    action_type: ActionType
    channel: Channel = Channel.NONE
    scheduled_for: datetime | None = None
    alternate_method: str | None = None
    message_template_id: str | None = None
    # Rendered copy, when a message is involved. Drafted by the LLM node,
    # but the template id and the gating happen deterministically.
    message_body: str | None = None

    @property
    def consumes_contact(self) -> bool:
        return self.action_type in CONTACTING_ACTIONS

    @property
    def consumes_attempt(self) -> bool:
        return self.action_type in ATTEMPTING_ACTIONS


class Diagnosis(Frozen):
    """Output of the diagnosis layer."""

    case_id: str
    root_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    recoverable_without_contact: bool
    issuer_degraded: bool = False
    evidence: tuple[str, ...] = ()
    narration: str | None = None  # LLM-written, never load-bearing


class ActionScore(Frozen):
    """Per-action economics. The decision is argmax over these."""

    action_type: ActionType
    p_recovery: float = Field(ge=0.0, le=1.0)
    uplift: float  # may be negative — sleeping dogs
    cost_paise: Paise = Field(ge=0)
    expected_value_paise: float


class Decision(BaseModel):
    """Chosen action plus the full scoring table that produced it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    decided_at: datetime
    action: Action
    scores: tuple[ActionScore, ...]
    policy_name: str
    model_version: str
    rationale: str
    # Probability with which the logging policy chose this action.
    # Required for unbiased off-policy evaluation; never optional.
    propensity: float = Field(gt=0.0, le=1.0)


class GateResult(Frozen):
    """Verdict of one compliance rule."""

    rule_id: str
    verdict: GateVerdict
    reason: str
    defer_until: datetime | None = None


class ComplianceReview(Frozen):
    """Aggregate of all gates for one proposed action."""

    case_id: str
    reviewed_at: datetime
    results: tuple[GateResult, ...]

    @property
    def allowed(self) -> bool:
        return all(r.verdict == GateVerdict.ALLOW for r in self.results)

    @property
    def blocking_rules(self) -> tuple[str, ...]:
        return tuple(r.rule_id for r in self.results if r.verdict == GateVerdict.BLOCK)


class ExecutionRecord(Frozen):
    """One real API interaction."""

    case_id: str
    idempotency_key: str
    action_type: ActionType
    executed_at: datetime
    api_endpoint: str
    request_digest: str
    succeeded: bool
    response_code: str | None = None
    error_reason: str | None = None
    retry_count: int = Field(default=0, ge=0)


class Outcome(Frozen):
    """What actually happened to the money."""

    case_id: str
    observed_at: datetime
    recovered: bool
    recovered_amount_paise: Paise = Field(default=0, ge=0)
    intervention_cost_paise: Paise = Field(default=0, ge=0)
    mandate_cancelled: bool = False  # sleeping-dog damage
    contacts_spent: int = Field(default=0, ge=0)
    attempts_spent: int = Field(default=0, ge=0)

    @property
    def net_paise(self) -> int:
        return self.recovered_amount_paise - self.intervention_cost_paise
