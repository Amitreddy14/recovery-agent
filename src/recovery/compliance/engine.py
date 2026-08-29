"""The compliance gate.

Every proposed action passes through here before execution. The rules are not
written in this file — they are loaded from `configs/compliance/policy.yaml`,
where each carries a citation and is marked `regulation` or
`operating_policy` (ADR-0004). This module is the interpreter.

Two properties matter more than the individual rules:

* **Denials are recorded, not just enforced.** A blocked action produces a
  `GateResult` naming the rule that fired and why. The denial log is the
  evidence that the agent is bounded; without it, "we respect the rules" is
  an assertion.
* **Deferral is distinct from denial.** An action outside permitted contact
  hours is not forbidden, it is early. Collapsing the two would either send
  messages at 3am or discard recoverable cases for a reason that expires.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, ConfigDict

from recovery.domain.actions import Action, ComplianceReview, GateResult
from recovery.domain.entities import Mandate, RecoveryCase
from recovery.domain.enums import ActionType, Channel, GateVerdict

DEFAULT_POLICY_PATH = Path("configs/compliance/policy.yaml")


class CaseContext(BaseModel):
    """Everything the gate needs beyond the action itself."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case: RecoveryCase
    mandate: Mandate | None = None
    now: datetime
    contacts_last_7d: int = 0
    contacts_last_30d: int = 0
    hours_since_last_contact: float | None = None
    dnd_registered: bool = False
    pre_debit_notice_sent_at: datetime | None = None


class CompliancePolicy(BaseModel):
    """Parsed rule set. Structure mirrors the YAML exactly so a reviewer can
    read one against the other."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    rules: tuple[dict[str, Any], ...]
    costs_paise: dict[str, int]
    sleeping_dog_penalty: dict[str, Any]

    def rule(self, rule_id: str) -> dict[str, Any]:
        for r in self.rules:
            if r["id"] == rule_id:
                return r
        raise KeyError(f"no rule {rule_id} in policy {self.version}")

    def params(self, rule_id: str) -> dict[str, Any]:
        return dict(self.rule(rule_id).get("params", {}))

    @property
    def rule_ids(self) -> tuple[str, ...]:
        return tuple(r["id"] for r in self.rules)


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> CompliancePolicy:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return CompliancePolicy(
        version=str(raw["version"]),
        rules=tuple(raw["rules"]),
        costs_paise=dict(raw["costs_paise"]),
        sleeping_dog_penalty=dict(raw["sleeping_dog_penalty"]),
    )


class ComplianceEngine:
    def __init__(self, policy: CompliancePolicy) -> None:
        self.policy = policy

    @classmethod
    def load(cls, path: Path = DEFAULT_POLICY_PATH) -> ComplianceEngine:
        return cls(load_policy(path))

    # -- entry point -------------------------------------------------------

    def review(self, action: Action, context: CaseContext) -> ComplianceReview:
        """Run every applicable rule. All of them, always — a short-circuit on
        the first denial would leave the audit trail unable to answer "what
        else was wrong with this?"."""
        results: list[GateResult] = [
            self._opt_out(action, context),
            self._dispute(action, context),
            self._promise_to_pay(action, context),
            self._mandate_live(action, context),
            self._afa_threshold(action, context),
            self._pre_debit_notification(action, context),
            self._attempt_budget(action, context),
            self._contact_budget(action, context),
            self._dnd(action, context),
            self._contact_hours(action, context),
        ]
        return ComplianceReview(
            case_id=context.case.case_id,
            reviewed_at=context.now,
            results=tuple(r for r in results if r is not None),
        )

    # -- hard stops --------------------------------------------------------

    def _opt_out(self, action: Action, ctx: CaseContext) -> GateResult:
        rid = "RBI_OPT_OUT_HONOURED"
        if ctx.case.customer_opted_out and action.action_type is not ActionType.NO_ACTION:
            return GateResult(
                rule_id=rid,
                verdict=GateVerdict.BLOCK,
                reason="customer opted out of this mandate or transaction",
            )
        return GateResult(rule_id=rid, verdict=GateVerdict.ALLOW, reason="no opt-out")

    def _dispute(self, action: Action, ctx: CaseContext) -> GateResult:
        rid = "DISPUTE_RAISED"
        if ctx.case.dispute_raised and action.action_type is not ActionType.NO_ACTION:
            return GateResult(
                rule_id=rid,
                verdict=GateVerdict.BLOCK,
                reason="open dispute freezes recovery activity",
            )
        return GateResult(rule_id=rid, verdict=GateVerdict.ALLOW, reason="no dispute")

    def _promise_to_pay(self, action: Action, ctx: CaseContext) -> GateResult:
        rid = "PROMISE_TO_PAY_ACTIVE"
        promise = ctx.case.promise_to_pay_at
        if promise is None or action.action_type is ActionType.NO_ACTION:
            return GateResult(rule_id=rid, verdict=GateVerdict.ALLOW, reason="no promise")
        grace = timedelta(hours=float(self.policy.params(rid)["grace_hours"]))
        if ctx.now < promise + grace:
            return GateResult(
                rule_id=rid,
                verdict=GateVerdict.DEFER,
                reason=f"promise to pay by {promise.isoformat()}; holding contact",
                defer_until=promise + grace,
            )
        return GateResult(rule_id=rid, verdict=GateVerdict.ALLOW, reason="promise lapsed")

    def _mandate_live(self, action: Action, ctx: CaseContext) -> GateResult:
        rid = "MANDATE_REVOKED"
        if not action.consumes_attempt or ctx.mandate is None:
            return GateResult(rule_id=rid, verdict=GateVerdict.ALLOW, reason="n/a")
        if not ctx.mandate.active:
            return GateResult(rule_id=rid, verdict=GateVerdict.BLOCK, reason="mandate revoked")
        if ctx.mandate.valid_until <= ctx.now:
            return GateResult(
                rule_id=rid,
                verdict=GateVerdict.BLOCK,
                reason=f"mandate expired {ctx.mandate.valid_until.date()}",
            )
        return GateResult(rule_id=rid, verdict=GateVerdict.ALLOW, reason="mandate live")

    # -- RBI e-mandate framework -------------------------------------------

    def _afa_threshold(self, action: Action, ctx: CaseContext) -> GateResult:
        """Above the AFA-free ceiling the agent must not debit silently.

        Note this blocks rather than defers. Waiting does not make an
        unauthenticated debit permissible; the case has to move to an
        authenticated flow, which is a human decision.
        """
        rid = "RBI_AFA_THRESHOLD"
        if not action.consumes_attempt or ctx.mandate is None:
            return GateResult(rule_id=rid, verdict=GateVerdict.ALLOW, reason="n/a")
        if ctx.mandate.requires_afa:
            ceiling = ctx.mandate.afa_free_ceiling_paise
            return GateResult(
                rule_id=rid,
                verdict=GateVerdict.BLOCK,
                reason=(
                    f"debit {ctx.mandate.debit_amount_paise} paise exceeds AFA-free "
                    f"ceiling {ceiling} for category "
                    f"{ctx.mandate.category.value}; requires authenticated flow"
                ),
            )
        return GateResult(rule_id=rid, verdict=GateVerdict.ALLOW, reason="below AFA-free ceiling")

    def _pre_debit_notification(self, action: Action, ctx: CaseContext) -> GateResult:
        """A recurring debit requires notice at least 24 hours ahead.

        A scheduled retry that lands sooner than the notice period is deferred
        to the earliest compliant moment rather than blocked: the debit is
        permissible, just not yet.
        """
        rid = "RBI_PRE_DEBIT_NOTIFICATION"
        params = self.policy.params(rid)
        lead = timedelta(hours=float(params["min_lead_hours"]))

        if ctx.mandate is None or action.action_type is not ActionType.RETRY_SCHEDULED:
            return GateResult(rule_id=rid, verdict=GateVerdict.ALLOW, reason="n/a")

        target = action.scheduled_for or ctx.now
        notice = ctx.pre_debit_notice_sent_at
        if notice is not None and target - notice >= lead:
            return GateResult(
                rule_id=rid,
                verdict=GateVerdict.ALLOW,
                reason=f"notice sent {notice.isoformat()}, {lead} ahead",
            )
        earliest = ctx.now + lead
        return GateResult(
            rule_id=rid,
            verdict=GateVerdict.DEFER,
            reason=(
                f"no compliant pre-debit notice; earliest permissible debit {earliest.isoformat()}"
            ),
            defer_until=earliest,
        )

    # -- budgets and conduct ------------------------------------------------

    def _attempt_budget(self, action: Action, ctx: CaseContext) -> GateResult:
        rid = "ATTEMPT_BUDGET"
        if not action.consumes_attempt:
            return GateResult(rule_id=rid, verdict=GateVerdict.ALLOW, reason="n/a")
        cap = int(self.policy.params(rid)["max_per_case"])
        used = ctx.case.attempts_used
        if used >= cap:
            return GateResult(
                rule_id=rid,
                verdict=GateVerdict.BLOCK,
                reason=f"{used}/{cap} attempts already used",
            )
        return GateResult(
            rule_id=rid, verdict=GateVerdict.ALLOW, reason=f"{used}/{cap} attempts used"
        )

    def _contact_budget(self, action: Action, ctx: CaseContext) -> GateResult:
        rid = "CONTACT_FREQUENCY_CAP"
        if not action.consumes_contact:
            return GateResult(rule_id=rid, verdict=GateVerdict.ALLOW, reason="n/a")
        p = self.policy.params(rid)
        if ctx.case.contacts_used >= int(p["max_per_case"]):
            return GateResult(
                rule_id=rid,
                verdict=GateVerdict.BLOCK,
                reason=f"case contact cap {p['max_per_case']} reached",
            )
        if ctx.contacts_last_7d >= int(p["max_per_customer_7d"]):
            return GateResult(
                rule_id=rid,
                verdict=GateVerdict.BLOCK,
                reason=f"customer 7d cap {p['max_per_customer_7d']} reached",
            )
        if ctx.contacts_last_30d >= int(p["max_per_customer_30d"]):
            return GateResult(
                rule_id=rid,
                verdict=GateVerdict.BLOCK,
                reason=f"customer 30d cap {p['max_per_customer_30d']} reached",
            )
        gap = float(p["min_gap_hours"])
        if ctx.hours_since_last_contact is not None and ctx.hours_since_last_contact < gap:
            return GateResult(
                rule_id=rid,
                verdict=GateVerdict.DEFER,
                reason=f"only {ctx.hours_since_last_contact:.1f}h since last contact",
                defer_until=ctx.now + timedelta(hours=gap - ctx.hours_since_last_contact),
            )
        return GateResult(rule_id=rid, verdict=GateVerdict.ALLOW, reason="within caps")

    def _dnd(self, action: Action, ctx: CaseContext) -> GateResult:
        """DND blocks promotional contact. The pre-debit notification is a
        transactional message required by the e-mandate framework, so it is
        exempt — the customer cannot opt out of being told money is about to
        leave their account."""
        rid = "DND_REGISTRY"
        exempt = set(self.policy.params(rid).get("exempt_action_types", []))
        if not action.consumes_contact:
            return GateResult(rule_id=rid, verdict=GateVerdict.ALLOW, reason="n/a")
        if action.action_type.value in exempt:
            return GateResult(
                rule_id=rid,
                verdict=GateVerdict.ALLOW,
                reason="transactional notification, exempt from DND",
            )
        if ctx.dnd_registered:
            return GateResult(
                rule_id=rid,
                verdict=GateVerdict.BLOCK,
                reason="customer on DND registry",
            )
        return GateResult(rule_id=rid, verdict=GateVerdict.ALLOW, reason="not on DND")

    def _contact_hours(self, action: Action, ctx: CaseContext) -> GateResult:
        rid = "CONTACT_HOURS"
        if not action.consumes_contact:
            return GateResult(rule_id=rid, verdict=GateVerdict.ALLOW, reason="n/a")
        p = self.policy.params(rid)
        tz = ZoneInfo(str(p["timezone"]))
        local = ctx.now.astimezone(tz)
        earliest = time.fromisoformat(str(p["earliest"]))
        latest = time.fromisoformat(str(p["latest"]))

        if earliest <= local.time() < latest:
            return GateResult(
                rule_id=rid,
                verdict=GateVerdict.ALLOW,
                reason=f"{local.strftime('%H:%M')} local, within window",
            )
        next_open = local.replace(
            hour=earliest.hour, minute=earliest.minute, second=0, microsecond=0
        )
        if local.time() >= latest:
            next_open = next_open + timedelta(days=1)
        return GateResult(
            rule_id=rid,
            verdict=GateVerdict.DEFER,
            reason=f"{local.strftime('%H:%M')} local, outside {p['earliest']}-{p['latest']}",
            defer_until=next_open.astimezone(UTC),
        )


def channel_for(action_type: ActionType) -> Channel:
    """Default channel per action. Retries are silent and consume no contact
    budget; that asymmetry is what makes well-timed retries the cheapest money
    in the system."""
    if action_type is ActionType.SEND_PAYMENT_LINK:
        return Channel.SMS
    if action_type is ActionType.PRE_DEBIT_NUDGE:
        return Channel.SMS
    return Channel.NONE
