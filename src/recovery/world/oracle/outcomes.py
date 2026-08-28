"""QUARANTINED: ground-truth potential outcomes.

For every case this computes Y(a) for every action a - including the actions
not taken. In production those counterfactuals are unobservable. Here they
exist, which is the entire reason a synthetic world is the right choice for
this project: it lets us validate an off-policy estimator against truth
before trusting it somewhere truth is unavailable.

CI forbids `diagnose`, `uplift`, `policy`, `compliance`, `execute` and `api`
from importing anything in this package. A violation fails the build.

Design notes that matter for credibility:

* **Coupled draws.** One uniform `u` per case, with `Y(a) = 1 iff u < p(a)`.
  A customer who would pay under a weak action also pays under a strong one,
  which is how real customers behave. Independent draws per action would
  manufacture uplift out of noise.
* **Segments are emergent.** Nothing here assigns a case to "persuadable" or
  "sleeping dog". Those fall out of the relationship between p(a) and p(0)
  after the fact. Labelling first and generating from the label would make
  the uplift model a label-recovery exercise.
* **Not a GLM.** The probability model is a product of thresholds, hazards
  and interactions. A two-model learner fitted downstream has to approximate
  it, not invert it.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from recovery.domain.enums import ActionType, FailureReason
from recovery.world.latent import LatentCustomer
from recovery.world.timeline import IssuerTimeline

# How often a failure resolves with no intervention at all, by reason.
# Bank downtime is high: the customer sees an error and simply tries again.
# A cancelled payment is low: they made a choice.
PASSIVE_RECOVERY: dict[FailureReason, float] = {
    # Transient technical failures are largely self-healing: the customer sees
    # an error, waits, and tries again unprompted. Setting these too low makes
    # the sure-thing segment structurally impossible and flatters every
    # intervention, because there is no population that recovers on its own
    # (INC-007).
    FailureReason.BANK_DOWNTIME: 0.74,
    FailureReason.GATEWAY_TECHNICAL_ERROR: 0.68,
    FailureReason.NETWORK_ERROR: 0.71,
    FailureReason.PAYMENT_TIMEOUT: 0.60,
    FailureReason.INSUFFICIENT_FUNDS: 0.18,
    FailureReason.PAYMENT_LIMIT_EXCEEDED: 0.30,
    FailureReason.INVALID_OTP: 0.38,
    FailureReason.CARD_EXPIRED: 0.06,
    FailureReason.PAYMENT_CANCELLED_BY_USER: 0.08,
    FailureReason.MANDATE_REVOKED: 0.0,
    FailureReason.MANDATE_EXPIRED: 0.0,
    FailureReason.AFA_REQUIRED: 0.12,
    FailureReason.OTHER: 0.25,
}

# Ceiling on what a same-rail retry can achieve, given the reason.
# An expired card cannot be fixed by retrying it.
RETRY_CEILING: dict[FailureReason, float] = {
    FailureReason.BANK_DOWNTIME: 0.92,
    FailureReason.GATEWAY_TECHNICAL_ERROR: 0.88,
    FailureReason.NETWORK_ERROR: 0.90,
    FailureReason.PAYMENT_TIMEOUT: 0.82,
    FailureReason.INSUFFICIENT_FUNDS: 0.75,
    FailureReason.PAYMENT_LIMIT_EXCEEDED: 0.70,
    FailureReason.INVALID_OTP: 0.62,
    FailureReason.CARD_EXPIRED: 0.04,
    FailureReason.PAYMENT_CANCELLED_BY_USER: 0.22,
    FailureReason.MANDATE_REVOKED: 0.0,
    FailureReason.MANDATE_EXPIRED: 0.0,
    FailureReason.AFA_REQUIRED: 0.10,
    FailureReason.OTHER: 0.55,
}

# Reasons an alternate rail genuinely helps. Switching method routes around a
# degraded issuer or a dead instrument; it does nothing for an empty account.
ALT_RAIL_BENEFIT: dict[FailureReason, float] = {
    FailureReason.BANK_DOWNTIME: 0.85,
    FailureReason.GATEWAY_TECHNICAL_ERROR: 0.78,
    FailureReason.NETWORK_ERROR: 0.70,
    FailureReason.CARD_EXPIRED: 0.72,
    FailureReason.PAYMENT_LIMIT_EXCEEDED: 0.60,
    FailureReason.INVALID_OTP: 0.55,
    FailureReason.PAYMENT_TIMEOUT: 0.50,
    FailureReason.INSUFFICIENT_FUNDS: 0.12,
    FailureReason.PAYMENT_CANCELLED_BY_USER: 0.15,
    FailureReason.MANDATE_REVOKED: 0.0,
    FailureReason.MANDATE_EXPIRED: 0.0,
    FailureReason.AFA_REQUIRED: 0.35,
    FailureReason.OTHER: 0.40,
}

# Marginal annoyance cost of each contacting action.
CONTACT_IRRITATION: dict[ActionType, float] = {
    ActionType.SEND_PAYMENT_LINK: 0.30,
    ActionType.PRE_DEBIT_NUDGE: 0.08,
}

SCHEDULED_RETRY_DELAY_HOURS = 26.0
"""Chosen to clear a typical degradation episode and land inside the next
day's balance cycle. The policy must *learn* that this delay helps; it is not
given the number."""


class PotentialOutcomes(BaseModel):
    """Y(a) for every action, plus the collateral damage each would cause."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    recovery_prob: dict[ActionType, float]
    recovered: dict[ActionType, bool]
    mandate_cancelled: dict[ActionType, bool]
    uniform_draw: float = Field(ge=0.0, le=1.0)
    cancel_draw: float = Field(ge=0.0, le=1.0)
    issuer_degraded_at_failure: bool = False
    """Ground truth for scoring the Phase 4 detector.

    Recorded, not generated: this reflects state the timeline had already
    determined, so adding the field moves no draw and shifts no stream
    position. Verified by regenerating and comparing (INC-008)."""

    def uplift(self, action: ActionType) -> float:
        """Causal effect of `action` relative to doing nothing."""
        return self.recovery_prob[action] - self.recovery_prob[ActionType.NO_ACTION]

    def best_action(self, values: dict[ActionType, float]) -> ActionType:
        """Oracle choice given the net value of each action."""
        return max(values, key=lambda a: values[a])


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _effective_intent(latent: LatentCustomer, reason: FailureReason) -> float:
    """Intent, adjusted for what the failure reason reveals about it.

    A cancellation is evidence about intent, not merely a technical event -
    which is why treating it like a downtime failure is the single most
    expensive mistake a naive retry policy makes.
    """
    if reason in (FailureReason.MANDATE_REVOKED, FailureReason.MANDATE_EXPIRED):
        return 0.0
    if reason == FailureReason.PAYMENT_CANCELLED_BY_USER:
        return latent.intent * 0.18
    if reason == FailureReason.INVALID_OTP:
        return latent.intent * 0.86
    return latent.intent


def compute_upcoming_outcomes(
    *,
    case_id: str,
    latent: LatentCustomer,
    days_since_salary: int,
    consecutive_failures: int,
    rng: np.random.Generator,
) -> PotentialOutcomes:
    """Potential outcomes for a mandate that has not yet been debited.

    This population is where sure things live. Roughly nine in ten recurring
    debits succeed untouched, so for most of these cases every intervention
    is pure cost - and a policy that cannot tell them apart from the
    at-risk minority will burn its entire contact budget on customers who
    needed nothing (INC-007).

    It is also where the pre-debit notification earns its keep. RBI requires
    the message regardless; the only question is whether it carries content
    that helps. For a customer whose balance is the binding constraint and
    who responds to messages, it converts a failure into a payment at zero
    incremental contact cost.
    """
    shortfall_risk = _clamp(
        0.10
        + 0.30 * (1.0 - latent.balance_recovery_rate / 0.6)
        + 0.12 * consecutive_failures
        - (0.08 if days_since_salary < 4 else 0.0)
    )
    p_success = _clamp((1.0 - shortfall_risk) * (0.85 + 0.15 * latent.intent))

    p: dict[ActionType, float] = {}
    p[ActionType.NO_ACTION] = p_success

    # Nothing has failed, so retries are not available. They are represented
    # at the do-nothing level rather than omitted, so the action space stays
    # rectangular for the estimators downstream.
    for action in (
        ActionType.RETRY_NOW,
        ActionType.RETRY_SCHEDULED,
        ActionType.RETRY_ALTERNATE_RAIL,
    ):
        p[action] = p_success

    nudge_irritation = CONTACT_IRRITATION[ActionType.PRE_DEBIT_NUDGE]
    topup_gain = shortfall_risk * latent.link_responsiveness * 0.72
    nudge_damage = 1.0 - latent.contact_sensitivity * nudge_irritation
    p[ActionType.PRE_DEBIT_NUDGE] = _clamp((p_success + topup_gain) * nudge_damage)

    link_irritation = CONTACT_IRRITATION[ActionType.SEND_PAYMENT_LINK]
    p[ActionType.SEND_PAYMENT_LINK] = _clamp(
        (p_success + topup_gain * 0.55) * (1.0 - latent.contact_sensitivity * link_irritation)
    )

    p[ActionType.ESCALATE_HUMAN] = _clamp(p_success + topup_gain * 0.9)

    u = float(rng.random())
    v = float(rng.random())
    cancelled = {
        a: v < latent.contact_sensitivity * CONTACT_IRRITATION.get(a, 0.0) * 0.55 for a in p
    }
    return PotentialOutcomes(
        case_id=case_id,
        recovery_prob=p,
        recovered={a: u < prob for a, prob in p.items()},
        mandate_cancelled=cancelled,
        uniform_draw=u,
        cancel_draw=v,
    )


def compute_potential_outcomes(
    *,
    case_id: str,
    latent: LatentCustomer,
    reason: FailureReason,
    issuer: str,
    failed_at: datetime,
    timeline: IssuerTimeline,
    is_mandate: bool,
    rng: np.random.Generator,
) -> PotentialOutcomes:
    """Compute Y(a) for the full action set."""
    intent = _effective_intent(latent, reason)
    passive = PASSIVE_RECOVERY.get(reason, 0.25)
    ceiling = RETRY_CEILING.get(reason, 0.55)

    p: dict[ActionType, float] = {}

    # --- Do nothing -------------------------------------------------------
    p[ActionType.NO_ACTION] = _clamp(intent * passive)

    # --- Retry immediately ------------------------------------------------
    # Retrying into a degraded issuer mostly burns an attempt. This is the
    # mechanism behind the "don't retry into a burning building" thesis.
    degraded_now = timeline.is_degraded(issuer, failed_at)
    health_now = 0.22 if degraded_now else 1.0
    funds_now = (
        0.15 if reason == FailureReason.INSUFFICIENT_FUNDS else 1.0
    )  # balance rarely returns within minutes
    p[ActionType.RETRY_NOW] = _clamp(intent * ceiling * health_now * funds_now)

    # --- Retry later ------------------------------------------------------
    later = failed_at + timedelta(hours=SCHEDULED_RETRY_DELAY_HOURS)
    degraded_later = timeline.is_degraded(issuer, later)
    health_later = 0.22 if degraded_later else 1.0
    if reason == FailureReason.INSUFFICIENT_FUNDS:
        days = SCHEDULED_RETRY_DELAY_HOURS / 24.0
        funds_later = 1.0 - math.exp(-latent.balance_recovery_rate * days)
    else:
        funds_later = 1.0
    decay = 0.94  # intent fades slightly with time
    p[ActionType.RETRY_SCHEDULED] = _clamp(intent * ceiling * health_later * funds_later * decay)

    # --- Switch rail ------------------------------------------------------
    alt = ALT_RAIL_BENEFIT.get(reason, 0.40)
    p[ActionType.RETRY_ALTERNATE_RAIL] = _clamp(intent * alt * latent.rail_flexibility * 0.96)

    # --- Send a payment link ---------------------------------------------
    # Two competing effects: it re-engages a willing customer, and it irritates
    # a sensitive one. Where irritation dominates, p falls below p(NO_ACTION)
    # and the case is a sleeping dog. Nothing labels it as such.
    irritation = CONTACT_IRRITATION[ActionType.SEND_PAYMENT_LINK]
    link_gain = latent.link_responsiveness * (0.55 + 0.45 * ceiling)
    link_damage = 1.0 - latent.contact_sensitivity * irritation
    p[ActionType.SEND_PAYMENT_LINK] = _clamp(intent * link_gain * link_damage)

    # --- Pre-debit nudge (mandates only) ----------------------------------
    # The RBI-mandated notification, used as a channel. Cheap and expected, so
    # it irritates far less - but it only helps where a balance top-up is the
    # binding constraint.
    if is_mandate:
        nudge_irritation = CONTACT_IRRITATION[ActionType.PRE_DEBIT_NUDGE]
        topup = (
            0.62 * latent.link_responsiveness
            if reason == FailureReason.INSUFFICIENT_FUNDS
            else 0.12
        )
        nudge_damage = 1.0 - latent.contact_sensitivity * nudge_irritation
        p[ActionType.PRE_DEBIT_NUDGE] = _clamp(intent * (passive + topup) * nudge_damage)
    else:
        p[ActionType.PRE_DEBIT_NUDGE] = p[ActionType.NO_ACTION]

    # --- Human escalation -------------------------------------------------
    # Effective but expensive; the cost side is priced by the policy, not here.
    p[ActionType.ESCALATE_HUMAN] = _clamp(intent * min(0.90, ceiling + 0.18))

    # --- Realise outcomes with a single coupled draw ----------------------
    u = float(rng.random())
    recovered = {action: (u < prob) for action, prob in p.items()}

    v = float(rng.random())
    cancelled: dict[ActionType, bool] = {}
    for action in p:
        irritation_a = CONTACT_IRRITATION.get(action, 0.0)
        p_cancel = latent.contact_sensitivity * irritation_a * (0.55 if is_mandate else 0.0)
        cancelled[action] = v < p_cancel

    return PotentialOutcomes(
        case_id=case_id,
        recovery_prob=p,
        recovered=recovered,
        mandate_cancelled=cancelled,
        uniform_draw=u,
        cancel_draw=v,
        issuer_degraded_at_failure=degraded_now,
    )
