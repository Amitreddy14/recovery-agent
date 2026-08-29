"""Batch policy evaluation.

This is where the project's headline number is produced: rupees recovered
across a batch, against three baselines and an oracle ceiling.

Counterfactual evaluation is exact here rather than estimated. Because the
world computed Y(a) for every action, we can ask what *would* have happened
under any policy on the same cases with the same random draws. That is not
available in production — which is precisely why the off-policy estimators
are validated against it rather than trusted blindly.

Four policies are compared:

* **blind_retry** — what most merchants do. Retry everything, message on
  repeat failure. The behaviour the naive logging policy encodes.
* **risk_ranking** — contact the top-N by predicted failure probability. The
  standard "smart" approach, and the one INC-017 showed spends over half its
  budget on lost causes.
* **uplift_ev** — this project's policy: uplift scores priced into expected
  value, gated by compliance.
* **oracle** — the best achievable given full knowledge of every
  counterfactual. Reported as a ceiling, never as a claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from recovery.domain.enums import ActionType
from recovery.world.oracle.outcomes import PotentialOutcomes
from recovery.world.oracle.segments import Segment, classify


@dataclass
class PolicyOutcome:
    """Realised result of running one policy over a batch."""

    name: str
    recovered_paise: int = 0
    intervention_cost_paise: int = 0
    cancellation_loss_paise: int = 0
    contacts: int = 0
    attempts: int = 0
    cases_actioned: int = 0
    cases_total: int = 0
    mandates_cancelled: int = 0
    gate_blocks: int = 0
    segment_contacted: dict[Segment, int] = field(default_factory=dict)

    @property
    def net_paise(self) -> int:
        return self.recovered_paise - self.intervention_cost_paise - self.cancellation_loss_paise

    @property
    def cost_per_rupee_recovered(self) -> float:
        if self.recovered_paise == 0:
            return float("inf")
        return (self.intervention_cost_paise + self.cancellation_loss_paise) / self.recovered_paise

    @property
    def recovery_rate(self) -> float:
        return self.cases_actioned / self.cases_total if self.cases_total else 0.0

    def rupees(self, paise: int) -> float:
        return paise / 100.0


def _apply(
    outcome: PolicyOutcome,
    *,
    action: ActionType,
    potential: PotentialOutcomes,
    amount_paise: int,
    mandate_value_paise: int,
    cost_paise: int,
    segment: Segment,
) -> None:
    """Charge one case's realised consequences to a policy's ledger."""
    outcome.cases_total += 1
    if action is not ActionType.NO_ACTION:
        outcome.cases_actioned += 1
    outcome.intervention_cost_paise += cost_paise

    if action in (ActionType.SEND_PAYMENT_LINK, ActionType.PRE_DEBIT_NUDGE):
        outcome.contacts += 1
        outcome.segment_contacted[segment] = outcome.segment_contacted.get(segment, 0) + 1
    if action in (
        ActionType.RETRY_NOW,
        ActionType.RETRY_SCHEDULED,
        ActionType.RETRY_ALTERNATE_RAIL,
    ):
        outcome.attempts += 1

    if potential.recovered.get(action, False):
        outcome.recovered_paise += amount_paise

    # The sleeping-dog cost, charged where it actually falls. A cancelled
    # mandate forfeits every future debit, which is invisible in a recovery
    # rate and dominant in rupees.
    if potential.mandate_cancelled.get(action, False):
        outcome.mandates_cancelled += 1
        outcome.cancellation_loss_paise += mandate_value_paise


def evaluate_policies(
    policies: dict[str, Sequence[ActionType]],
    outcomes: Sequence[PotentialOutcomes],
    amounts_paise: Sequence[int],
    mandate_values_paise: Sequence[int],
    costs_paise: dict[ActionType, int],
) -> dict[str, PolicyOutcome]:
    """Run every policy over the same cases and the same draws.

    Each policy supplies one action per case, in case order. Holding the
    world fixed across policies is what makes the comparison a controlled
    experiment rather than four separate measurements.
    """
    results: dict[str, PolicyOutcome] = {}
    segments = [classify(o) for o in outcomes]

    for name, actions in policies.items():
        if len(actions) != len(outcomes):
            raise ValueError(
                f"policy {name} produced {len(actions)} actions for {len(outcomes)} cases"
            )
        outcome = PolicyOutcome(name=name)
        for i, action in enumerate(actions):
            _apply(
                outcome,
                action=action,
                potential=outcomes[i],
                amount_paise=amounts_paise[i],
                mandate_value_paise=mandate_values_paise[i],
                cost_paise=costs_paise.get(action, 0),
                segment=segments[i],
            )
        results[name] = outcome
    return results


def oracle_actions(
    outcomes: Sequence[PotentialOutcomes],
    amounts_paise: Sequence[int],
    mandate_values_paise: Sequence[int],
    costs_paise: dict[ActionType, int],
    candidates: Sequence[ActionType],
) -> list[ActionType]:
    """The best action per case given full knowledge of the counterfactuals.

    Uses probabilities rather than realised draws: choosing on the realised
    outcome would let the oracle exploit the specific random draw rather than
    the underlying structure, producing a ceiling no policy could approach
    even in principle. That would flatter our percentage-of-oracle figure by
    inflating the denominator.
    """
    chosen: list[ActionType] = []
    for i, potential in enumerate(outcomes):
        best_action = ActionType.NO_ACTION
        best_value = float("-inf")
        for action in candidates:
            p = potential.recovery_prob.get(action)
            if p is None:
                continue
            cancel_risk = (
                mandate_values_paise[i] if potential.mandate_cancelled.get(action, False) else 0
            )
            value = p * amounts_paise[i] - costs_paise.get(action, 0) - cancel_risk
            if value > best_value:
                best_value, best_action = value, action
        chosen.append(best_action)
    return chosen


def blind_retry_actions(
    case_types: Sequence[str], consecutive_failures: Sequence[int]
) -> list[ActionType]:
    """What a typical merchant does: retry everything, message on repeats."""
    actions: list[ActionType] = []
    for case_type, failures in zip(case_types, consecutive_failures, strict=True):
        if case_type == "upcoming_at_risk":
            actions.append(ActionType.PRE_DEBIT_NUDGE)
        elif failures >= 2:
            actions.append(ActionType.SEND_PAYMENT_LINK)
        else:
            actions.append(ActionType.RETRY_NOW)
    return actions


def top_n_actions(
    scores: Sequence[float],
    budget: int,
    action: ActionType = ActionType.SEND_PAYMENT_LINK,
) -> list[ActionType]:
    """Contact the top `budget` cases by score, do nothing otherwise.

    This is how risk ranking is operationalised: a ranking plus a budget. It
    has no way to decline to spend the budget, which is the structural
    difference from an expected-value policy.
    """
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:budget]
    selected = set(order)
    return [action if i in selected else ActionType.NO_ACTION for i in range(len(scores))]
