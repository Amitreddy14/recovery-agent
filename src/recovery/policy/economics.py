"""The economics of a recovery action.

Phase 5 produced a ranking. A ranking cannot decide *whether* to act, only
what to act on first — so on its own it will happily contact a customer whose
expected value is negative, provided that customer ranks above someone worse.

This module converts a ranking into a decision:

    EV(a) = p_recovery(a) x amount
            - cost(a)
            - p_cancel(a) x remaining_mandate_value

The third term is the point of this phase. INC-017 recorded that uplift
targeting selects *more* sleeping dogs than risk ranking (9% against 3%),
because the learner finds the positive tail of the treatment-effect
distribution and is largely blind to the negative one.

Ranking harder was never going to fix that. The damage a sleeping dog suffers
is not a smaller recovery — it is a cancelled mandate, which forfeits every
future debit. Expressed as a probability of recovery it is nearly invisible;
expressed in rupees over a twelve-cycle horizon it dominates the decision. So
we estimate it separately and subtract it.

Cancellation probability is *observable*: `RealizedOutcome.mandate_cancelled`
is recorded for the arm actually taken, so a second uplift model can be fitted
on it with no access to the oracle.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from recovery.domain.actions import ActionScore
from recovery.domain.entities import Mandate
from recovery.domain.enums import ActionType

DEFAULT_HORIZON_CYCLES = 12


class Economics(BaseModel):
    """Cost model, loaded from the compliance policy so that costs and rules
    version together."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    costs_paise: dict[ActionType, int]
    horizon_cycles: int = Field(default=DEFAULT_HORIZON_CYCLES, gt=0)

    @classmethod
    def from_policy(cls, costs: dict[str, int], penalty: dict[str, Any]) -> Economics:
        return cls(
            costs_paise={ActionType(k): int(v) for k, v in costs.items()},
            horizon_cycles=int(penalty.get("horizon_cycles", DEFAULT_HORIZON_CYCLES)),
        )

    def cost_of(self, action: ActionType) -> int:
        return self.costs_paise.get(action, 0)

    def remaining_mandate_value_paise(self, mandate: Mandate | None) -> int:
        """What a cancellation forfeits.

        Deliberately simple: debit amount times horizon, no discounting. A
        discount rate would be a fifth assumption to defend for a second-order
        correction, and the horizon is already swept in Phase 9. The magnitude
        is what matters here — a Rs 499 subscription carries roughly Rs 6,000
        of remaining value, which is over a hundred times the cost of the SMS
        that might destroy it.
        """
        if mandate is None:
            return 0
        return mandate.debit_amount_paise * self.horizon_cycles


def score_action(
    *,
    action: ActionType,
    p_recovery: float,
    uplift: float,
    p_cancel: float,
    amount_paise: int,
    mandate: Mandate | None,
    economics: Economics,
) -> ActionScore:
    """Expected value of one action, in paise.

    `p_recovery` is the absolute probability under this action; `uplift` is
    its effect relative to doing nothing. Both are carried into the score so
    the audit trail records not just what was chosen but the counterfactual
    reasoning behind it.
    """
    gross = p_recovery * amount_paise
    cost = economics.cost_of(action)
    cancellation = p_cancel * economics.remaining_mandate_value_paise(mandate)
    return ActionScore(
        action_type=action,
        p_recovery=max(0.0, min(1.0, p_recovery)),
        uplift=uplift,
        cost_paise=cost,
        expected_value_paise=gross - cost - cancellation,
    )


def incremental_value_paise(score: ActionScore, baseline: ActionScore) -> float:
    """Value of acting *versus* doing nothing.

    The decision is made on this, not on raw expected value. Ranking on
    absolute EV would favour high-value cases that were going to pay anyway,
    which is the sure-thing failure mode: real money on the table, none of it
    attributable to the intervention.
    """
    return score.expected_value_paise - baseline.expected_value_paise
