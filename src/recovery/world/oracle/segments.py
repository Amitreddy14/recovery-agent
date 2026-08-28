"""QUARANTINED: segment labels, derived after outcomes are computed.

These labels exist only so evaluation can report *which* population the
policy captured. They are never inputs to generation: `outcomes.py` computes
p(a) from traits and reasons, and a case's segment is read off the resulting
probabilities afterwards.

The direction matters. If generation started from a label - "make this one a
sleeping dog" - then any uplift model fitted downstream would be recovering
our labels rather than learning a causal effect, and every reported number
would be circular.
"""

from __future__ import annotations

from enum import StrEnum

from recovery.domain.enums import ActionType
from recovery.world.oracle.outcomes import PotentialOutcomes

PERSUASION_THRESHOLD = 0.05
"""Minimum absolute uplift to count as persuadable or sleeping dog. Below
this, treatment effect is indistinguishable from modelling noise."""

CERTAINTY_THRESHOLD = 0.65
"""Above this baseline probability a customer is treated as a sure thing."""

FUTILITY_THRESHOLD = 0.08
"""Below this best-case probability the case is a lost cause."""


class Segment(StrEnum):
    SURE_THING = "sure_thing"
    LOST_CAUSE = "lost_cause"
    PERSUADABLE = "persuadable"
    SLEEPING_DOG = "sleeping_dog"
    INDIFFERENT = "indifferent"


CONTACTING = (ActionType.SEND_PAYMENT_LINK, ActionType.PRE_DEBIT_NUDGE)

# Human escalation is excluded from segmentation. It lifts almost any case,
# so including it would classify every customer as persuadable and collapse
# the segment structure entirely. Conceptually it is not a targeting choice
# but an expensive fallback: the question "would an intervention change this
# outcome" is about the automated action set (INC-007).
SEGMENTING_ACTIONS = tuple(
    a for a in ActionType if a not in (ActionType.NO_ACTION, ActionType.ESCALATE_HUMAN)
)


def classify(outcomes: PotentialOutcomes) -> Segment:
    """Read a case's segment off its potential outcomes.

    Order of tests is deliberate. Sleeping dogs are checked *before* sure
    things, because a customer who would have paid anyway and is harmed by
    contact is exactly a sleeping dog - and calling it a sure thing would
    hide the harm the naive policy does.
    """
    baseline = outcomes.recovery_prob[ActionType.NO_ACTION]
    best = max(outcomes.recovery_prob[a] for a in SEGMENTING_ACTIONS)

    worst_contact_uplift = min(outcomes.uplift(a) for a in CONTACTING)
    best_uplift = max(outcomes.uplift(a) for a in SEGMENTING_ACTIONS)

    if worst_contact_uplift < -PERSUASION_THRESHOLD:
        return Segment.SLEEPING_DOG
    if best < FUTILITY_THRESHOLD:
        return Segment.LOST_CAUSE
    if baseline >= CERTAINTY_THRESHOLD and best_uplift < PERSUASION_THRESHOLD:
        return Segment.SURE_THING
    if best_uplift >= PERSUASION_THRESHOLD:
        return Segment.PERSUADABLE
    return Segment.INDIFFERENT
