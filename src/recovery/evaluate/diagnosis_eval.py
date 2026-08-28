"""Scoring the diagnosis layer against ground truth.

This module may read the oracle. Nothing in `diagnose` may — the detector is
scored here, never informed here.

Accuracy is deliberately not the headline. Degradation is rare, so a detector
that never fires scores well on accuracy and is worthless. The reported
metrics are precision, recall and, most importantly, the **cost** of being
wrong in each direction, which differ by roughly an order of magnitude:

* A **false positive** suppresses an immediate retry that would have worked.
  The case is deferred, not lost — cost is a delay plus one wasted slot.
* A **false negative** retries into a degraded issuer. The attempt is burned
  against a hard per-case cap, and the money is materially less likely to be
  recovered at all.

Optimising a threshold on F1 would weight these equally. They are not equal,
so the threshold is chosen on expected cost.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from recovery.diagnose.issuer_health import (
    HealthAssessment,
    IssuerHealthModel,
    ThresholdHealthModel,
)
from recovery.domain.observations import CaseFeatures
from recovery.world.oracle.outcomes import PotentialOutcomes

FALSE_NEGATIVE_COST = 10.0
"""Retrying into a degraded issuer: burns a capped attempt and forfeits most
of the recovery probability."""

FALSE_POSITIVE_COST = 1.0
"""Suppressing a retry that would have worked: a delay, not a loss."""


class DetectorScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    n: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    threshold: float | None = None

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def weighted_cost(self) -> float:
        """Expected cost per case, in units of a suppressed retry."""
        total = (
            self.false_negatives * FALSE_NEGATIVE_COST + self.false_positives * FALSE_POSITIVE_COST
        )
        return total / self.n if self.n else 0.0

    @property
    def positive_rate(self) -> float:
        return (self.true_positives + self.false_positives) / self.n if self.n else 0.0


def score_detector(
    name: str,
    assessments: Sequence[HealthAssessment],
    truth: Sequence[bool],
    threshold: float | None = None,
) -> DetectorScore:
    tp = fp = tn = fn = 0
    for assessment, actual in zip(assessments, truth, strict=True):
        predicted = assessment.is_degraded
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1
    return DetectorScore(
        name=name,
        n=len(truth),
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        threshold=threshold,
    )


def evaluate_health_models(
    features: Sequence[CaseFeatures],
    outcomes: Sequence[PotentialOutcomes],
    *,
    thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
) -> tuple[list[DetectorScore], DetectorScore]:
    """Sweep the Bayesian detector's threshold and compare against the rule.

    Returns (bayesian scores by threshold, threshold-rule score).
    """
    truth_by_id = {o.case_id: o.issuer_degraded_at_failure for o in outcomes}
    truth = [truth_by_id[f.case_id] for f in features]

    scores: list[DetectorScore] = []
    for threshold in thresholds:
        model = IssuerHealthModel(decision_threshold=threshold).fit(features)
        assessments = [model.assess(f) for f in features]
        scores.append(score_detector(f"bayesian@{threshold:.2f}", assessments, truth, threshold))

    naive = ThresholdHealthModel().fit(features)
    naive_score = score_detector("fixed_threshold_3pct", [naive.assess(f) for f in features], truth)
    return scores, naive_score


def volume_stratified_report(
    features: Sequence[CaseFeatures],
    outcomes: Sequence[PotentialOutcomes],
    model: IssuerHealthModel,
    naive: ThresholdHealthModel,
) -> dict[str, dict[str, float]]:
    """Break performance out by observation volume.

    This is where the two approaches diverge most sharply. The fixed rule is
    at its most confident exactly where the sample is smallest, so its
    false-positive rate should climb steeply as volume falls, while the
    Bayesian detector's posterior stays wide and it declines to fire.
    """
    truth_by_id = {o.case_id: o.issuer_degraded_at_failure for o in outcomes}
    volumes = np.array([f.issuer_volume_last_hour for f in features])
    edges = np.quantile(volumes, [0.0, 0.33, 0.66, 1.0])

    report: dict[str, dict[str, float]] = {}
    for i in range(3):
        lo, hi = edges[i], edges[i + 1]
        bucket = [f for f in features if lo <= f.issuer_volume_last_hour <= hi]
        if not bucket:
            continue
        truth = [truth_by_id[f.case_id] for f in bucket]
        bayes = score_detector("bayes", [model.assess(f) for f in bucket], truth)
        rule = score_detector("rule", [naive.assess(f) for f in bucket], truth)
        report[f"volume {int(lo)}-{int(hi)}"] = {
            "n": float(len(bucket)),
            "bayes_precision": bayes.precision,
            "bayes_fire_rate": bayes.positive_rate,
            "rule_precision": rule.precision,
            "rule_fire_rate": rule.positive_rate,
        }
    return report
