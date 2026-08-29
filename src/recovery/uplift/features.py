"""Feature encoding.

Turns `CaseFeatures` into a numeric matrix. Two properties matter more than
any modelling choice downstream:

**Nothing latent leaks in.** Every column is derived from fields a production
system reads off its own logs. The import contract already forbids reaching
into `recovery.world`, and `test_no_latent_features` asserts the column set
independently.

**The issuer-health signal is deliberately weak.** `issuer_failure_rate_last_hour`
is a small-sample ratio; at 20 observations it is nearly noise. That is the
honest representation of what a production system sees, and it is why the
Phase 4 detector's posterior is included as a separate, better-calibrated
column rather than leaving the model to rediscover it from the raw ratio.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from recovery.diagnose.issuer_health import IssuerHealthModel
from recovery.diagnose.taxonomy import Recoverability, classify
from recovery.domain.enums import CaseType
from recovery.domain.observations import CaseFeatures

RECOVERABILITY_ORDER: tuple[Recoverability, ...] = tuple(Recoverability)
CASE_TYPE_ORDER: tuple[CaseType, ...] = tuple(CaseType)

FEATURE_NAMES: tuple[str, ...] = (
    "log_amount",
    "tenure_days",
    "prior_payment_count",
    "prior_failure_count",
    "prior_recovery_count",
    "prior_failure_rate",
    "prior_recovery_rate",
    "contacts_last_30d",
    "dnd_registered",
    "hour_of_day",
    "day_of_month",
    "days_since_salary",
    "in_salary_window",
    "consecutive_mandate_failures",
    "issuer_failure_rate_last_hour",
    "issuer_volume_last_hour",
    "log_issuer_volume",
    "degradation_probability",
    *(f"recoverability_{r.value}" for r in RECOVERABILITY_ORDER),
    *(f"case_type_{c.value}" for c in CASE_TYPE_ORDER),
)


class FeatureEncoder:
    """Encodes cases for the uplift learners.

    Holds a fitted `IssuerHealthModel` so the Phase 4 posterior becomes a
    feature. That is deliberate reuse rather than duplication: the detector
    already solves the small-sample problem, and handing the model a
    well-calibrated probability is better than handing it a raw ratio and
    hoping a tree rediscovers the shrinkage.
    """

    def __init__(self, health_model: IssuerHealthModel) -> None:
        self.health_model = health_model

    @classmethod
    def fit(cls, features: Sequence[CaseFeatures]) -> FeatureEncoder:
        return cls(IssuerHealthModel().fit(features))

    def transform(self, features: Sequence[CaseFeatures]) -> np.ndarray:
        rows = [self._row(f) for f in features]
        return np.asarray(rows, dtype=np.float64)

    def _row(self, f: CaseFeatures) -> list[float]:
        recovery_rate = (
            f.prior_recovery_count / f.prior_failure_count if f.prior_failure_count else 0.0
        )
        assessment = self.health_model.assess(f)
        recoverability = classify(f.reason)

        row: list[float] = [
            float(np.log1p(f.amount_paise)),
            float(f.tenure_days),
            float(f.prior_payment_count),
            float(f.prior_failure_count),
            float(f.prior_recovery_count),
            f.prior_failure_rate,
            recovery_rate,
            float(f.contacts_last_30d),
            float(f.dnd_registered),
            float(f.hour_of_day),
            float(f.day_of_month),
            float(f.days_since_salary),
            float(f.days_since_salary < 4),
            float(f.consecutive_mandate_failures),
            f.issuer_failure_rate_last_hour,
            float(f.issuer_volume_last_hour),
            float(np.log1p(f.issuer_volume_last_hour)),
            assessment.degradation_probability,
        ]
        row.extend(float(recoverability is r) for r in RECOVERABILITY_ORDER)
        row.extend(float(f.case_type is c) for c in CASE_TYPE_ORDER)
        assert len(row) == len(FEATURE_NAMES)
        return row
