"""The diagnosis engine.

Produces a `Diagnosis` per case: root cause, whether it can be recovered
without spending a customer contact, whether the issuer looks degraded, and
the evidence behind each conclusion.

The evidence tuple is not decoration. It is what makes an action explainable
in the audit trail, and it is written even when the diagnosis is uncertain —
"unmapped reason, routed to review" is a legitimate and useful record.
"""

from __future__ import annotations

from collections.abc import Sequence

from recovery.diagnose.issuer_health import HealthAssessment, IssuerHealthModel
from recovery.diagnose.taxonomy import (
    Recoverability,
    classify,
    expected_decline_class,
    is_silently_recoverable,
    is_terminal,
)
from recovery.domain.actions import Diagnosis
from recovery.domain.enums import CaseType, FailureReason
from recovery.domain.observations import CaseFeatures

CONFIDENCE_BY_CLASS: dict[Recoverability, float] = {
    Recoverability.TERMINAL: 0.99,
    Recoverability.INSTRUMENT: 0.95,
    Recoverability.TRANSIENT: 0.90,
    Recoverability.FUNDING: 0.88,
    Recoverability.AUTHENTICATION: 0.80,
    Recoverability.INTENT: 0.75,
    Recoverability.UNKNOWN: 0.30,
}
"""Confidence in the *root-cause attribution*, not in recovery.

Intent sits lower than it might: a cancellation is strong evidence the
customer chose not to pay, but it is also what a confused checkout looks
like, and the two are indistinguishable from the error code alone.
"""


class DiagnosisEngine:
    def __init__(self, health_model: IssuerHealthModel) -> None:
        self.health_model = health_model

    @classmethod
    def fit(cls, features: Sequence[CaseFeatures]) -> DiagnosisEngine:
        return cls(IssuerHealthModel().fit(features))

    def diagnose(self, features: CaseFeatures) -> Diagnosis:
        recoverability = classify(features.reason)
        evidence: list[str] = []

        # --- Pre-failure cases are a different question entirely ----------
        if features.case_type is CaseType.UPCOMING_AT_RISK:
            evidence.append("mandate due; no failure has occurred")
            if features.consecutive_mandate_failures:
                evidence.append(
                    f"{features.consecutive_mandate_failures} consecutive prior failures"
                )
            if features.days_since_salary >= 20:
                evidence.append(
                    f"{features.days_since_salary}d since salary; balance risk elevated"
                )
            return Diagnosis(
                case_id=features.case_id,
                root_cause="upcoming_debit",
                confidence=0.85,
                recoverable_without_contact=True,
                issuer_degraded=False,
                evidence=tuple(evidence),
            )

        evidence.append(f"reason={features.reason.value} -> {recoverability.value}")

        # --- Consistency check against the recorded decline class ---------
        expected = expected_decline_class(features.reason)
        if expected is not None and expected is not features.decline_class:
            # Do not diagnose confidently from a self-contradictory record.
            evidence.append(
                f"INCONSISTENT: {features.reason.value} recorded as "
                f"{features.decline_class.value}, expected {expected.value}"
            )
            return Diagnosis(
                case_id=features.case_id,
                root_cause="inconsistent_record",
                confidence=0.20,
                recoverable_without_contact=False,
                issuer_degraded=False,
                evidence=tuple(evidence),
            )

        # --- Issuer health, inferred ---------------------------------------
        assessment: HealthAssessment | None = None
        if recoverability is Recoverability.TRANSIENT:
            assessment = self.health_model.assess(features)
            evidence.append(assessment.evidence)
            if assessment.is_degraded:
                evidence.append("issuer degraded; immediate retry likely to fail")

        # --- Corroborating signals ----------------------------------------
        if recoverability is Recoverability.FUNDING:
            if features.days_since_salary < 4:
                evidence.append(
                    f"only {features.days_since_salary}d since salary; "
                    "shortfall unusual, may indicate a limit rather than balance"
                )
            else:
                evidence.append(
                    f"{features.days_since_salary}d since salary; "
                    "balance likely to recover next cycle"
                )

        if recoverability is Recoverability.INTENT and features.prior_failure_rate > 0.5:
            evidence.append(
                f"prior failure rate {features.prior_failure_rate:.0%}; "
                "repeated non-completion supports low intent"
            )

        root_cause = self._root_cause(features.reason, recoverability, assessment)
        confidence = CONFIDENCE_BY_CLASS[recoverability]
        if assessment is not None and not assessment.is_degraded:
            # Ruling degradation out is itself informative.
            confidence = min(confidence + 0.04, 0.99)

        return Diagnosis(
            case_id=features.case_id,
            root_cause=root_cause,
            confidence=confidence,
            recoverable_without_contact=(
                is_silently_recoverable(features.reason) and not is_terminal(features.reason)
            ),
            issuer_degraded=bool(assessment and assessment.is_degraded),
            evidence=tuple(evidence),
        )

    @staticmethod
    def _root_cause(
        reason: FailureReason,
        recoverability: Recoverability,
        assessment: HealthAssessment | None,
    ) -> str:
        if assessment is not None and assessment.is_degraded:
            return "issuer_degradation"
        if recoverability is Recoverability.UNKNOWN:
            return "unmapped_reason"
        return recoverability.value
