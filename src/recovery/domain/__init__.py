"""Domain layer: the contract every other module is written against.

This package must not import from any other `recovery` module. That is
enforced by an import-linter contract in CI.
"""

from recovery.domain.actions import (
    Action,
    ActionScore,
    ComplianceReview,
    Decision,
    Diagnosis,
    ExecutionRecord,
    GateResult,
    Outcome,
)
from recovery.domain.entities import (
    Customer,
    Mandate,
    Merchant,
    Paise,
    PaymentAttempt,
    PaymentError,
    RecoveryCase,
)
from recovery.domain.enums import (
    ActionType,
    CaseState,
    CaseType,
    Channel,
    DeclineClass,
    ErrorSource,
    FailureReason,
    GateVerdict,
    MandateCategory,
    PaymentMethod,
    PaymentStep,
)

__all__ = [
    "Action",
    "ActionScore",
    "ActionType",
    "CaseState",
    "CaseType",
    "Channel",
    "ComplianceReview",
    "Customer",
    "Decision",
    "DeclineClass",
    "Diagnosis",
    "ErrorSource",
    "ExecutionRecord",
    "FailureReason",
    "GateResult",
    "GateVerdict",
    "Mandate",
    "MandateCategory",
    "Merchant",
    "Outcome",
    "Paise",
    "PaymentAttempt",
    "PaymentError",
    "PaymentMethod",
    "PaymentStep",
    "RecoveryCase",
]
