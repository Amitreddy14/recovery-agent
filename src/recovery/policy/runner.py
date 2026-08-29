"""End-to-end policy runner.

Fits two uplift models on observable data, prices every action, gates it, and
picks the best survivor per case.

The second model is the point of this phase. Phase 5's model predicts the
effect of contact on *recovery*. This one predicts the effect of contact on
*mandate cancellation* — trained on `RealizedOutcome.mandate_cancelled`, which
a production system observes directly. INC-017 recorded that ranking alone
selects more sleeping dogs than risk ranking does; the fix is not a better
ranking but a second objective, converted to rupees and subtracted.

Both models see only observable data. Neither can import `recovery.world`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from lightgbm import LGBMClassifier

from recovery.compliance.engine import CaseContext, ComplianceEngine
from recovery.diagnose.engine import DiagnosisEngine
from recovery.domain.actions import ActionScore
from recovery.domain.entities import Mandate, RecoveryCase
from recovery.domain.enums import ActionType, CaseState, CaseType, MandateCategory
from recovery.domain.observations import CaseFeatures, LoggedDecision, RealizedOutcome
from recovery.policy.decision import CANDIDATE_ACTIONS, CaseDecision, DecisionEngine
from recovery.policy.economics import Economics, score_action
from recovery.uplift.features import FeatureEncoder
from recovery.uplift.learners import TreatmentData, XLearner

CONTACTING = (ActionType.SEND_PAYMENT_LINK, ActionType.PRE_DEBIT_NUDGE)


@dataclass(frozen=True)
class FittedPolicy:
    """Two models, one for each thing contact can do to a customer."""

    recovery: XLearner
    cancellation: LGBMClassifier | None
    encoder: FeatureEncoder

    def score_all(
        self, x: np.ndarray, features: Sequence[CaseFeatures], economics: Economics
    ) -> list[list[ActionScore]]:
        base = np.clip(self.recovery.predict_baseline(x), 0.0, 1.0)
        lift = self.recovery.predict_uplift(x)
        if self.cancellation is not None:
            proba = np.asarray(self.cancellation.predict_proba(x), dtype=np.float64)
            cancel_lift = np.clip(proba[:, 1], 0.0, 1.0)
        else:
            cancel_lift = np.zeros(len(x))

        rows: list[list[ActionScore]] = []
        for i, feature in enumerate(features):
            mandate = _mandate_for(feature)
            scores: list[ActionScore] = []
            for action in CANDIDATE_ACTIONS:
                p, u, pc = _action_estimates(
                    action, float(base[i]), float(lift[i]), float(cancel_lift[i])
                )
                scores.append(
                    score_action(
                        action=action,
                        p_recovery=p,
                        uplift=u,
                        p_cancel=pc,
                        amount_paise=feature.amount_paise,
                        mandate=mandate,
                        economics=economics,
                    )
                )
            rows.append(scores)
        return rows


def _action_estimates(
    action: ActionType, base: float, lift: float, cancel_lift: float
) -> tuple[float, float, float]:
    """Map the two fitted effects onto the action set.

    The models are fitted on the contact decision, which is the arm with a
    genuine control group in the logged data. Silent actions are given a
    fraction of the contact effect and no cancellation risk — a retry does not
    annoy anyone, which is precisely why well-timed retries are the cheapest
    money in the system.
    """
    if action is ActionType.NO_ACTION:
        return base, 0.0, 0.0
    if action in CONTACTING:
        damping = 0.35 if action is ActionType.PRE_DEBIT_NUDGE else 1.0
        return base + lift, lift, cancel_lift * damping
    # Silent actions: retries.
    share = {
        ActionType.RETRY_NOW: 0.45,
        ActionType.RETRY_SCHEDULED: 0.70,
        ActionType.RETRY_ALTERNATE_RAIL: 0.40,
    }.get(action, 0.0)
    return base + lift * share, lift * share, 0.0


def _mandate_for(feature: CaseFeatures) -> Mandate | None:
    if feature.mandate_category is None:
        return None
    from datetime import timedelta

    return Mandate(
        mandate_id=f"mdt_{feature.case_id}",
        customer_id=feature.customer_id,
        merchant_id="mrc_1",
        category=feature.mandate_category,
        max_amount_paise=feature.amount_paise * 2,
        debit_amount_paise=feature.amount_paise,
        registered_at=feature.created_at - timedelta(days=90),
        valid_until=feature.created_at + timedelta(days=365),
        next_debit_at=feature.created_at + timedelta(days=1),
        consecutive_failures=feature.consecutive_mandate_failures,
    )


def _case_for(feature: CaseFeatures) -> RecoveryCase:
    return RecoveryCase(
        case_id=feature.case_id,
        case_type=feature.case_type,
        merchant_id="mrc_1",
        customer_id=feature.customer_id,
        amount_at_risk_paise=feature.amount_paise,
        created_at=feature.created_at,
        state=CaseState.SCORED,
        contacts_used=0,
    )


def fit_policy(
    features: Sequence[CaseFeatures],
    logged: Sequence[LoggedDecision],
    realized: Sequence[RealizedOutcome],
    encoder: FeatureEncoder,
) -> FittedPolicy:
    """Fit the recovery and cancellation models on the contact arm.

    Restricted to cases whose logged action was either a contact or true
    inaction. Those are the only rows with a defined control group, and
    training on the rest would compare a contacted customer against a
    retried one — a different question entirely.
    """
    logged_by_id = {d.case_id: d for d in logged}
    realized_by_id = {r.case_id: r for r in realized}
    usable = [
        f
        for f in features
        if logged_by_id[f.case_id].action in CONTACTING
        or logged_by_id[f.case_id].action is ActionType.NO_ACTION
    ]
    x = encoder.transform(usable)
    treated_mask = np.array([logged_by_id[f.case_id].action in CONTACTING for f in usable])
    control_mask = ~treated_mask
    recovered = np.array([float(realized_by_id[f.case_id].recovered) for f in usable])
    cancelled = np.array([float(realized_by_id[f.case_id].mandate_cancelled) for f in usable])
    # Logged propensities are carried into training so the learners can
    # correct for the naive policy's selection bias (ADR-0005).
    propensity = np.array([logged_by_id[f.case_id].propensity for f in usable])

    recovery = XLearner().fit(
        TreatmentData(
            x=x[control_mask], y=recovered[control_mask], propensity=propensity[control_mask]
        ),
        TreatmentData(
            x=x[treated_mask], y=recovered[treated_mask], propensity=propensity[treated_mask]
        ),
    )

    # Cancellation uses a plain classifier on the treated arm, NOT an uplift
    # learner. A customer who is never contacted never cancels *because of*
    # contact, so the control arm carries structurally zero events (INC-019)
    # and there is no treatment effect to difference out. Where the control
    # outcome is identically zero, P(cancel | contacted, x) *is* the uplift.
    cancellation: LGBMClassifier | None = None
    if cancelled[treated_mask].sum() >= 10:
        cancellation = LGBMClassifier(
            n_estimators=200,
            num_leaves=15,
            learning_rate=0.05,
            min_child_samples=30,
            verbose=-1,
            random_state=0,
        )
        cancellation.fit(x[treated_mask], cancelled[treated_mask])
    return FittedPolicy(recovery=recovery, cancellation=cancellation, encoder=encoder)


def run_policy(
    policy: FittedPolicy,
    features: Sequence[CaseFeatures],
    x: np.ndarray,
    *,
    compliance: ComplianceEngine,
    economics: Economics,
    diagnosis: DiagnosisEngine,
) -> list[CaseDecision]:
    """Score, gate and decide for every case."""
    engine = DecisionEngine(compliance, economics)
    all_scores = policy.score_all(x, features, economics)

    decisions: list[CaseDecision] = []
    for feature, scores in zip(features, all_scores, strict=True):
        mandate = _mandate_for(feature)
        context = CaseContext(
            case=_case_for(feature),
            mandate=mandate,
            now=feature.created_at,
            contacts_last_30d=feature.contacts_last_30d,
            dnd_registered=feature.dnd_registered,
            pre_debit_notice_sent_at=(
                feature.created_at if feature.case_type is CaseType.UPCOMING_AT_RISK else None
            ),
        )
        decisions.append(
            engine.decide(
                scores=scores,
                context=context,
                diagnosis=diagnosis.diagnose(feature),
                mandate=mandate,
            )
        )
    return decisions


def mandate_values(features: Sequence[CaseFeatures], economics: Economics) -> list[int]:
    return [
        f.amount_paise * economics.horizon_cycles
        if f.mandate_category is not None and f.mandate_category is not MandateCategory.OTHER
        else 0
        for f in features
    ]
