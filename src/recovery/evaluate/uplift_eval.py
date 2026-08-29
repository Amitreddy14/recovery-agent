"""Uplift evaluation harness.

May read the oracle — for the ceiling and for segment breakdowns. Nothing in
`recovery.uplift` may, and CI enforces that.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from recovery.domain.enums import ActionType
from recovery.domain.observations import CaseFeatures, LoggedDecision, RealizedOutcome
from recovery.uplift.features import FeatureEncoder
from recovery.uplift.targeting import TargetingResult, compare_targeting
from recovery.world.oracle.outcomes import PotentialOutcomes
from recovery.world.oracle.segments import Segment, classify

CONTACTING_TREATMENTS: frozenset[ActionType] = frozenset({ActionType.SEND_PAYMENT_LINK})
"""What counts as 'treated' for the headline comparison.

Contact is the scarce, costly, harm-capable action: it consumes a budget, it
is capped by compliance rules, and it is the only thing that can make a case
*worse*. Retries are cheap and silent, so pooling them with messaging would
dilute the effect we want to measure.

`PRE_DEBIT_NUDGE` is deliberately excluded despite also being a contact. It
is a no-op on already-failed one-off payments — a pre-debit notification
cannot precede a debit that has already happened — so on that population its
true uplift is exactly zero. Including it labelled a large block of untreated
units as treated, which is label noise no learner can recover from
(INC-013). The nudge is evaluated separately on the prevention population,
where it is a real intervention.
"""


def build_matrices(
    features: Sequence[CaseFeatures],
    logged: Sequence[LoggedDecision],
    realized: Sequence[RealizedOutcome],
    encoder: FeatureEncoder,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, treated, y, propensity, is_holdout) aligned by case."""
    logged_by_id = {d.case_id: d for d in logged}
    realized_by_id = {r.case_id: r for r in realized}

    usable = [
        f
        for f in features
        if logged_by_id[f.case_id].action in CONTACTING_TREATMENTS
        or logged_by_id[f.case_id].action is ActionType.NO_ACTION
    ]
    x = encoder.transform(usable)
    treated = np.array([logged_by_id[f.case_id].action in CONTACTING_TREATMENTS for f in usable])
    y = np.array([float(realized_by_id[f.case_id].recovered) for f in usable])
    propensity = np.array([logged_by_id[f.case_id].propensity for f in usable])
    holdout = np.array([logged_by_id[f.case_id].is_holdout for f in usable])
    return x, treated, y, propensity, holdout


def evaluate_uplift(
    features: Sequence[CaseFeatures],
    logged: Sequence[LoggedDecision],
    realized: Sequence[RealizedOutcome],
    outcomes: Sequence[PotentialOutcomes],
    *,
    seed: int = 0,
) -> tuple[list[TargetingResult], dict[str, int]]:
    """Fit on logged data, evaluate on the randomised holdout."""
    encoder = FeatureEncoder.fit(features)
    x, treated, y, propensity, holdout = build_matrices(features, logged, realized, encoder)

    logged_by_id = {d.case_id: d for d in logged}
    usable_ids = [
        f.case_id
        for f in features
        if logged_by_id[f.case_id].action in CONTACTING_TREATMENTS
        or logged_by_id[f.case_id].action is ActionType.NO_ACTION
    ]
    outcome_by_id = {o.case_id: o for o in outcomes}
    true_uplift = np.array(
        [max(outcome_by_id[cid].uplift(a) for a in CONTACTING_TREATMENTS) for cid in usable_ids]
    )

    # Split the randomised holdout. The logging policy almost never chooses
    # NO_ACTION, so training on logged data alone starves the control arm
    # (~800 rows against ~4400 treated) — and mu0 is precisely the term uplift
    # depends on. Half the holdout joins training to supply balanced control
    # data; the other half stays untouched for unbiased evaluation (INC-014).
    rng = np.random.default_rng(seed)
    holdout_idx = np.flatnonzero(holdout)
    rng.shuffle(holdout_idx)
    split = len(holdout_idx) // 2
    to_train, to_eval = holdout_idx[:split], holdout_idx[split:]

    train = ~holdout
    train[to_train] = True
    eval_mask = np.zeros_like(holdout)
    eval_mask[to_eval] = True

    results = compare_targeting(
        x_train=x[train],
        treated_train=treated[train],
        y_train=y[train],
        propensity_train=propensity[train],
        x_eval=x[eval_mask],
        treated_eval=treated[eval_mask],
        y_eval=y[eval_mask],
        propensity_eval=propensity[eval_mask],
        true_uplift_eval=true_uplift[eval_mask],
        seed=seed,
    )

    counts = {
        "n_usable": len(usable_ids),
        "n_train": int(train.sum()),
        "n_train_control": int((~treated[train]).sum()),
        "n_eval": int(eval_mask.sum()),
        "n_treated": int(treated.sum()),
    }
    return results, counts


def segment_capture(
    features: Sequence[CaseFeatures],
    logged: Sequence[LoggedDecision],
    realized: Sequence[RealizedOutcome],
    outcomes: Sequence[PotentialOutcomes],
    *,
    top_fraction: float = 0.30,
    seed: int = 0,
) -> dict[str, Counter[Segment]]:
    """Which segments each ranking actually selects.

    The Qini coefficient says one ranking is better. This says *why*, and it
    is the more persuasive artefact: the risk ranking should fill its budget
    with lost causes and sleeping dogs, while the uplift ranking should
    concentrate persuadables.
    """
    encoder = FeatureEncoder.fit(features)
    x, treated, y, propensity, holdout = build_matrices(features, logged, realized, encoder)

    logged_by_id = {d.case_id: d for d in logged}
    usable_ids = [
        f.case_id
        for f in features
        if logged_by_id[f.case_id].action in CONTACTING_TREATMENTS
        or logged_by_id[f.case_id].action is ActionType.NO_ACTION
    ]
    outcome_by_id = {o.case_id: o for o in outcomes}

    from recovery.uplift.learners import XLearner
    from recovery.uplift.targeting import rank_by_risk, split_arms

    rng = np.random.default_rng(seed)
    holdout_idx = np.flatnonzero(holdout)
    rng.shuffle(holdout_idx)
    split = len(holdout_idx) // 2
    train = ~holdout
    train[holdout_idx[:split]] = True
    eval_mask = np.zeros_like(holdout)
    eval_mask[holdout_idx[split:]] = True

    control, treated_arm = split_arms(x[train], treated[train], y[train], propensity[train])
    model = XLearner(seed=seed).fit(control, treated_arm)

    eval_ids = [cid for cid, h in zip(usable_ids, eval_mask, strict=True) if h]
    x_eval = x[eval_mask]
    rankings = {
        "uplift": model.predict_uplift(x_eval, propensity[eval_mask]),
        "risk": rank_by_risk(model, x_eval),
    }

    cutoff = max(1, int(len(eval_ids) * top_fraction))
    report: dict[str, Counter[Segment]] = {}
    for name, scores in rankings.items():
        order = np.argsort(-scores, kind="stable")[:cutoff]
        report[name] = Counter(classify(outcome_by_id[eval_ids[i]]) for i in order)
    return report
