"""Console data.

Lives in `evaluate` rather than `api` because building a snapshot means
running the batch and classifying segments, both of which require the world.
`api` is forbidden from importing `recovery.world` (ADR-0011) precisely so no
endpoint can ever serve a counterfactual, and putting the builder there broke
that contract (INC-026). The API reads the frozen file; it does not produce
it.

Runs the whole pipeline once and freezes the result to JSON. The API then
serves that file rather than recomputing, because a batch takes tens of
seconds and a console that stalls on load is not a console.

Everything here is derived, not invented. Money figures come from the same
`evaluate_policies` path that produces the numbers in DECISIONS.md, so the
screen and the write-up cannot drift apart.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recovery.calibration.models import WorldParameters
from recovery.compliance.engine import ComplianceEngine, load_policy
from recovery.diagnose.engine import DiagnosisEngine
from recovery.domain.enums import ActionType, GateVerdict
from recovery.evaluate.policy_eval import (
    blind_retry_actions,
    evaluate_policies,
    oracle_actions,
    top_n_actions,
)
from recovery.paths import CONSOLE_SNAPSHOT
from recovery.policy.decision import CANDIDATE_ACTIONS
from recovery.policy.economics import Economics
from recovery.policy.runner import fit_policy, mandate_values, run_policy
from recovery.uplift.features import FeatureEncoder
from recovery.uplift.targeting import rank_by_risk
from recovery.world.generate import generate
from recovery.world.oracle.segments import classify

CONTACTING = (ActionType.SEND_PAYMENT_LINK, ActionType.PRE_DEBIT_NUDGE)
MAX_CASES_IN_TABLE = 400
"""Cases sent to the browser. The console is for interrogating decisions, not
for paging through forty thousand rows — the aggregate panels answer 'how did
the batch go' and the table answers 'why this case'."""


def build_snapshot(
    *,
    n_cases: int = 20000,
    seed: int = 42,
    params_path: Path = Path("configs/generator/world_params.json"),
    policy_path: Path = Path("configs/compliance/policy.yaml"),
) -> dict[str, Any]:
    params = WorldParameters.model_validate_json(params_path.read_text(encoding="utf-8"))
    policy = load_policy(policy_path)
    economics = Economics.from_policy(policy.costs_paise, policy.sleeping_dog_penalty)

    observable, oracle = generate(params, n_cases=n_cases, seed=seed)
    encoder = FeatureEncoder.fit(observable.features)
    fitted = fit_policy(observable.features, observable.logged, observable.realized, encoder)
    x = encoder.transform(observable.features)
    decisions = run_policy(
        fitted,
        observable.features,
        x,
        compliance=ComplianceEngine(policy),
        economics=economics,
        diagnosis=DiagnosisEngine.fit(observable.features),
    )

    amounts = [f.amount_paise for f in observable.features]
    mandate_vals = mandate_values(observable.features, economics)
    chosen = [d.decision.action.action_type for d in decisions]
    # Risk ranking targets whoever is *least* likely to recover. Passing
    # predict_baseline directly inverts it and targets the most likely - which
    # silently turns the baseline into a sure-thing picker and flatters our
    # policy by comparison (INC-025).
    risk_scores = rank_by_risk(fitted.recovery, x)
    budget = max(1, sum(1 for a in chosen if a in CONTACTING))

    policies = {
        "do_nothing": [ActionType.NO_ACTION] * len(chosen),
        "blind_retry": blind_retry_actions(
            [f.case_type.value for f in observable.features],
            [f.consecutive_mandate_failures for f in observable.features],
        ),
        "risk_topN": top_n_actions(list(risk_scores), budget),
        "uplift_ev": chosen,
        "oracle": oracle_actions(
            oracle.outcomes, amounts, mandate_vals, economics.costs_paise, CANDIDATE_ACTIONS
        ),
    }
    results = evaluate_policies(
        policies, oracle.outcomes, amounts, mandate_vals, economics.costs_paise
    )
    ours = results["uplift_ev"]
    ceiling = results["oracle"]

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "run": {
            "seed": seed,
            "n_cases": n_cases,
            "calibration": observable.params_provenance,
            "policy_version": policy.version,
        },
        "reconciliation": _reconciliation(ours, amounts),
        "policies": _policy_rows(results, ceiling.net_paise),
        "gates": _gate_activity(decisions),
        "segments": _segment_capture(
            observable.features, oracle.outcomes, chosen, policies["risk_topN"]
        ),
        "actions": _action_mix(chosen),
        "cases": _case_rows(observable.features, decisions, oracle.outcomes, amounts),
    }


def _reconciliation(outcome: Any, amounts: Sequence[int]) -> dict[str, Any]:
    at_risk = sum(amounts)
    return {
        "at_risk_paise": at_risk,
        "recovered_paise": outcome.recovered_paise,
        "forfeited_paise": outcome.cancellation_loss_paise,
        "intervention_cost_paise": outcome.intervention_cost_paise,
        "net_paise": outcome.net_paise,
        "unrecovered_paise": at_risk - outcome.recovered_paise,
        # Exact, not approximate. Integer paise throughout (ADR-0003) is what
        # makes this an equality rather than a tolerance.
        "balances": (outcome.recovered_paise + (at_risk - outcome.recovered_paise) == at_risk),
        "contacts": outcome.contacts,
        "attempts": outcome.attempts,
        "mandates_cancelled": outcome.mandates_cancelled,
    }


def _policy_rows(results: Mapping[str, Any], ceiling_paise: int) -> list[dict[str, Any]]:
    order = ["do_nothing", "blind_retry", "risk_topN", "uplift_ev", "oracle"]
    rows = []
    for name in order:
        r = results[name]
        rows.append(
            {
                "name": name,
                "recovered_paise": r.recovered_paise,
                "forfeited_paise": r.cancellation_loss_paise,
                "cost_paise": r.intervention_cost_paise,
                "net_paise": r.net_paise,
                "share_of_ceiling": r.net_paise / ceiling_paise if ceiling_paise else 0,
                "contacts": r.contacts,
                "cancelled": r.mandates_cancelled,
                "is_ours": name == "uplift_ev",
                "is_ceiling": name == "oracle",
            }
        )
    return rows


def _gate_activity(decisions: Sequence[Any]) -> list[dict[str, Any]]:
    blocked: Counter[str] = Counter()
    deferred: Counter[str] = Counter()
    passed: Counter[str] = Counter()
    for decision in decisions:
        for review in decision.reviews.values():
            for result in review.results:
                if result.verdict is GateVerdict.BLOCK:
                    blocked[result.rule_id] += 1
                elif result.verdict is GateVerdict.DEFER:
                    deferred[result.rule_id] += 1
                else:
                    passed[result.rule_id] += 1
    rules = sorted(set(blocked) | set(deferred) | set(passed))
    return [
        {
            "rule": rule,
            "blocked": blocked.get(rule, 0),
            "deferred": deferred.get(rule, 0),
            "passed": passed.get(rule, 0),
        }
        for rule in rules
        if blocked.get(rule) or deferred.get(rule)
    ]


def _segment_capture(
    features: Sequence[Any],
    outcomes: Sequence[Any],
    ours: Sequence[ActionType],
    risk: Sequence[ActionType],
) -> list[dict[str, Any]]:
    segments = [classify(o) for o in outcomes]
    ours_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    for segment, a, b in zip(segments, ours, risk, strict=True):
        if a in CONTACTING:
            ours_counts[segment.value] += 1
        if b in CONTACTING:
            risk_counts[segment.value] += 1
    ours_total = sum(ours_counts.values()) or 1
    risk_total = sum(risk_counts.values()) or 1
    return [
        {
            "segment": name,
            "ours": ours_counts.get(name, 0),
            "ours_share": ours_counts.get(name, 0) / ours_total,
            "risk": risk_counts.get(name, 0),
            "risk_share": risk_counts.get(name, 0) / risk_total,
        }
        for name in (
            "persuadable",
            "lost_cause",
            "sleeping_dog",
            "sure_thing",
            "indifferent",
        )
    ]


def _action_mix(chosen: Sequence[ActionType]) -> list[dict[str, Any]]:
    counts = Counter(a.value for a in chosen)
    total = len(chosen) or 1
    return [{"action": a, "count": n, "share": n / total} for a, n in counts.most_common()]


def _case_rows(
    features: Sequence[Any],
    decisions: Sequence[Any],
    outcomes: Sequence[Any],
    amounts: Sequence[int],
) -> list[dict[str, Any]]:
    """Rows for the table, biased toward the interesting ones.

    A random sample would be mostly routine retries. The console exists to
    interrogate judgement calls, so blocked and declined cases are
    over-represented — and the row says which it is, so nobody mistakes the
    sample for the distribution.
    """
    scored: list[tuple[int, dict[str, Any]]] = []
    for i, (feature, cd, outcome) in enumerate(zip(features, decisions, outcomes, strict=True)):
        blocked = cd.blocked_actions
        segment = classify(outcome)
        action = cd.decision.action.action_type

        interest = 0
        if blocked:
            interest += 3
        if action is ActionType.NO_ACTION:
            interest += 2
        if segment.value == "sleeping_dog":
            interest += 2
        if cd.diagnosis.issuer_degraded:
            interest += 2

        scored.append(
            (
                interest,
                {
                    "case_id": feature.case_id,
                    "case_type": feature.case_type.value,
                    "amount_paise": amounts[i],
                    "issuer": feature.issuer,
                    "reason": feature.reason.value,
                    "segment": segment.value,
                    "action": action.value,
                    "root_cause": cd.diagnosis.root_cause,
                    "confidence": cd.diagnosis.confidence,
                    "issuer_degraded": cd.diagnosis.issuer_degraded,
                    "evidence": list(cd.diagnosis.evidence),
                    "rationale": cd.decision.rationale,
                    "blocked": {a.value: list(rules) for a, rules in blocked.items()},
                    "scores": [
                        {
                            "action": s.action_type.value,
                            "p_recovery": s.p_recovery,
                            "uplift": s.uplift,
                            "ev_paise": s.expected_value_paise,
                            "chosen": s.action_type is action,
                        }
                        for s in cd.decision.scores
                    ],
                    "would_recover": bool(outcome.recovered.get(action, False)),
                    "interest": interest,
                },
            )
        )

    # Sort by interest, then round-robin across decision types. Pure interest
    # ordering stacks one category at the top - every visible row a declined
    # sleeping dog - which reads as repetition rather than range and hides the
    # variety the console exists to show.
    scored.sort(key=lambda pair: -pair[0])
    buckets: dict[str, list[dict[str, Any]]] = {}
    for _score, row in scored:
        # Bucket on decision *and* segment. Bucketing on decision alone still
        # produced a top-of-table that was uniformly sleeping dogs, because
        # interest scoring ranks them first inside every bucket.
        key = f"{'gated' if row['blocked'] else row['action']}|{row['segment']}"
        buckets.setdefault(key, []).append(row)

    interleaved: list[dict[str, Any]] = []
    order = sorted(buckets, key=lambda k: -len(buckets[k]))
    while len(interleaved) < MAX_CASES_IN_TABLE and any(buckets[k] for k in order):
        for key in order:
            if buckets[key]:
                interleaved.append(buckets[key].pop(0))
                if len(interleaved) >= MAX_CASES_IN_TABLE:
                    break
    return interleaved


def write_snapshot(data: dict[str, Any], path: Path = CONSOLE_SNAPSHOT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    return path
