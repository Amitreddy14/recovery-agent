"""Batch generation.

`generate()` returns observable data and oracle data as two separate objects,
written to two separate files. Policy code receives only the first. The
split is physical, not conventional.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from recovery.calibration import assumptions
from recovery.calibration.models import WorldParameters
from recovery.domain.enums import (
    CaseType,
    DeclineClass,
    FailureReason,
    MandateCategory,
    PaymentMethod,
)
from recovery.domain.observations import RealizedOutcome
from recovery.world.cases import CaseFeatures, LoggedDecision, log_action
from recovery.world.latent import observable_history, sample_latent
from recovery.world.oracle.outcomes import (
    PotentialOutcomes,
    compute_potential_outcomes,
    compute_upcoming_outcomes,
)
from recovery.world.timeline import IssuerTimeline

UPCOMING_FRACTION = 0.30
"""Share of cases that are mandates due to debit but not yet failed.

Without this population there are no sure things: once a payment has failed,
some action always beats doing nothing, so every case is persuadable or lost.
The prevention population is also what makes the pre-debit notification a
recovery channel rather than a compliance checkbox."""

MANDATE_FRACTION = 0.42
"""Share of cases that are recurring-mandate failures rather than one-off."""

ONE_OFF_METHODS: tuple[PaymentMethod, ...] = (
    PaymentMethod.UPI,
    PaymentMethod.CARD,
    PaymentMethod.NETBANKING,
)

MANDATE_CATEGORY_WEIGHTS: dict[MandateCategory, float] = {
    MandateCategory.SUBSCRIPTION: 0.38,
    MandateCategory.UTILITY: 0.18,
    MandateCategory.LOAN_EMI: 0.16,
    MandateCategory.INSURANCE_PREMIUM: 0.12,
    MandateCategory.MUTUAL_FUND: 0.10,
    MandateCategory.CREDIT_CARD_BILL: 0.06,
}


@dataclass(frozen=True)
class ObservableBatch:
    """What the policy is allowed to see."""

    features: tuple[CaseFeatures, ...]
    logged: tuple[LoggedDecision, ...]
    realized: tuple[RealizedOutcome, ...]
    seed: int
    params_provenance: str


@dataclass(frozen=True)
class OracleBatch:
    """QUARANTINED counterfactuals. Never handed to policy code."""

    outcomes: tuple[PotentialOutcomes, ...]
    seed: int


def _pick_issuer(params: WorldParameters, rng: np.random.Generator) -> str:
    names = [p.bank_name for p in params.issuers]
    weights = np.array([p.volume_share for p in params.issuers])
    return str(rng.choice(names, p=weights / weights.sum()))


def _pick_reason(
    params: WorldParameters,
    is_technical: bool,
    is_mandate: bool,
    rng: np.random.Generator,
) -> FailureReason:
    mix = params.td_reason_mix if is_technical else params.bd_reason_mix
    reasons = list(mix)
    weights = np.array([mix[r] for r in reasons], dtype=float)

    if is_mandate and not is_technical:
        # Recurring debits skew toward balance problems: the customer is not
        # present to correct anything at the moment of the debit.
        for i, reason in enumerate(reasons):
            if reason == FailureReason.INSUFFICIENT_FUNDS:
                weights[i] *= params.mandate_bd_uplift_factor
            if reason == FailureReason.PAYMENT_CANCELLED_BY_USER:
                weights[i] *= 0.25

    return FailureReason(rng.choice(reasons, p=weights / weights.sum()))


def generate(
    params: WorldParameters,
    *,
    n_cases: int,
    seed: int,
    window_days: int = 30,
    start: datetime | None = None,
) -> tuple[ObservableBatch, OracleBatch]:
    """Generate a batch. Deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    start = start or datetime(2026, 7, 1, tzinfo=UTC)

    timeline = IssuerTimeline(
        profiles=params.issuers,
        start=start,
        days=window_days,
        duration_hours_range=assumptions.DEGRADATION_DURATION_HOURS_RANGE,
        rng=rng,
    )

    features: list[CaseFeatures] = []
    logged: list[LoggedDecision] = []
    realized: list[RealizedOutcome] = []
    outcomes: list[PotentialOutcomes] = []

    for i in range(n_cases):
        case_id = f"case_{seed}_{i:06d}"
        latent = sample_latent(rng)

        issuer = _pick_issuer(params, rng)
        failed_at = start + timedelta(hours=float(rng.uniform(0, window_days * 24)))
        is_upcoming = bool(rng.random() < UPCOMING_FRACTION)
        is_mandate = is_upcoming or bool(rng.random() < MANDATE_FRACTION)

        # Technical vs business decline, conditioned on this issuer's state
        # right now. A degraded issuer produces proportionally more TD, which
        # is the signal the policy must learn to read.
        td_now = timeline.td_rate_at(issuer, failed_at)
        profile = next(p for p in params.issuers if p.bank_name == issuer)
        bd_now = profile.published_bd_rate
        if is_mandate:
            bd_now *= params.mandate_bd_uplift_factor

        days_since_salary = (failed_at.day - assumptions.SALARY_DAY_OF_MONTH) % 30
        if days_since_salary < assumptions.SALARY_WINDOW_DAYS:
            bd_now *= params.salary_window_bd_multiplier

        if is_upcoming:
            # No failure has occurred yet, so there is no decline to classify.
            is_technical = False
            reason = FailureReason.OTHER
        else:
            is_technical = bool(rng.random() < td_now / max(td_now + bd_now, 1e-9))
            reason = _pick_reason(params, is_technical, is_mandate, rng)

        tenure_days = int(rng.integers(1, 1500))
        payments, failures, recoveries = observable_history(latent, tenure_days, rng)

        amount_paise = int(
            rng.choice([49900, 99900, 199900, 499900, 1299900, 4999900])
            * float(rng.uniform(0.85, 1.15))
        )

        category: MandateCategory | None = None
        consecutive = 0
        if is_mandate:
            cats = list(MANDATE_CATEGORY_WEIGHTS)
            w = np.array([MANDATE_CATEGORY_WEIGHTS[c] for c in cats])
            category = MandateCategory(rng.choice(cats, p=w / w.sum()))
            consecutive = int(rng.integers(0, 3))

        # Observable issuer-health proxy: noisy small-sample estimate.
        volume_last_hour = int(rng.integers(20, 400))
        expected = td_now * volume_last_hour
        issuer_failures = int(rng.poisson(max(expected, 0.01)))

        feature = CaseFeatures(
            case_id=case_id,
            case_type=(
                CaseType.UPCOMING_AT_RISK
                if is_upcoming
                else CaseType.MANDATE_FAILURE
                if is_mandate
                else CaseType.PAYMENT_FAILURE
            ),
            created_at=failed_at,
            amount_paise=max(amount_paise, 100),
            method=(
                PaymentMethod.EMANDATE
                if is_mandate
                else ONE_OFF_METHODS[int(rng.integers(len(ONE_OFF_METHODS)))]
            ),
            issuer=issuer,
            reason=reason,
            decline_class=(DeclineClass.TECHNICAL if is_technical else DeclineClass.BUSINESS),
            customer_id=f"cust_{seed}_{i:06d}",
            tenure_days=tenure_days,
            prior_payment_count=payments,
            prior_failure_count=failures,
            prior_recovery_count=recoveries,
            contacts_last_30d=int(rng.integers(0, 4)),
            dnd_registered=bool(rng.random() < 0.14),
            hour_of_day=failed_at.hour,
            day_of_month=failed_at.day,
            days_since_salary=days_since_salary,
            mandate_category=category,
            consecutive_mandate_failures=consecutive,
            issuer_failures_last_hour=issuer_failures,
            issuer_volume_last_hour=volume_last_hour,
        )
        features.append(feature)
        decision = log_action(feature, rng)
        logged.append(decision)
        if is_upcoming:
            outcomes.append(
                compute_upcoming_outcomes(
                    case_id=case_id,
                    latent=latent,
                    days_since_salary=days_since_salary,
                    consecutive_failures=consecutive,
                    rng=rng,
                )
            )
        else:
            outcomes.append(
                compute_potential_outcomes(
                    case_id=case_id,
                    latent=latent,
                    reason=reason,
                    issuer=issuer,
                    failed_at=failed_at,
                    timeline=timeline,
                    is_mandate=is_mandate,
                    rng=rng,
                )
            )

    # Reveal Y(a) for the action actually taken. Everything else stays in the
    # oracle. This is a projection of already-computed state, not a new draw,
    # so the random stream is untouched (verified against the frozen batch).
    for decision, outcome in zip(logged, outcomes, strict=True):
        realized.append(
            RealizedOutcome(
                case_id=decision.case_id,
                action=decision.action,
                recovered=outcome.recovered[decision.action],
                mandate_cancelled=outcome.mandate_cancelled[decision.action],
            )
        )

    provenance = f"{params.provenance.source_name} / {params.provenance.reporting_period}"
    return (
        ObservableBatch(
            features=tuple(features),
            logged=tuple(logged),
            realized=tuple(realized),
            seed=seed,
            params_provenance=provenance,
        ),
        OracleBatch(outcomes=tuple(outcomes), seed=seed),
    )


def write_batch(
    observable: ObservableBatch,
    oracle: OracleBatch,
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write observable and oracle data to separate files.

    The oracle goes to `oracle/` in its own subdirectory so that a glob over
    the batch directory cannot pick it up by accident.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir = out_dir / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)

    features_path = out_dir / "cases.jsonl"
    with features_path.open("w", encoding="utf-8") as fh:
        for f in observable.features:
            fh.write(f.model_dump_json() + "\n")

    logged_path = out_dir / "logged_decisions.jsonl"
    with logged_path.open("w", encoding="utf-8") as fh:
        for d in observable.logged:
            fh.write(d.model_dump_json() + "\n")

    realized_path = out_dir / "realized_outcomes.jsonl"
    with realized_path.open("w", encoding="utf-8") as fh:
        for r in observable.realized:
            fh.write(r.model_dump_json() + "\n")

    oracle_path = oracle_dir / "potential_outcomes.jsonl"
    with oracle_path.open("w", encoding="utf-8") as fh:
        for o in oracle.outcomes:
            fh.write(o.model_dump_json() + "\n")

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "seed": observable.seed,
                "n_cases": len(observable.features),
                "calibration_provenance": observable.params_provenance,
                "generated_at": datetime.now(UTC).isoformat(),
                "warning": (
                    "oracle/ contains ground-truth counterfactuals. No policy, "
                    "model or feature-engineering code may read it."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return features_path, logged_path, oracle_path
