"""Issuer degradation detection.

The problem: an issuer's failure rate is observed as `f` failures out of `n`
transactions in the last hour. Is it degraded right now, or is this a normal
hour that happened to look bad?

Why a threshold rule is not good enough. Suppose the trigger is "failure rate
above 3%". A large issuer with 400 transactions and 13 failures (3.25%) fires
the rule on strong evidence. A small issuer with 20 transactions and 1 failure
(5.00%) fires it harder — on a single event. One failure out of twenty is
entirely consistent with a healthy 1% issuer. The rule is most confident
exactly where the data is weakest, and it will suppress retries for small
banks almost continuously.

The fix is to model uncertainty rather than point estimates:

1. **Pool across issuers.** Estimate a population distribution of hourly
   failure rates from the whole batch, by method of moments on a Beta.
2. **Shrink each issuer toward it.** A bank with millions of transactions
   gets a prior close to its own history; a bank with a few thousand gets
   pulled toward the population mean. Nobody has to hand-pick a minimum
   volume.
3. **Ask a posterior question.** `P(rate > k x baseline | f, n)`. With
   n = 20 the posterior stays wide and the probability never gets high
   enough to act on. With n = 400 it sharpens and a real spike is detected.

None of the generator's state is visible here. `world.timeline.is_degraded()`
is the ground truth this module has to infer, and CI forbids importing it
(ADR-0011). The detector is scored against it in `evaluate`, never informed
by it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from recovery.domain.observations import CaseFeatures

DEGRADATION_MULTIPLE = 3.0
"""How far above its own baseline an issuer must plausibly be to count as
degraded. Calibration models episodes at 4x-20x baseline, so 3x sits below
the weakest real episode: we would rather catch a mild one than miss it."""

DEFAULT_DECISION_THRESHOLD = 0.70
"""Posterior probability required before declaring degradation. Tuned on the
holdout in `evaluate`, not chosen by eye."""

PRIOR_STRENGTH = 40.0
"""Pseudo-observations of prior weight per issuer. Roughly: the prior carries
the same influence as 40 observed transactions, so an issuer seen 400 times
this hour is dominated by data while one seen 20 times is not."""

MIN_POPULATION_VARIANCE = 1e-8


class IssuerBaseline(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    issuer: str
    observed_failures: int = Field(ge=0)
    observed_volume: int = Field(ge=0)
    raw_rate: float = Field(ge=0.0, le=1.0)
    shrunk_rate: float = Field(ge=0.0, le=1.0)
    prior_alpha: float = Field(gt=0.0)
    prior_beta: float = Field(gt=0.0)

    @property
    def shrinkage_weight(self) -> float:
        """Share of the estimate coming from this issuer's own data."""
        n = self.observed_volume
        return n / (n + PRIOR_STRENGTH) if n else 0.0


class HealthAssessment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    issuer: str
    observed_failures: int
    observed_volume: int
    baseline_rate: float
    posterior_mean: float
    degradation_probability: float = Field(ge=0.0, le=1.0)
    is_degraded: bool
    evidence: str


class IssuerHealthModel:
    """Empirical-Bayes detector. Fit on observable data only."""

    def __init__(
        self,
        *,
        degradation_multiple: float = DEGRADATION_MULTIPLE,
        decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
        prior_strength: float = PRIOR_STRENGTH,
    ) -> None:
        self.degradation_multiple = degradation_multiple
        self.decision_threshold = decision_threshold
        self.prior_strength = prior_strength
        self._baselines: dict[str, IssuerBaseline] = {}
        self._population_rate: float = 0.0
        self._fitted = False

    # -- fitting -----------------------------------------------------------

    def fit(self, features: Sequence[CaseFeatures]) -> IssuerHealthModel:
        """Learn per-issuer baselines from observed hourly windows.

        Note what this uses: only `issuer_failures_last_hour` and
        `issuer_volume_last_hour`, both of which a production system reads off
        its own gateway logs. No published rate, no generator config.
        """
        totals: dict[str, list[int]] = {}
        for f in features:
            entry = totals.setdefault(f.issuer, [0, 0])
            entry[0] += f.issuer_failures_last_hour
            entry[1] += f.issuer_volume_last_hour

        rates = [fails / vol for fails, vol in totals.values() if vol > 0]
        if not rates:
            raise ValueError("no observed volume; cannot fit issuer baselines")

        # Method-of-moments Beta fit to the population of issuer rates. This
        # is what makes the shrinkage empirical rather than a guessed prior.
        mean = float(np.mean(rates))
        var = max(float(np.var(rates)), MIN_POPULATION_VARIANCE)
        self._population_rate = mean

        if var >= mean * (1 - mean):
            # Over-dispersed relative to a Beta; fall back to a weak prior
            # rather than producing negative parameters.
            pop_alpha, pop_beta = mean * 2.0, (1 - mean) * 2.0
        else:
            common = mean * (1 - mean) / var - 1.0
            pop_alpha = max(mean * common, 1e-3)
            pop_beta = max((1 - mean) * common, 1e-3)

        for issuer, (fails, vol) in totals.items():
            raw = fails / vol if vol else mean
            # Shrink toward the population mean in proportion to volume.
            weight = vol / (vol + self.prior_strength) if vol else 0.0
            shrunk = weight * raw + (1 - weight) * mean
            scale = pop_alpha + pop_beta
            self._baselines[issuer] = IssuerBaseline(
                issuer=issuer,
                observed_failures=fails,
                observed_volume=vol,
                raw_rate=min(raw, 1.0),
                shrunk_rate=min(shrunk, 1.0),
                prior_alpha=max(shrunk * scale, 1e-3),
                prior_beta=max((1 - shrunk) * scale, 1e-3),
            )

        self._fitted = True
        return self

    # -- inference ---------------------------------------------------------

    def assess(self, features: CaseFeatures) -> HealthAssessment:
        """Posterior probability that this issuer is degraded right now."""
        if not self._fitted:
            raise RuntimeError("IssuerHealthModel.fit() must be called first")

        baseline = self._baselines.get(features.issuer)
        if baseline is None:
            # Unseen issuer: fall back to the population, and say so.
            baseline = IssuerBaseline(
                issuer=features.issuer,
                observed_failures=0,
                observed_volume=0,
                raw_rate=self._population_rate,
                shrunk_rate=self._population_rate,
                prior_alpha=max(self._population_rate * 2.0, 1e-3),
                prior_beta=max((1 - self._population_rate) * 2.0, 1e-3),
            )

        f = features.issuer_failures_last_hour
        n = features.issuer_volume_last_hour
        alpha = baseline.prior_alpha + f
        beta = baseline.prior_beta + max(n - f, 0)

        cutoff = min(baseline.shrunk_rate * self.degradation_multiple, 0.999)
        prob = float(stats.beta.sf(cutoff, alpha, beta))
        posterior_mean = float(alpha / (alpha + beta))

        degraded = prob >= self.decision_threshold
        evidence = (
            f"{f}/{n} failures last hour "
            f"({(f / n if n else 0.0):.2%}) vs baseline {baseline.shrunk_rate:.2%}; "
            f"P(rate > {cutoff:.2%}) = {prob:.2f}"
        )
        return HealthAssessment(
            issuer=features.issuer,
            observed_failures=f,
            observed_volume=n,
            baseline_rate=baseline.shrunk_rate,
            posterior_mean=posterior_mean,
            degradation_probability=prob,
            is_degraded=degraded,
            evidence=evidence,
        )

    def baseline_for(self, issuer: str) -> IssuerBaseline | None:
        return self._baselines.get(issuer)

    @property
    def population_rate(self) -> float:
        return self._population_rate


class ThresholdHealthModel:
    """Naive baseline: fire whenever the observed rate clears a fixed bar.

    Kept as a comparator, not as a fallback. Its failure mode is instructive
    and shows up clearly in evaluation: it is most confident on the smallest
    samples, so it fires constantly for low-volume issuers and suppresses
    retries that would have succeeded.
    """

    def __init__(self, threshold: float = 0.03) -> None:
        self.threshold = threshold

    def fit(self, features: Sequence[CaseFeatures]) -> ThresholdHealthModel:
        return self

    def assess(self, features: CaseFeatures) -> HealthAssessment:
        n = features.issuer_volume_last_hour
        f = features.issuer_failures_last_hour
        rate = f / n if n else 0.0
        return HealthAssessment(
            issuer=features.issuer,
            observed_failures=f,
            observed_volume=n,
            baseline_rate=self.threshold,
            posterior_mean=rate,
            degradation_probability=1.0 if rate > self.threshold else 0.0,
            is_degraded=rate > self.threshold,
            evidence=f"{rate:.2%} vs fixed threshold {self.threshold:.2%}",
        )
