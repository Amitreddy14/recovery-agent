"""Latent customer state and the observable proxies derived from it.

The policy never sees a latent trait. It sees noisy, aggregated proxies -
prior failure counts, tenure, and so on - exactly as a production system
would. The gap between latent truth and observable proxy is what makes the
targeting problem non-trivial; a generator that exposed the latents directly
would produce an uplift model that looks far better than any real one.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class LatentCustomer(BaseModel):
    """Hidden traits. Never serialised into the observable case file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent: float = Field(ge=0.0, le=1.0)
    """Whether the customer actually wanted to complete this payment.
    Low intent is the dominant driver of lost causes."""

    balance_recovery_rate: float = Field(ge=0.0)
    """Daily hazard that funds become available after an insufficient-funds
    decline. Drives whether a *scheduled* retry beats an immediate one."""

    contact_sensitivity: float = Field(ge=0.0, le=1.0)
    """How much an unsolicited message costs. High values produce sleeping
    dogs: customers who would have paid until we messaged them."""

    link_responsiveness: float = Field(ge=0.0, le=1.0)
    """Propensity to act on a payment link. Independent of intent - plenty of
    willing customers simply ignore SMS."""

    rail_flexibility: float = Field(ge=0.0, le=1.0)
    """Willingness to complete on a different payment method."""


def sample_latent(rng: np.random.Generator) -> LatentCustomer:
    """Draw one customer's hidden traits.

    Intent is deliberately bimodal rather than a smooth beta: real checkout
    populations contain a mass of committed payers and a mass of
    browsers/abandoners, and a unimodal distribution would understate both
    the lost-cause and sure-thing segments.
    """
    committed = rng.random() < 0.72
    intent = float(rng.beta(6.0, 2.0)) if committed else float(rng.beta(1.6, 5.0))

    return LatentCustomer(
        intent=intent,
        balance_recovery_rate=float(rng.gamma(shape=2.0, scale=0.18)),
        contact_sensitivity=float(rng.beta(1.8, 5.0)),
        link_responsiveness=float(rng.beta(2.4, 3.2)),
        rail_flexibility=float(rng.beta(3.0, 2.6)),
    )


def observable_history(
    latent: LatentCustomer,
    tenure_days: int,
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    """Derive (payments, failures, recoveries) as noisy proxies for latents.

    Returns counts a production system would actually have. The mapping is
    intentionally lossy: a low-intent customer *tends* to have more failures,
    but the observed count is a small-sample draw, so the proxy is weak for
    short-tenure customers. That short-tenure ambiguity is a large part of
    why uplift targeting is hard, and it must be present in the data.
    """
    activity_rate = 0.02 + 0.06 * latent.intent
    payments = int(rng.poisson(max(1.0, tenure_days * activity_rate)))

    fail_prob = 0.06 + 0.22 * (1.0 - latent.intent)
    failures = int(rng.binomial(payments + 1, min(fail_prob, 0.95)))

    recover_prob = 0.25 + 0.5 * latent.intent
    recoveries = int(rng.binomial(failures, min(recover_prob, 0.95)))

    return payments, failures, recoveries
