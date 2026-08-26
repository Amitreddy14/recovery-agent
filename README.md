# Recovery Agent

A compliance-gated agent that recovers revenue from failed payments and failed
recurring mandates — and knows when not to try.

> **Status:** Phase 1 of 11. Domain schema and compliance rules complete.
> See [DECISIONS.md](DECISIONS.md) for the reasoning behind each choice.

---

## The thesis

Most recovery systems rank customers by *risk* and intervene on the riskiest.
That is the wrong objective. The population splits four ways:

| Segment | Behaviour | Correct action |
|---|---|---|
| Sure things | Pay regardless | Do nothing — intervention is pure cost |
| Lost causes | Never pay | Do nothing — intervention is pure cost |
| **Persuadables** | Pay **only if** intervened on | **Intervene. This is the recoverable money.** |
| Sleeping dogs | Would have paid, but contact makes them cancel | **Do nothing — intervention destroys revenue** |

Risk-ranking cannot tell these apart, because it predicts *outcome*, not
*effect of treatment on outcome*. This project ranks by **uplift** instead,
and prices the cost of being wrong in both directions.

Three claims, each measured against a baseline:

1. **Targeting on uplift beats targeting on risk** at equal contact budget.
2. **The mandatory pre-debit notification is a free recovery channel.** RBI's
   2026 e-mandate framework requires a notification 24 hours before every
   recurring debit. Merchants send a template. We send a risk-scored,
   compliance-valid message at the moment a balance top-up would prevent the
   failure — moving recovery upstream of the failure at zero incremental
   contact cost.
3. **Retry timing against live issuer health** recovers money with no customer
   contact at all.

## Evaluation integrity

Outcomes are simulated; execution is real. The boundary is stated explicitly
rather than blurred.

- **Calibrated, not invented.** Issuer decline rates are derived from NPCI's
  published per-bank Technical Decline / Business Decline statistics, not
  from assumed numbers.
- **Generator frozen before policy.** The world simulator is committed and
  tagged before any policy code is written. Git history is the proof.
- **Ground truth is quarantined by CI.** Potential outcomes live in
  `recovery.world.oracle`; an import-linter contract forbids any decision-making
  module from importing it, and the build fails on violation.
- **Reported against an oracle ceiling.** Because the data-generating process
  is known, the theoretically optimal policy is computable. Results are
  reported as a percentage of it.
- **Robustness sweep, not a single world.** Results are reported across a grid
  of generator configurations, including the regions where the approach
  does *not* win.
- **Off-policy estimators validated against truth.** In production the
  counterfactual is unobservable, so an OPE estimator can never be checked.
  Here it can — so we check it, and report the error.

## Architecture

```
ingest → diagnose → score (uplift) → decide (expected value)
                                          ↓
                                   compliance gate
                                    ↓           ↓
                                 allow        block / defer
                                    ↓           ↓
                                 execute     logged with the rule that fired
                                    ↓
                                  ledger (append-only, replayable)
```

The decision loop is a hand-written state machine, not an agent framework —
see ADR-0001. LLM calls are confined to three named nodes: diagnosis
narration, message drafting, and exception explanation. No LLM performs
arithmetic or selects an action.

## Quickstart

```bash
make install
make check      # lint + types + import contracts + tests
```

## Layout

```
configs/compliance/policy.yaml   Regulatory + operating rules, as data
src/recovery/domain/             Schema. Imports nothing internal.
src/recovery/world/              Simulator
src/recovery/world/oracle/       QUARANTINED ground truth
src/recovery/diagnose/           Root-cause attribution
src/recovery/uplift/             T-learner / X-learner
src/recovery/policy/             Expected-value decision engine + baselines
src/recovery/compliance/         Gate engine
src/recovery/execute/            Razorpay test-mode adapter
src/recovery/ledger/             Append-only audit trail
src/recovery/evaluate/           Baselines, oracle, OPE, robustness sweep
```

## Compliance basis

Rules are cited individually in `configs/compliance/policy.yaml` and marked
either `regulation` or `operating_policy`. Primary sources:

- RBI **Digital Payments – E-mandate Framework, 2026** (issued 21 April 2026)
- TRAI **TCCCPR, 2018** — commercial communication preferences
- NPCI **Circular OC-149** — decline classification and targets

## Licence

MIT
