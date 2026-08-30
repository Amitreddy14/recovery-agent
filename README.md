# Recovery Agent

A compliance-gated agent that recovers revenue from failed payments and failed
recurring mandates — and prices the cost of chasing money it shouldn't.

Built for Razorpay Buildathon 2026, Track 3.

```bash
docker compose up          # console on http://localhost:8000
```

---

## The result

40,000 cases. `risk_topN` and `uplift_ev` are given the **same contact budget**
(6,952). Net is after subtracting intervention cost and the mandate value
destroyed by contacting the wrong people.

| policy | recovered | forfeited | **net** | cancels | dogs contacted |
|---|---:|---:|---:|---:|---:|
| do nothing | 128,802,859 | 0 | **128,802,859** | 0 | 0 |
| blind retry | 151,595,555 | 45,550,015 | **106,029,698** | 323 | 2,441 |
| risk ranking | 131,238,808 | 13,363,779 | **117,872,596** | 106 | 133 |
| **uplift + EV** | 136,760,437 | 5,935,562 | **130,813,491** | 22 | 833 |
| oracle ceiling | 183,463,827 | 0 | **183,450,393** | 0 | 1,304 |

**Blind retry — what most merchants do — recovers ₹2.28cr more than doing
nothing and ends up ₹2.28cr worse off.** It destroys more mandate value than
it recovers in payments.

Reproduce: `python scripts/compare_policies.py`

---

## Three claims, each measured against a baseline

### 1. Targeting on uplift beats targeting on risk

Risk ranking predicts *who will fail*. That is the wrong quantity. The
population splits four ways, and only one of them is worth contacting:

| segment | behaviour | correct action |
|---|---|---|
| sure things | pay regardless | do nothing — intervention is pure cost |
| lost causes | never pay | do nothing — intervention is pure cost |
| **persuadables** | pay **only if** contacted | **contact** |
| sleeping dogs | would have paid until you messaged them | **do not contact** |

Who each ranking spends its budget on, top 30%:

| segment | uplift | risk |
|---|---:|---:|
| persuadable | **77%** | 40% |
| lost cause | **7%** | **52%** |

**Risk ranking spends over half its contact budget on customers who will never
pay.** Ranking by probability of failure selects exactly the people whose
failure is least changeable.

Across 5 independent worlds of 90,000 cases, X-learner beats risk ranking
**5/5** (Qini 0.261 ± 0.096 against 0.118 ± 0.132).

Reproduce: `python -m recovery.cli uplift --cases 90000 --seeds 5`

### 2. Compliance obligation as a free recovery channel

RBI's [E-mandate Framework, 2026](https://www.rbi.org.in) requires a
notification at least 24 hours before every recurring debit — merchant name,
amount, debit time, mandate reference, reason, opt-out. Merchants treat it as
a compliance checkbox.

It is a legally mandated, zero-marginal-cost touchpoint with the customer,
24 hours before money moves, at precisely the moment a balance top-up would
prevent the failure.

The agent scores upcoming mandates *before* they fail and uses the mandatory
notification as a targeted intervention. Of the 833 sleeping dogs it contacts,
**100% receive a pre-debit notification and none receives a payment link** —
irritation 0.08 against 0.30. It reaches a population that risk ranking must
avoid, through a message that was going out anyway.

### 3. Retry timing against inferred issuer health

Technical declines cluster. Retrying into a degraded issuer burns an attempt
against a hard per-case cap for near-zero probability of success.

The agent infers degradation from the observable failure stream using an
empirical-Bayes posterior — never from the simulator's state, which CI
forbids it from importing.

| detector | precision | recall | cost/case |
|---|---:|---:|---:|
| fixed 3% threshold | 0.294 | 0.635 | 0.0479 |
| **empirical Bayes** | **0.424** | **0.676** | **0.0385** |

On the lowest-volume third of issuers, the fixed rule fires **3.2%** of the
time at **0.188** precision; the Bayesian detector fires **1.2%** at **0.333**.
A threshold is most confident exactly where the evidence is weakest — one
failure in twenty is a 5% rate on a single event.

Reproduce: `python -m recovery.cli diagnose --cases 8000`

---

## Why the numbers are believable

Simulated outcomes, real execution. The boundary is stated, not blurred.

**Calibrated, not invented.** Per-issuer decline rates come from NPCI's
published Top 50 Remitter table for July 2026 — all 48 issuing banks, two
non-bank entities excluded. Volume-weighted TD 0.402%, BD 10.852%, approval
88.75%. Assumptions are isolated in one file, enumerated in a registry, and a
test fails if one is added without registering it.

**Ground truth is quarantined by CI, not by convention.** Counterfactuals live
in `recovery.world`; an import-linter contract forbids every decision module
from importing it, and the build fails on violation. It has caught two real
violations — `world.timeline` exposing `is_degraded()` to the diagnosis layer
(ADR-0011), and the console's snapshot builder giving the web layer a
transitive path to the oracle (INC-026).

```bash
lint-imports        # Contracts: 2 kept, 0 broken
```

**Generator frozen before the policy existed.** Tagged `v0.4-world-frozen`.
Git history is the proof.

**Reported across worlds, never one.** A single Qini estimate here has a
standard error comparable to the effect. The first run produced *negative*
Qini for both learners — an apparent refutation of the whole thesis — which a
sample-size sweep showed to be variance (INC-016).

**Off-policy estimators validated against truth.** In production the
counterfactual is unobservable, so an estimator can never be checked. Here it
can:

| | effective sample | IPS | SNIPS | doubly robust |
|---|---:|---:|---:|---:|
| naive log | 3.5% of n | **+30.9%** | +8.5% | +15.7% |
| randomised holdout | 16.6% of n | −5.1% | −5.0% | −5.0% |

Doubly robust did **not** rescue the biased log. Off-policy accuracy is
governed by overlap, not estimator sophistication — which is why 20% of cases
are logged with uniformly random actions.

**Money reconciles exactly.** All amounts are integer paise. The ledger sum
equals the reported total with zero tolerance, not a threshold.

**Swept, including where it loses.** Across 16 configurations of the
assumption space: beats risk ranking **16/16**, beats blind retry 14/16, beats
*doing nothing* **12/16**. It loses where a long cancellation horizon meets a
weak salary-window effect — recorded in INC-023 rather than tuned away.

---

## Architecture

```
ingest → diagnose → score (uplift) → decide (expected value)
                                          ↓
                                   compliance gate
                                    ↓           ↓
                                 allow        block / defer
                                    ↓           ↓
                              execute      logged with the rule that fired
                                    ↓
                              ledger (append-only, replayable)
```

Three decisions worth defending:

**No agent framework.** The loop is a hand-written state machine with an
explicit state enum. Every money action must be explainable and replayable
from the ledger; a framework's internal scheduling is not (ADR-0001).

**Rules where rules are right, models where they are not.** Failure reason →
recoverability is a lookup table, because Razorpay's error contract already
tells us what happened and no training data improves on "an expired card
cannot be fixed by retrying it." Issuer degradation is a model, because it is
genuinely uncertain and must be inferred from a small noisy sample (ADR-0012).

**LLMs draft, the system verifies.** Three named nodes only. The pre-debit
notification is generated from a deterministic template, optionally rephrased
by a model, and always re-validated against the required-field list in
`policy.yaml`. A fluent notification missing its opt-out instruction is a
compliance failure that reads perfectly well — the validator does not care how
the sentence reads (ADR-0023).

**No double charges.** Idempotency keys are deterministic in
`(case_id, action, attempt, amount)`, passed to Razorpay as the order receipt
so the provider rejects duplicates independently, and written to a SQLite
ledger *before* the request so a mid-flight crash leaves a recoverable trace
(ADR-0020).

---

## The audit trail

```
$ recovery audit --case case_1

2026-07-06 09:00  ingested mandate_failure Rs 999.00 reason=insufficient_funds issuer=SBI
2026-07-06 09:00  diagnosed funding (confidence 0.88)
2026-07-06 09:00  send_payment_link BLOCKED by DND_REGISTRY
2026-07-06 11:00  not recovered (contact blocked)
```

₹999 the agent could have chased and deliberately did not, with the rule
named. Case state is derived by folding an append-only event log, never
stored and mutated — a status column cannot answer *what did you consider and
reject, and on what evidence*. Append-only is enforced by SQL triggers, not
convention (ADR-0024).

Denials are a first-class artifact. A run reporting zero of them is either
operating in a world with no rules or is not checking (ADR-0026).

---

## Running it

```bash
docker compose up                  # console on :8000, snapshot pre-built
```

Or locally:

```bash
pip install -e ".[dev]"
make check                         # lint, types, import contracts, 240 tests

python -m recovery.cli calibrate --snapshot data/external/npci/snapshot_2026_07.csv
python -m recovery.cli generate  --cases 5000 --seed 42
python -m recovery.cli diagnose  --cases 8000
python -m recovery.cli uplift    --cases 90000 --seeds 5
python -m recovery.cli console   --build --serve
python -m recovery.cli audit     --case <id>
```

Generated batches are not committed — they are reproducible from seed.
Counterfactuals are written to `data/generated/batch/oracle/`, which no policy
or model module can read.

### Reproducing the reported numbers

| script | reproduces |
|---|---|
| `scripts/compare_policies.py` | the policy comparison table (ADR-0018) |
| `scripts/sweep_run.py` | the robustness sweep (INC-023) |
| `recovery uplift --cases 90000 --seeds 5` | the targeting comparison (ADR-0017) |
| `recovery diagnose --cases 8000` | the degradation detector (ADR-0013) |

---

## What is not finished

**The policy loses to inaction in a quarter of the assumption space**
(INC-023). It beats risk ranking everywhere tested, but in 4 of 16
configurations doing nothing would have been better. The mechanism is
identified — a long cancellation horizon makes it conservative for the wrong
reason — and two fixes are proposed but not built.

**Uplift ranking does not avoid sleeping dogs** (INC-017). It contacts six
times more of them than risk ranking. The expected-value layer compensates by
routing them to the cheap channel, which turned out to be better than
avoidance, but the ranking itself remains blind to negative uplift.

**Scheduled actions are decided but not dispatched.** Deferred actions record
a `defer_until` timestamp; nothing runs them yet.

---

## Compliance basis

Rules live in `configs/compliance/policy.yaml`, cited individually and marked
`regulation` or `operating_policy`, so a regulatory change is a config diff
rather than a code change.

- RBI **Digital Payments – E-mandate Framework, 2026** (21 April 2026)
- TRAI **TCCCPR, 2018** — commercial communication preferences
- NPCI **Circular OC-149** — decline classification and targets
- NPCI **UPI Ecosystem Statistics**, Top 50 Remitter, July 2026

---

## Decision log

[`DECISIONS.md`](DECISIONS.md) — 31 architecture decisions and 27 incidents,
written as they happened. Four worth reading:

- **INC-016** — an apparent negative result that was sampling noise. The
  worrying part is the asymmetry: had the first seed produced a *good* number,
  it would have shipped unexamined.
- **INC-019** — the sleeping-dog correction silently did nothing for a full
  run, because cancellation was framed as an uplift problem when the control
  arm has structurally zero events. Caught only because `do_nothing` was in
  the comparison.
- **INC-025** — a baseline reimplemented at a second call site drifted from
  the one the reported numbers came from, inverting risk ranking into a
  sure-thing picker. The error flattered our own policy by a factor of three.
- **INC-022** — found by a live API call, not by 166 passing tests. A test
  suite that has never touched the provider is testing your model of the
  provider.

## Licence

MIT
