# Decision log

Every non-obvious choice, with the reasoning and the alternative that was
rejected. Appended to as the build progresses — including the decisions that
turned out to be wrong and had to be reversed.

---

## ADR-0001 — No agent framework for the decision loop

**Date:** Phase 0
**Status:** Accepted

**Context.** The recovery loop (diagnose → score → decide → gate → execute →
observe) is naturally expressible in LangGraph or a similar orchestration
framework, and doing so would have been faster to stand up.

**Decision.** The loop is a hand-written state machine with an explicit
`CaseState` enum and explicit transitions.

**Reasoning.** The track's bar is that every money action be *explainable,
bounded and gated*. An orchestration framework moves control flow into
library internals, which means the audit trail records what the framework
decided to do rather than what we specified. When the artifact under review
is the audit trail itself, that indirection is a liability. An explicit state
machine is also replayable from the ledger, which a framework's internal
scheduling is not.

**Rejected alternative.** LangGraph. Faster to build, harder to defend.

**Cost accepted.** More code to write and test.

---

## ADR-0002 — Ground-truth outcomes are quarantined by CI, not by convention

**Date:** Phase 0
**Status:** Accepted

**Context.** The world simulator generates potential outcomes Y(a) for every
action. If any model or policy code reads them, every metric reported by this
project is invalid — and the leak would be invisible in the results.

**Decision.** Ground truth lives in `recovery.world.oracle`. An import-linter
contract in `pyproject.toml` forbids `diagnose`, `uplift`, `policy`,
`compliance`, `execute` and `api` from importing it. CI runs `lint-imports`
and fails the build on violation.

**Reasoning.** "We were careful not to leak" is a claim. A failing build is
evidence. A reviewer can verify this in thirty seconds without reading the
model code.

---

## ADR-0003 — Money is integer paise, everywhere

**Date:** Phase 0
**Status:** Accepted

**Decision.** All monetary values are `int` paise. No floats, no `Decimal`,
no rupee-denominated arithmetic. Conversion to rupees happens only at
presentation boundaries.

**Reasoning.** This project's headline claim is a rupee figure. Float
accumulation error across a 500-case batch is small but non-zero, and a
recovery total that does not reconcile exactly against the ledger would
undermine the one number the whole submission rests on. Razorpay's own APIs
denominate in paise, so this also removes a conversion boundary.

---

## ADR-0004 — Compliance rules are configuration, not code

**Date:** Phase 1
**Status:** Accepted

**Decision.** Regulatory and operating rules live in
`configs/compliance/policy.yaml` with a cited basis per rule. The compliance
engine interprets them; it does not hard-code them.

**Reasoning.** Three benefits. An auditor can read the rules without reading
Python. A rule change appears as a diff in one reviewable file rather than
scattered across conditionals. And the distinction between `regulation` and
`operating_policy` is made explicit per rule, which forces us to be honest
about which constraints are legally required and which are our own judgment.

---

## ADR-0005 — `propensity` is a required field on `Decision`

**Date:** Phase 1
**Status:** Accepted

**Decision.** The schema refuses to construct a `Decision` without a logged
action-selection probability in `(0, 1]`.

**Reasoning.** Inverse-propensity and doubly-robust estimators are undefined
without it, and a zero propensity divides by zero. Making it optional would
mean discovering at evaluation time that some slice of logged decisions is
unusable. Making it mandatory in the type system means that class of bug
cannot reach the evaluation stage.

---

## Incidents

> Running log of things that broke, what the symptom was, what the cause
> turned out to be, and what changed as a result. Appended in real time —
> this is not reconstructed after the fact.

_(none yet)_
