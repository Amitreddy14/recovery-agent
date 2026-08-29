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

**Date:** Phase 0
**Status:** Superseded by ADR-0011 (Phase 4). The contract described here
forbade only `recovery.world.oracle`, which left `world.timeline` and
`world.latent` reachable from decision code. Retained for the record.


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

## ADR-0006 — Calibration is split into Tier 1 (empirical) and Tier 2 (assumed)

**Date:** Phase 2
**Status:** Accepted

**Context.** NPCI publishes per-bank Technical Decline and Business Decline
rates monthly, but not the reason-level composition inside those buckets, and
not the time-clustering of failures. The simulator needs both.

**Decision.** Parameters are split by epistemic status.

- **Tier 1** lives in the ingested snapshot: per-issuer TD/BD rates and volume
  shares. Sourced, versioned, and not overridable — `calibrate()` deliberately
  exposes no keyword that can alter a published figure, and a test asserts this.
- **Tier 2** lives in `calibration/assumptions.py`: reason mixes, degradation
  dynamics, salary-window effects. Every entry is registered in
  `ASSUMPTION_REGISTRY`, and a test fails if an assumption is added without
  registering it.

**Reasoning.** The credibility of every rupee figure downstream rests on a
reviewer being able to see exactly which numbers are evidence and which are
modelling choices. Mixing them in one config would make the whole set look
invented. Isolating Tier 2 also gives Phase 9's robustness sweep a precise
target: perturb the registry, hold the snapshot fixed.

**Rejected alternative.** One flat generator config. Faster, but indefensible
under questioning.

---

## ADR-0007 — Published TD is treated as a mixture, not a constant rate

**Date:** Phase 2
**Status:** Accepted

**Context.** NPCI's monthly per-bank TD is an average that already contains
degradation episodes. Applying it uniformly in time would produce a world with
no clustering — and the "retry timing against issuer health" thesis would be
untestable by construction, because there would never be a bad moment to avoid.

**Decision.** Calibration solves for a *baseline* rate below the published
figure, such that baseline plus sampled degradation episodes reproduces the
published mean:

    baseline = published / (1 - f + m * f)

where `f` is the long-run degraded time fraction and `m` the TD multiplier
while degraded.

**Reasoning.** The simulator must match published aggregates *and* exhibit the
temporal structure a recovery policy has to cope with. Doing only the first is
a world too easy to win in; doing only the second abandons the calibration.

**Verification.** `test_round_trips_to_published` asserts the inversion is
exact for every issuer. If it breaks, the simulator no longer reproduces
published decline rates and the calibration claim is void.

---

## ADR-0008 — Potential outcomes use one coupled draw per case

**Date:** Phase 3
**Status:** Accepted

**Context.** For each case the world must produce Y(a) for every action,
including the six not taken.

**Decision.** One uniform draw `u ~ U(0,1)` per case, with
`Y(a) = 1 iff u < p(a)`.

**Reasoning.** Independent draws per action would make outcomes conditionally
independent given the probabilities, which is wrong twice over. It breaks the
realism — a customer who would pay under a weak intervention almost certainly
pays under a stronger one — and it manufactures apparent treatment effects out
of sampling noise, so an uplift model would appear to work on data containing
no real heterogeneity. With the coupled draw, uplift is exactly `p(a) - p(0)`,
including negative values, which is where sleeping dogs live.

**Verification.** `test_coupled_draw_preserves_outcome_monotonicity`.

---

## ADR-0009 — Segments are read off outcomes, never used to generate them

**Date:** Phase 3
**Status:** Accepted

**Decision.** `outcomes.py` computes p(a) from latent traits, failure reason
and issuer state. `segments.py` then classifies a case by comparing those
probabilities. Generation never consults a segment label.

**Reasoning.** The direction of the arrow is the whole argument. If generation
started from "make this one a sleeping dog", any uplift model fitted
downstream would be recovering our labels rather than learning a causal
effect, and every number in the submission would be circular. Reading labels
off emergent probabilities means the model has to find structure we did not
hand it.

---

## ADR-0010 — Human escalation is excluded from segmentation

**Date:** Phase 3
**Status:** Accepted

**Context.** `ESCALATE_HUMAN` has roughly +0.29 mean uplift and lifts almost
any case. Including it in segmentation classified every customer as
persuadable and collapsed the segment structure to a single bucket.

**Decision.** Segmentation considers only the automated action set.

**Reasoning.** "Would an intervention change this outcome" is a question about
the actions being targeted. Human escalation is not a targeting choice but an
expensive fallback, priced by the decision engine rather than selected by the
uplift model. Its cost is roughly 80x a payment link, so it is never the
answer to "who should we contact".

---

## ADR-0011 — Decision code cannot import the world at all

**Date:** Phase 4
**Status:** Accepted. Supersedes the narrower contract in ADR-0002.

**Context.** The original contract forbade decision modules from importing
`recovery.world.oracle`. Starting Phase 4 exposed the gap: `world.timeline`
exposes `is_degraded()` — precisely the fact the diagnosis layer exists to
infer — and `world.latent` holds the hidden customer traits. Neither lives
under `oracle/`, so both were reachable from `diagnose`, `uplift` and
`policy`.

**Decision.** `CaseFeatures` and `LoggedDecision` moved to
`recovery.domain.observations`, and the contract now forbids
`recovery.world` entirely.

**Reasoning.** Those two models are the interface between simulator and
policy, not part of the simulator, so `domain` is where they belong on
architectural grounds alone. Moving them also removes the last reason
decision code had to import `world`, which turns a contract with a carve-out
into a flat prohibition. "The policy cannot import the world" is a claim a
reviewer can check in one line; "the policy cannot import certain modules
within the world" is one they have to audit.

**Cost accepted.** A refactor mid-build. Batch output verified byte-identical
afterwards.

---

## ADR-0012 — Root cause is rules; issuer health is a model

**Date:** Phase 4
**Status:** Accepted

**Decision.** `taxonomy.py` maps failure reason to recoverability class by
lookup. `issuer_health.py` estimates degradation with an empirical-Bayes
posterior.

**Reasoning.** These are different kinds of question. What an error code
means is knowledge we already have from Razorpay's error contract; no volume
of training data improves on "an expired card cannot be fixed by retrying
it," and a model there would be slower, unexplainable in an audit trail, and
less accurate than the rule. Whether an issuer is degraded *right now* is
genuinely uncertain, must be inferred from a small noisy sample, and is
exactly where a model earns its place.

This is the concrete answer to the track's "where you chose not to use AI."

---

## ADR-0013 — Detection threshold is chosen on expected cost, not F1

**Date:** Phase 4
**Status:** Accepted

**Context.** The two detection errors have very different consequences. A
false positive defers a retry that would have worked — a delay. A false
negative retries into a degraded issuer, burning an attempt against a hard
per-case cap and forfeiting most of the recovery probability.

**Decision.** False negatives are weighted 10x false positives, and the
operating threshold is selected by minimising expected cost.

**Reasoning.** F1 weights both errors equally, which is simply false here. On
the current batch F1 prefers 0.90 while cost prefers 0.80: the higher
threshold looks better on a symmetric metric precisely because it trades
recall for precision, and recall is the expensive side.

**Honest caveat.** The 10:1 ratio is a modelling judgement, not a measured
quantity. It is registered as a Tier-2 assumption and swept in Phase 9.

---

## ADR-0014 — Uplift learners implemented rather than imported

**Date:** Phase 5
**Status:** Accepted

**Decision.** T-learner and X-learner written directly on LightGBM; Qini and
AUUC written from the definition. `causalml` and `scikit-uplift` not used.

**Reasoning.** Two specific reasons, not a preference for hand-rolling.

The X-learner's final blend weights `tau0` and `tau1` by propensity. Library
implementations estimate a propensity score from the data. We *know* the
assignment probabilities exactly, because the logging policy records them
(ADR-0005), so using an estimate would introduce error we do not have to
accept. That substitution is a modification a library default would obscure.

Qini is one formula whose entire content is a correction term. Writing it
makes explicit why `Y_t(k) - Y_c(k)` alone is wrong: the arms are not the same
size at depth k, so the control count must be rescaled by `N_t/N_c`, and
without that an arm the logging policy favoured looks better purely for being
larger.

**Cost accepted.** Roughly 200 lines to write and test.

---

## ADR-0015 — Treatment is defined as the payment link alone

**Date:** Phase 5
**Status:** Accepted

**Decision.** For the headline uplift comparison, "treated" means
`SEND_PAYMENT_LINK`. Retries are excluded as cheap and silent;
`PRE_DEBIT_NUDGE` is excluded despite also being a contact.

**Reasoning.** Contact is the action worth targeting: it consumes a capped
budget, is constrained by compliance rules, and is the only action that can
make a case worse. Retries share none of those properties.

The nudge is excluded for a stronger reason. A pre-debit notification cannot
precede a debit that has already failed, so on the one-off payment population
its potential outcome equals the do-nothing outcome exactly. Pooling it with
payment links labelled thousands of untreated units as treated (INC-013). It
is evaluated separately on the prevention population, where it is a genuine
intervention.

---

## ADR-0016 — The randomised holdout is split, not used whole for evaluation

**Date:** Phase 5
**Status:** Accepted

**Context.** The logging policy imitates a real merchant and therefore almost
never chooses to do nothing. Training on logged data alone gave roughly 800
control observations against 4,400 treated.

**Decision.** Half the randomised holdout joins the training set; the other
half is reserved for evaluation and never trained on.

**Reasoning.** Uplift is `mu1 - mu0`, so the control arm's model is half the
estimate. Fitting it on 800 rows made `mu0` the noisy term, and the difference
inherited that noise — which is why risk ranking initially outscored the
uplift learners (INC-014). Moving half the holdout into training raised
control to ~2,200 and reversed the result.

The reserved half remains uniformly randomised, so treatment assignment there
is independent of covariates and the reported Qini figures stay unbiased. The
cost is a smaller evaluation set, which is the right trade: a noisy unbiased
estimate is usable, a precise biased one is not.

**Status:** Closed in Phase 6 by ADR-0018. Cancellation is now priced as an
explicit rupee cost rather than left to the ranking. At equal contact budget,
sleeping dogs contacted fell from 1,451 (risk ranking) to 833, and mandate
cancellations from 315 to 22.

---

## ADR-0018 — Mandate cancellation is priced in rupees, not ranked

**Date:** Phase 6
**Status:** Accepted. Closes INC-017.

**Context.** INC-017 recorded that uplift targeting selected *more* sleeping
dogs than risk ranking (9% against 3%). The learner finds the positive tail of
the treatment-effect distribution and is largely blind to the negative one,
because contact sensitivity has only a weak observable proxy.

**Decision.** Rather than improve the ranking, add a second objective and
subtract it in rupees:

    EV(a) = p_recovery(a) x amount - cost(a) - p_cancel(a) x remaining_mandate_value

**Reasoning.** A better ranking could not have fixed this. The damage a
sleeping dog suffers is not a smaller recovery — it is a cancelled mandate,
which forfeits every future debit. On a recovery-probability scale that
damage is nearly invisible; on a rupee scale over a twelve-cycle horizon it
dominates. A Rs 499 subscription carries roughly Rs 6,000 of remaining value,
so a 5% cancellation risk costs more than the payment being chased is worth.

Ranking also cannot express *declining to act at all*. It orders cases; it has
no way to leave budget unspent. Expected value can return "none of these
beat inaction", which is the behaviour a sleeping dog requires.

**Result at 40,000 cases, equal contact budget (6,952 contacts):**

    policy       net Rs        cancellations   sleeping dogs contacted
    risk_topN    88,639,228              315                     1,451
    uplift_ev   130,813,491               22                       833

---

## ADR-0019 — Deferral is a distinct verdict from denial

**Date:** Phase 6
**Status:** Accepted

**Decision.** The compliance gate returns ALLOW, BLOCK or DEFER, and DEFER
carries a `defer_until` timestamp.

**Reasoning.** Two rules that both stop an action now are not the same rule.
A revoked mandate can never be debited. A message at 01:30 IST is not
forbidden, it is early. Collapsing them either sends messages at 2am or
discards recoverable cases for a reason that expires in a few hours. The
distinction also matters for the audit trail: "deferred to 09:00" and
"blocked, customer on DND" are different answers to a regulator.

---

## ADR-0020 — Idempotency keys are deterministic and reserved before the call

**Date:** Phase 7
**Status:** Accepted

**Context.** The defect this prevents: an executor retries a call it believes
failed, when the call actually succeeded and only the response was lost. The
customer is charged twice. This is caused by ordinary network behaviour, not
by anything exotic.

**Decision.** Two mechanisms, because either alone is insufficient.

1. A key derived deterministically from `(case_id, action, attempt, amount)`,
   passed to Razorpay as the order `receipt` so the provider itself rejects a
   duplicate. Deterministic, not a fresh UUID per attempt — a random key would
   make every retry look like a new charge, which is the bug rather than the
   fix. The wall clock is excluded: two attempts on the same operation hours
   apart are the same operation.
2. A local SQLite ledger recording every key issued. The provider's guarantee
   only applies once the request reaches it; the ledger catches duplicates
   before the call and lets a crashed run resume without reissuing work.

**Ordering.** The ledger row is written *before* the network call. A crash
mid-flight then leaves an `in_flight` row, which is recoverable by
reconciliation. The reverse ordering leaves no trace at all, and the operation
silently repeats on restart.

**Attempt counting** reads from the ledger rather than from case state, since
in-memory state can be stale after a crash while the ledger records what
actually reached the provider.

---

## ADR-0021 — Provider clients are record/replay, not live, in tests

**Date:** Phase 7
**Status:** Accepted

**Context.** CI has no network and no credentials. A test suite calling the
live API would either be skipped in CI, making it decorative, or would embed
secrets, making it a liability.

**Decision.** `RazorpayClient` is a protocol with three implementations:
`LiveClient` (real test-mode SDK), `RecordingClient` (wraps live, writes
fixtures), `ReplayClient` (serves fixtures, no network).

**Reasoning.** Record once, replay forever. Tests then exercise real response
shapes — including error shapes — without a network dependency. It also makes
the error paths testable at all: provoking a genuine gateway timeout on demand
is impractical, replaying a recorded one is trivial, and error handling that
has never executed is not error handling.

`LiveClient` refuses to construct on a key that does not begin `rzp_test_`.
The check is cheap; the failure it prevents is not.

---

## ADR-0022 — Unmapped provider errors are never retried

**Date:** Phase 7
**Status:** Accepted

**Decision.** Only error codes explicitly classified retriable are retried.
Anything unrecognised is treated as permanent and escalated.

**Reasoning.** The alternative — assume an unfamiliar failure is transient —
is the reasoning that produces duplicate charges. A new error code shipped by
the provider next quarter should surface as an escalation, not as a silent
retry loop against an operation that may have already succeeded. Retrying a
`BAD_REQUEST_ERROR` also cannot succeed and only delays the escalation.

---

## ADR-0023 — The LLM drafts; the system verifies

**Date:** Phase 7
**Status:** Accepted

**Context.** The RBI pre-debit notification must state merchant name, amount,
debit date and time, mandate reference, reason, and how to opt out. This is
one of the three places an LLM is used in this project.

**Decision.** A deterministic template produces a compliant message. The model
is asked only to improve phrasing, and its output is then validated against
the same required-field list from `configs/compliance/policy.yaml`. A draft
missing any required field is discarded and the template is sent instead.

**Reasoning.** A fluent notification missing its opt-out instruction is a
compliance failure that reads perfectly well — exactly the class of error a
language model produces and a human reviewer skims past. The validator does
not care how the sentence reads. The model is never load-bearing: no API key,
an API error, or a bad draft all fall back to the template silently, because
a compliant plain message beats an elegant non-compliant one and there is no
scenario where waiting for better copy justifies delaying a mandated notice.

---

## ADR-0024 — The ledger is event-sourced; case state is derived, never stored

**Date:** Phase 8
**Status:** Accepted

**Context.** The track requires an audit trail. The cheap version is a
`status` column updated as a case progresses.

**Decision.** An append-only event log. Case state is recomputed by folding
the events, and `replay()` is the only way to obtain it.

**Reasoning.** A mutable status field answers "where is this case now". It
cannot answer "what did the agent consider, what did it reject, and on what
evidence" — and those are the questions an auditor actually asks. A status
field can also be corrected after the fact leaving no trace, which makes it
worthless as evidence precisely when evidence matters.

Deriving state also makes disagreement detectable. If `replay()` and the live
system ever differ about a case, the log is right and the system has drifted.
With a stored status there is nothing to compare against.

**Enforcement.** Append-only is enforced by SQL triggers on UPDATE and DELETE,
not only by the absence of methods on `EventStore`. A future contributor
reaching for raw SQL hits a wall rather than a convention. Two tests assert
this by attempting both operations through a direct `sqlite3` connection.

**Cost accepted.** Reading a case costs a fold over its events rather than a
row lookup. At this scale that is irrelevant, and the property is worth more
than the microseconds.

---

## ADR-0025 — Reconciliation is exact equality, not a tolerance

**Date:** Phase 8
**Status:** Accepted

**Decision.** `Reconciliation.reconciles` returns `difference_paise == 0`.

**Reasoning.** The headline claim of this project is a rupee figure. If the
reported total and the ledger sum disagree by any amount, the figure is
wrong and the disagreement needs explaining rather than absorbing. A
tolerance would quietly permit exactly the drift the check exists to catch.

This is what integer paise (ADR-0003) was for. With floats the equality test
would be unsound and a tolerance unavoidable, so the two decisions are one
decision made in two places.

---

## ADR-0026 — The denial log is a first-class artifact

**Date:** Phase 8
**Status:** Accepted

**Decision.** Every compliance verdict is persisted — passes, blocks and
deferrals alike — and `recovery audit` reports blocked and deferred counts per
rule as a headline table.

**Reasoning.** Recording only the chosen action produces a trail that shows a
well-behaved agent and proves nothing, because a system with no gates would
produce the same trail. The denials are the evidence that the agent is
bounded. A run reporting zero denials is either operating in a world with no
rules or is not checking, and the CLI says so in those words rather than
printing an empty table.

Passing verdicts are kept for the same reason: an auditor asking "what else
was checked" needs the passes, not just the failure.


---

## Incidents

> Running log of things that broke, what the symptom was, what the cause
> turned out to be, and what changed as a result. Appended in real time —
> this is not reconstructed after the fact.

_(none yet)_


### INC-001 — mypy silently skipped the entire domain package

**Phase:** 1
**Symptom:** Editor reported `import-untyped` on `recovery.domain`; strict
mypy was passing because it was analyzing nothing.
**Cause:** Package shipped without a PEP 561 `py.typed` marker, so mypy
treated our own installed package as an untyped third-party library.
**Fix:** Added `src/recovery/py.typed` and declared it in
`[tool.setuptools.package-data]`.
**Changed as a result:** CI now runs `mypy src tests` rather than `mypy src`,
so type coverage of the test suite is enforced too — that is where the gap
surfaced.


### INC-002 — mypy aborted on numpy stubs before checking any project code

**Phase:** 1
**Symptom:** `mypy src tests` failed with a syntax error inside
`numpy/__init__.pyi`, reporting "errors prevented further checking". Zero
project files were analysed.
**Cause:** `[tool.mypy] python_version` was pinned to 3.11 while the venv ran
3.12. Numpy's stubs use PEP 695 `type` statements, which mypy rejects as
invalid syntax when targeting 3.11.
**Fix:** Set `python_version = "3.12"` and `requires-python = ">=3.12,<3.14"`
so the declared floor, the venv and the type-check target agree.
**Changed as a result:** Version is now declared in exactly one consistent
place. A mismatch between the runtime and the type-check target silently
disables type checking, which is worse than a loud failure.



### INC-003 — Pylance and mypy disagreed on **dict unpacking

**Phase:** 1
**Symptom:** 16 `reportArgumentType` errors in the editor on
`RecoveryCase(**base)`; mypy reported success on the same line.
**Cause:** Mypy's pydantic plugin models the generated `__init__` and does not
narrow through `**dict[str, object]`; Pylance has no such plugin and checks the
unpack strictly. Two checkers, two rulesets, one line.
**Fix:** Typed the test helper's overrides as `Any`, which is accurate — the
helper exists to accept arbitrary field overrides.
**Changed as a result:** mypy is the single source of truth for type errors
because it is what CI runs; Pylance stays in `basic` mode as an assist. Any
future disagreement gets resolved by changing the code, not by silencing one
of the two.


### INC-004 — CI skipped every quality gate after a Python version mismatch

**Phase:** 1
**Symptom:** GitHub Actions run #2 failed at `Install`. Lint, type check,
import contracts and tests all reported 0s and did not execute.
**Cause:** `requires-python` was raised to `>=3.12` (INC-002) but the CI
workflow still requested 3.11 via `setup-python`. Pip refused to install the
package, and the remaining steps never ran.
**Fix:** Pinned the runner to 3.12 and updated the type-check step to
`mypy src tests` to match local invocation.
**Changed as a result:** The Python version is now stated in two places that
must agree — `pyproject.toml` and the CI workflow. Noted as a future
consolidation candidate. This also confirmed that a failed install skips
rather than silently passes downstream gates, which is the behaviour we want.

### INC-005 — Two assumptions were invisible to the robustness sweep

**Phase:** 2
**Symptom:** `test_registry_covers_every_assumption` failed on first run,
reporting `SALARY_DAY_OF_MONTH` and `SALARY_WINDOW_DAYS` as unregistered.
**Time lost:** ~0 (caught immediately by the test that was written for it).
**Cause:** Both were added to `assumptions.py` without a corresponding entry in
`ASSUMPTION_REGISTRY`. Phase 9's sweep iterates the registry, so both would
have been silently held fixed while being reported as swept.
**Fix:** Registered both.
**Changed as a result:** Nothing structural — the guard already existed and
worked. Recording it because it demonstrates why the registry test is worth
its cost: an unswept assumption reported as swept is a false claim, and it
would have been invisible in the results.


**Follow-up:** the full 50-row file contained a *second* non-bank entity
(One Mobikwik Systems Limited) not visible in the sampled screenshot, with
the same signature as the first: 100.00% approved, zero BD, zero TD. The
exclusion rule is therefore stated as a signature rather than a name list —
a technology provider reporting perfect approval and no declines is not an
issuing bank. Patching the one row I had seen would have left the second in.

### INC-007 — Two of the four segments were structurally impossible

**Phase:** 3
**Symptom:** `sure_thing` prevalence was exactly 0% across 6,000 generated
cases. The uplift thesis rests on distinguishing four segments; with one
absent, a third of the argument was untestable.

**Time lost:** ~40 minutes across three distinct causes.

**Cause:** three separate problems wearing one symptom.

1. **Arithmetic ceiling.** `p(NO_ACTION) = intent * PASSIVE_RECOVERY[reason]`,
   and the largest passive value was 0.55, while `CERTAINTY_THRESHOLD` was
   0.65. No case could clear the bar under any parameter draw. Confirmed by
   computing the maximum directly rather than by inspecting samples.
2. **Escalation swamped the classifier.** With the ceiling raised, still zero:
   `ESCALATE_HUMAN` lifts nearly every case above the persuasion threshold, so
   high-baseline cases were classified persuadable rather than sure things.
3. **The population was wrong.** With both fixed, still zero — and this one
   was not a bug. In a *failed payment* population some action always beats
   doing nothing, so sure things cannot exist there. They exist only before
   the failure, among mandates that will succeed untouched.
**Fix:**
- Raised passive recovery for transient technical failures (bank downtime
  0.55 → 0.74, network 0.52 → 0.71). The original values implied that most
  customers hitting a transient error never retry unprompted, which is
  wrong and flattered every intervention by removing the self-healing
  population entirely.
- Excluded `ESCALATE_HUMAN` from segmentation (ADR-0010).
- Added the `UPCOMING_AT_RISK` population: mandates due to debit but not yet
  failed, at 30% of the batch.
Result: sure things 4.2%, lost causes 16.5%, persuadables 59.6%, sleeping
dogs 14.6%.

**Changed as a result:** The third cause was the valuable one, and it was not
a defect. It is a real structural property — sure things are a pre-failure
phenomenon — and it is now asserted by
`test_sure_things_are_pre_failure_only`, which will fail if a future change
lets them appear in the recovery population.

It also strengthens the compliance-as-channel argument rather than
weakening it. The RBI pre-debit notification fires on every recurring debit,
including the ~89% that would have succeeded anyway. That is precisely a
population of sure things, and it is exactly why targeting matters: a policy
that cannot separate them from the at-risk minority spends its entire contact
budget on customers who needed nothing.

**Process note:** the first cause was found by computing the maximum
achievable value analytically rather than by staring at generated samples.
Worth repeating whenever a category has zero prevalence — check whether it is
reachable at all before assuming the sampler is at fault.

### INC-008 — A type fix was reported clean by a checker that never ran

**Phase:** 3
**Symptom:** mypy flagged `PaymentMethod(rng.choice([...]))` as passing a
numpy scalar where a `str` was expected. A fix was applied and reported
clean, but the identical error reappeared on the next run.

**Time lost:** ~15 minutes, plus one wasted round trip.

**Cause:** two failures stacked, and either alone would have been caught.

1. The edit was a string replacement written against post-format text and
   applied to a file that had not yet been formatted. The match failed
   silently — the new `ONE_OFF_METHODS` constant was added, the call site was
   left untouched. A failed replacement raises nothing.
2. The verifying environment lacked numpy's type stubs, so mypy inferred
   `Any` for the `rng.choice` return and reported no error at all. The green
   result meant "not checked", not "correct".
**Fix:** Index a module-level `ONE_OFF_METHODS` tuple with `rng.integers`.
Verified by reading the changed line back from disk rather than by observing
a later command exit zero.

**Changed as a result:**
- Edits are confirmed by re-reading the file, not by a subsequent command
  succeeding. A silent no-op patch and a correct patch look identical from
  the outside.
- A checker's silence counts as evidence only when the checker can see the
  types involved. A stub-less mypy run is indistinguishable from a passing
  one, which makes it worse than no check.
**Note on the frozen world:** batch output was byte-identical after the
change, because `Generator.choice` delegates to `integers` when no `p` is
supplied, so the random stream position was preserved. This was verified by
regenerating and comparing, not assumed. Any change to how the generator
draws randomness can otherwise shift every reported number with no visible
error — which is a large part of what `v0.4-world-frozen` exists to protect.

### INC-009 — A test asserted a property of the sample, not of the code

**Phase:** 4
**Symptom:** `test_cost_weighting_changes_the_chosen_threshold` passed at
n=8000 and failed at n=6000.

**Cause:** The test compared the F1-optimal threshold against the
cost-optimal one and asserted they differ. That comparison depends on where
the sampled scores happen to fall, not on whether the cost function weights
the two errors asymmetrically. At 8000 cases F1 chose 0.90 and cost chose
0.80; at 6000 both chose 0.80. Nothing was wrong with the code — the
assertion was about the batch.

**Fix:** Replaced with two tests that assert the actual properties.
`test_cost_penalises_missed_degradation_more_than_false_alarms` constructs
two scores with identical F1 and opposite error splits and asserts the
false-negative-heavy one costs more — deterministic, no sampling involved.
`test_cost_optimum_is_interior` checks the chosen threshold is not at either
endpoint, which would indicate the threshold is not doing real work.

**Changed as a result:** A test whose outcome depends on sample size is a
measurement, not an assertion. Where a property is genuinely about the
objective function, it should be tested against constructed inputs rather
than fitted ones — otherwise a future batch-size change produces a red build
with no defect behind it.

### INC-010 — A dependency-file overwrite silently reverted a prior fix

**Phase:** 4
**Symptom:** `mypy src tests` failed with `Type statement is only supported in
Python 3.12 and greater` inside numpy's stubs — the identical failure as
INC-002, already fixed in Phase 1.

**Cause:** The Phase 4 file set included `pyproject.toml`, because the import
contract needed tightening (ADR-0011). That file also carries
`python_version` and `requires-python`, both corrected locally during Phase 1.
The overwrite delivered the intended change and silently reverted two
unrelated ones.

**Fix:** Restored `python_version = "3.12"` and
`requires-python = ">=3.12,<3.14"`. Added `scipy` as an explicit dependency
(it was previously only transitive via scikit-learn, while
`diagnose/issuer_health.py` imports it directly) and to the
`ignore_missing_imports` overrides.

**Changed as a result:** Configuration files that accumulate local
corrections are not safe to overwrite wholesale, even when a change to them
is genuinely required. Structural changes to `pyproject.toml` are applied as
targeted edits from here on. Running the four gates immediately after any
config change is what caught this, and that ordering is now deliberate rather
than incidental.

### INC-011 — Fourteen type suppressions, all misplaced, none needed

**Phase:** 4
**Symptom:** `mypy src tests` reported 27 errors in `test_diagnose.py` — one
`no-untyped-def` and one `unused-ignore` per test — followed by a fourteenth
on a separate line after those were fixed.

**Cause:** two related failures.

1. Each test signature was wrapped across lines with the
   `# type: ignore[no-untyped-def]` comment on the parameter line. mypy
   attributes that error to the `def` line, so every suppression sat one line
   below the error it was meant to silence. The error fired *and* the comment
   was flagged as dead — two errors from one bad habit.
2. A fourteenth ignore on `CaseFeatures(**base)` was dead for the same reason
   as INC-003: mypy's pydantic plugin does not narrow through
   `**dict[str, object]`, so there was no error to suppress.
**Fix:** Defined `Batch = tuple[ObservableBatch, OracleBatch]` and annotated
every fixture parameter. Typed the `_features` helper's overrides as
`dict[str, Any]`, which is accurate — the helper exists to accept arbitrary
field overrides — and keeps Pylance and mypy in agreement. All fourteen
suppressions removed; none were replaced.

**Changed as a result:**
- A suppression is a last resort, not a first response. This is the third
  incident in the same family (INC-003, INC-008, INC-011) and in every case
  the type was expressible and the annotation was shorter than the workaround.
- `# type: ignore` binds to a specific line, so any formatter that rewraps a
  signature silently detaches it from the error it was suppressing. That makes
  a wrapped signature carrying an ignore comment inherently fragile.
- Test helpers accepting arbitrary overrides use `dict[str, Any]` by default.

**Fourth occurrence (Phase 5):** `learners.py` passed hyperparameters as
`dict[str, object]` into LightGBM constructors — 12 errors from one
annotation. Also three `no-any-return` on NumPy operator results, which
degrade to `Any` under strict mode and need an explicit dtype at the return
boundary, and one variance error where `Counter[Segment]` was assigned to
`dict[Segment, int]` (dict is invariant in its value type; `Mapping` is not).

**Standing rule adopted:** `dict[str, Any]` for any mapping destined for
`**kwargs`. `object` is never right there — it satisfies no concrete
parameter type, so it produces one error per keyword the callee declares.

**Fifth occurrence (Phase 6):** `economics.py` and three helpers in
`test_compliance.py`, one phase after the standing rule was written. The
rule was correct and was simply not applied.

**Root cause of the recurrence:** the authoring environment cannot run the
pydantic mypy plugin, so `dict[str, object]` passes there and fails
downstream. A rule that depends on remembering it, in an environment that
cannot enforce it, will keep being broken. The durable fix is that the
receiving environment runs the gates before anything is committed — which is
what has caught it every time.


### INC-012 — Verified the import contract by deliberately breaking it

**Phase:** 4
**Status:** Not a defect. Recorded because the result is the project's central
integrity claim and it should be demonstrable, not merely asserted.

**Action:** Appended `from recovery.world.timeline import IssuerTimeline` to
`diagnose/taxonomy.py` and ran `lint-imports`.

**Result:** Build failed, naming the contract, the module pair and the line
numbers:

    Decision-making code cannot import the world BROKEN
    recovery.diagnose.taxonomy -> recovery.world.timeline (l.120, l.122)

`world.timeline` exposes `is_degraded()` — precisely the fact the diagnosis
layer exists to infer. Had this import been possible, the detector could have
scored perfectly and the Phase 4 results would have been meaningless. CI
rejects it.

**Secondary finding:** `git checkout <path>` failed to revert the change,
because `taxonomy.py` was newly added and not yet staged, so git had no
version to restore. The file had to be repaired by hand.

**Changed as a result:** Stage work before deliberately breaking something.
An untracked file has no safety net, and the moment of wanting to undo an
experiment is the worst moment to discover that.


### INC-013 — A null treatment was labelled as treatment

**Phase:** 5
**Symptom:** Both uplift learners scored *worse than random* on Qini. Oracle
ranking peaked at 84% depth, when a ranking with access to true effects should
concentrate them early.

**Cause:** Treatment pooled `SEND_PAYMENT_LINK` and `PRE_DEBIT_NUDGE`. On
one-off payment failures the nudge is a no-op by construction — a pre-debit
notification cannot precede a debit that already happened — so its potential
outcome equals the do-nothing outcome. Measured directly: 8,164 payment-failure
cases, mean nudge uplift +0.000000, zero non-zero values. Roughly half the
treated group had received no treatment at all.

**Fix:** Treatment restricted to `SEND_PAYMENT_LINK` (ADR-0015).

**Changed as a result:** When results are worse than random, the first
hypothesis is a labelling error, not a modelling one. Checking the true effect
of each pooled action on each population took one query and would have caught
this before any model was fitted.

### INC-014 — The control arm was starved and uplift inherited its noise

**Phase:** 5
**Symptom:** After fixing INC-013, risk ranking (Qini 0.0987) still beat the
X-learner (0.0238), contradicting the project's central claim.

**Cause:** Arm imbalance in training: 4,390 treated against 801 control.
`NO_ACTION` is reachable in logged data only through 15% epsilon exploration
spread over six actions, so the naive policy supplies almost no control data.
Since uplift is `mu1 - mu0`, the estimate was dominated by the sparse arm's
error.

**Fix:** Split the randomised holdout, half to training (ADR-0016). Control
rose to 2,166 and the ordering reversed: X-learner 0.2010, T-learner 0.1740,
risk ranking -0.0101, random -0.0204.

**Changed as a result:** Arm balance is now printed by the `uplift` command
alongside the results, because a starved control arm is invisible in the
metrics and produces a plausible-looking wrong answer rather than an error.
The episode also confirmed the X-learner's reason for existing: it beats the
T-learner precisely under the imbalance that caused this.

### INC-015 — A test compared curve minima dominated by single observations

**Phase:** 5
**Symptom:** `test_risk_ranking_destroys_more_value` passed at 60,000 cases
and failed at 40,000.

**Cause:** It compared `QiniCurve.min_value`, the global minimum. At 1% depth
a 2,000-case holdout has roughly twenty observations split across two arms, so
the corrected difference swings on single events. The assertion was about
early-depth sampling noise, not about either ranking.

**Fix:** Added `WARMUP_DEPTH = 0.05` and `min_value_after_warmup`, and pointed
`destroys_value` at it. The test now compares `uplift_at_30` — recovery at a
realistic contact budget — which is both the operational question and stable.

**Changed as a result:** Second occurrence of this pattern after INC-009.
A claim about a curve must be made at a depth where the curve is supported by
enough observations to mean anything, and the metric now enforces that rather
than leaving it to whoever writes the assertion.

---

## ADR-0017 — Uplift results are reported across independent worlds, never one

**Date:** Phase 5
**Status:** Accepted

**Context.** The Qini coefficient on a single evaluation holdout has a
standard deviation comparable to the effect being measured. At 12,000 cases
the unbiased holdout is roughly 400 rows, and the estimate ranges from -5.6
to +27 across configurations that differ only by sample.

**Decision.** The uplift command generates N independent worlds (default 5 at
90,000 cases), reports Qini as mean +/- sd, and states how many worlds each
ranking beat risk ranking in.

**Reasoning.** A point estimate here is indistinguishable from noise, and
reporting one would have meant publishing whichever number the first seed
happened to produce. "Beats risk in 5/5 worlds" is a claim a reviewer can
interrogate; "Qini 0.261" is not.

**Cost accepted.** The command takes several minutes rather than seconds.

---


### INC-016 — An apparent negative result was sampling noise

**Phase:** 5
**Symptom:** Both uplift learners scored *negative* Qini — worse than random —
on the first run at the default 12,000 cases:

    uplift_x_learner   -2.74
    uplift_t_learner   -5.57
    risk_ranking       +2.08
    random             +4.78

Read at face value this says the entire uplift thesis is wrong.

**Cause:** insufficient data, in a place that was easy to miss. Of 12,000
generated cases only ~2,300 are contactable (logged action in {NO_ACTION,
SEND_PAYMENT_LINK}); after the train/eval split the unbiased holdout was ~400
rows. A Qini coefficient estimated on 400 rows with a control arm of a few
hundred has a standard error larger than the effect. The learners were not
failing — the measurement was.

**Diagnosis:** Rather than adjusting the model, swept sample size first:

    n_cases   x_learner   t_learner   risk    random   oracle
     12,000      -2.743      -5.570   2.084    4.784   27.067
     40,000      +0.289      -0.050  -0.005   -0.236    0.550
     90,000      +0.173      +0.098   0.062   -0.121    0.435
    160,000      +0.175      +0.179   0.166   -0.021    0.827

The wild values at 12,000 and their collapse toward a stable ordering as n
grows is the signature of a variance problem, not a bias one.

**Confirmation:** five independent worlds at 90,000 cases:

    oracle_uplift      0.772 +/- 0.248
    uplift_x_learner   0.261 +/- 0.096   beats risk 5/5
    uplift_t_learner   0.225 +/- 0.123   beats risk 5/5
    risk_ranking       0.118 +/- 0.132
    random             0.024 +/- 0.065

**Fix:** Raised the default to 90,000 cases and made the command report mean
+/- sd across seeds with a win count (ADR-0017).

**Changed as a result:**
- A result whose sign flips with the seed is not a result. Effect estimates
  are reported with dispersion, or not reported.
- When a measurement contradicts a strong prior, sweep the measurement before
  touching the model. The instinct to tune the learner here would have been
  wrong twice: it would not have helped, and any improvement would have been
  fitted to noise.
- The cost of getting this wrong in the other direction is worse. Had the
  first seed produced +0.4 instead of -2.7, the result would have looked like
  a success and shipped unexamined.
### INC-017 — Uplift targeting does not avoid sleeping dogs

**Phase:** 5
**Status:** Open limitation, not yet fixed. Recorded so it is not quietly
omitted from the write-up.

**Observation:** In the top 30% selected by each ranking:

    segment        uplift    risk
    persuadable    77%       40%
    lost_cause      7%       52%
    sleeping_dog    9%        3%

Uplift targeting does what it was built for on the first two rows: it nearly
doubles persuadable capture and cuts lost causes from 52% to 7%. Risk ranking
spends over half its budget on customers who will never pay, which is the
direct consequence of ranking by probability of failure rather than by effect
of treatment.

But it selects *more* sleeping dogs than risk ranking does, not fewer.

**Interpretation:** The learner is finding the positive tail of the CATE
distribution and is largely blind to the negative one. Sleeping dogs are
customers with high baseline recovery and high contact sensitivity, and
contact sensitivity has only a weak observable proxy (`contacts_last_30d`),
so the negative-uplift population is close to indistinguishable from
persuadables in feature space.

**Not yet addressed.** Candidate directions for Phase 6: penalise predicted
negative uplift asymmetrically in the decision engine rather than relying on
the ranking; treat mandate cancellation as an explicit cost term rather than
folding it into recovery probability.

**Why recorded now:** the persuadable and lost-cause numbers are the strongest
result in the project so far, and presenting them without this row would be
selective reporting. The honest claim is that uplift targeting solves the
lost-cause problem and does not yet solve the sleeping-dog problem.

### INC-018 — Duplicate identifiers from an interrupted session

**Phase:** 5
**Status:** Resolved by renumbering. Recorded because the failure mode is
structural, not careless.

**Symptom:** `DECISIONS.md` contained two `## ADR-0014` headings, two
`### INC-013`, two `### INC-014` and two `## Incidents` sections.

**Cause:** Phase 5 was built across two working sessions. The second session
began without the first session's entries in view, and allocated the next
identifiers from the last numbers it could see — which were Phase 4's. Both
blocks were internally consistent; the collision was only visible when the
file was read end to end.

**Fix:** Renumbered the second block to ADR-0017, INC-016 and INC-017,
updated the cross-references, and removed the duplicated `## Incidents`
header.

**Changed as a result:** Before appending to the decision log, grep the
existing identifiers rather than assuming the next number follows the last
one written in the current session:

    grep -n "^## ADR-\|^### INC-" DECISIONS.md | tail -5

**Why this matters beyond tidiness:** the log is a primary artifact for this
submission. Two ADR-0014s tell a reader the record is not maintained
carefully, which undermines every claim the log is supposed to support —
including the evaluation-integrity ones that have no other evidence behind
them.


### INC-019 — An uplift model was the wrong estimator for cancellation risk

**Phase:** 6
**Symptom:** The cancellation model silently declined to fit. `p_cancel` was
zero for every case, so the entire sleeping-dog correction was inert. The
policy still ran and still produced plausible numbers — net Rs 127.8M against
Rs 128.8M for doing nothing, i.e. *worse than inaction* — which is the only
reason the failure was noticed.

**Cause:** Cancellation was framed as an uplift problem, mirroring recovery.
But a customer who is never contacted never cancels *because of* contact, so
the control arm carries structurally zero events:

    usable 9,998   treated 7,835   control 2,163
    cancellations in treated arm: 198
    cancellations in control arm: 0

An uplift learner differences two arms. With one arm identically zero there
is nothing to difference, and the guard against fitting a single-class arm
correctly refused — silently, as designed.

**Fix:** Replaced the uplift learner with a plain classifier on the treated
arm. Where the control outcome is structurally zero,
`P(cancel | contacted, x)` *is* the uplift, and estimating it directly uses
all 198 events instead of discarding them.

**Changed as a result:**
- Symmetry of framing is not a reason to reuse an estimator. Recovery and
  cancellation look like the same shape of problem and are not: one has a
  genuine control arm, the other cannot have one.
- The near-miss is the more important part. A silently inert correction
  produced a policy that lost money against doing nothing, and it would have
  passed unnoticed had the `do_nothing` baseline been absent from the
  comparison. Every policy evaluation now carries `do_nothing` as a
  mandatory row, because a recovery policy that cannot beat inaction is the
  failure mode least likely to look like one.

### INC-020 — The decision rationale named the action, not the rule

**Phase:** 6
**Symptom:** `test_rationale_names_the_blocking_rule` failed. The rationale
read `send_payment_link scored above inaction but was gated
(send_payment_link)`.

**Cause:** The explanation collected the set of gated *actions* rather than
the rule ids that gated them.

**Fix:** Rationale now names the blocking or deferring rule, and distinguishes
the two: `blocked by DND_REGISTRY` versus `deferred by CONTACT_HOURS`.

**Changed as a result:** An audit trail that records what was blocked but not
why is decoration. The test was written to assert the rule id appears, and
it caught this on first run — worth noting because the original wording read
perfectly well in English while conveying nothing an auditor could act on.


### INC-021 — A validator silently ignored requirements it did not recognise

**Phase:** 7
**Symptom:** `test_compose_raises_when_template_diverges_from_policy` failed:
adding `customer_grievance_contact` to the required-field list produced no
error, and the notification was reported compliant.

**Cause:** `validate()` was a chain of `elif field == "..."` branches. A field
name matching none of them fell through the chain and was never appended to
`missing`. The function therefore checked only the fields it already knew
about and treated everything else as satisfied.

**Why it matters more than it looks:** the required-field list lives in
`configs/compliance/policy.yaml` precisely so a regulatory change can be made
as a config edit (ADR-0004). With this bug, adding a field to that file would
appear to tighten the rules while changing nothing — an unenforced
requirement that looks enforced, which is worse than no requirement at all
because it removes the reason to look.

**Fix:** Rewrote as an explicit `dict[str, bool]` of checks. A field absent
from that dict is reported missing. The validator now fails closed.

**Changed as a result:** Validation driven by external configuration must fail
closed on unrecognised entries. The test that caught this was written to
assert the loud-failure behaviour rather than the happy path, which is the
only reason it surfaced — a test asserting that a correct template passes
would have gone green throughout.


### INC-022 — A duplicate-reference rejection was classified as a failure

**Phase:** 7
**Found by:** a live test-mode call against a real Razorpay account. Not by
the test suite, which was passing 166 tests at the time.

**Symptom:** Re-creating a payment link with an existing `reference_id`
raised `razorpay.errors.BadRequestError`. The error map classified it as
`FailureReason.OTHER`, non-retriable, and the executor would have recorded
the case as failed and escalated it to a human.

**Cause:** "Duplicate reference" is provider-side idempotency working
correctly — it means an earlier request with the same reference already
succeeded. Razorpay returns a generic `BAD_REQUEST_ERROR` code for it, which
at the code level is indistinguishable from a genuinely malformed request.
Treating the two alike conflates "you already did this" with "your request was
wrong": opposite meanings, opposite correct responses.

The consequence would have been mild in money terms and severe in trust
terms. No double charge — the provider prevented that, which was the point.
But every successfully-deduplicated operation would have surfaced as a failed
case in the audit trail, so the artifact meant to prove the system is bounded
would have been reporting phantom failures.

**Fix:**
- Added `DuplicateOperationError`, detected on the provider's description
  rather than its code, since the code is not specific enough.
- The existing reference is parsed out of the description, so a duplicate
  tells us *which* earlier operation it collides with rather than merely that
  one exists.
- The executor treats it as a successful, already-completed operation and
  records the recovered reference. The reason is surfaced explicitly:
  succeeding because of an earlier attempt is materially different from
  succeeding because of this one, and the trail must say which.

**Changed as a result:**
- Seven regression tests added, using the exact response text the live account
  returned rather than a shape I guessed at.
- **The general lesson:** no unit test would have found this. The condition
  only exists at the boundary with a real system holding real prior state,
  and it took a second run against the same account to produce it. A test
  suite that has never touched the provider is testing our model of the
  provider, not the provider. Recording live fixtures (ADR-0021) is what
  converts a finding like this into permanent coverage.
