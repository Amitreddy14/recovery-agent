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


## ADR-0002 — Ground-truth outcomes are quarantined by CI, not by convention

**Date:** Phase 0
**Status:** Superseded by ADR-0011 (Phase 4). The contract described here
forbade only `recovery.world.oracle`, which left `world.timeline` and
`world.latent` reachable from decision code. Retained for the record.

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
