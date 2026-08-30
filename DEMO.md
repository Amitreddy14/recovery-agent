# Demo runbook

Five minutes. The temptation is to explain the architecture; resist it. Show
three numbers and one refusal, and let the repo carry the rest.

## Before recording

```bash
docker compose up -d               # confirm it serves on :8000
python scripts/compare_policies.py > /tmp/policy_table.txt
python -m recovery.cli audit --case <a-blocked-case> > /tmp/trail.txt
```

Have open: the console, a terminal, `DECISIONS.md`, and the Razorpay test-mode
dashboard showing a real payment link.

Screenshot the console before recording. Do not rebuild mid-edit.

---

## 0:00 — 0:40 · The problem, in one number

> "Most merchants recover failed payments by retrying everything and messaging
> whoever fails twice. Here is what that costs."

Show the policy table. Point at two rows only:

```
do nothing    net  Rs 12,88,02,859
blind retry   net  Rs 10,60,29,698     323 mandates cancelled
```

> "Blind retry recovers ₹2.28 crore more than doing nothing — and ends up
> ₹2.28 crore worse off, because it cancelled 323 mandates. Every recovery
> system I looked at optimises the first number. None of them measures the
> second."

Do not explain the method yet. Let the number sit.

## 0:40 — 1:40 · Why risk ranking is the wrong objective

> "The obvious fix is to target better — score customers by failure risk and
> contact the riskiest. That is worse, and it is worse for a structural
> reason."

Show the segment table:

```
                uplift    risk
persuadable       77%      40%
lost cause         7%      52%
```

> "Risk ranking spends over half its budget on customers who will never pay.
> That is not a bug in the model — ranking by probability of failure selects
> exactly the people whose failure is least changeable.
>
> What you want is the people whose *outcome changes* if you act. That is a
> different quantity, and it needs a different model."

Thirty seconds on uplift: two models, difference the arms, rank by effect not
outcome. No more.

## 1:40 — 2:40 · The refusal

This is the segment that separates the submission. Go to the console, filter
to **blocked**, click a gated case.

```
case 002608   bank_downtime → no_action
blocked: retry_now, retry_scheduled, retry_alternate_rail
         all RBI_AFA_THRESHOLD
"issuer degraded; retry_now scored above inaction,
 blocked by RBI_AFA_THRESHOLD"
```

> "The agent diagnosed a degraded issuer, scored a retry as worth doing, and
> then refused — the debit exceeds the AFA-free ceiling under RBI's 2026
> e-mandate framework, so it cannot be attempted without authentication.
>
> That refusal is logged with the rule that caused it. Across this run there
> are 4,212 of them. A recovery agent that never reports a denial is either
> operating in a world with no rules, or is not checking."

Show the gate activity panel. Note that contact-hours **deferred** 23,386
actions rather than blocking them — a message at 2am is early, not forbidden,
and conflating those would either spam customers or discard recoverable cases.

## 2:40 — 3:20 · It touches real money

Switch to the Razorpay dashboard, test mode. Show a payment link the agent
created. Open the checkout page.

> "Execution is real. Test-mode calls, real idempotency keys, real error
> taxonomy. The key is deterministic in case, action, attempt and amount, and
> it goes to Razorpay as the order receipt — so the provider rejects a
> duplicate even if our own ledger were bypassed.
>
> I found that out by accident. Re-running the demo hit Razorpay's duplicate
> rejection, and my error handler had classified it as a failure. It is not a
> failure — it means the operation already succeeded. 166 passing tests did
> not catch it, because it only exists at the boundary with a real account
> holding real prior state."

## 3:20 — 4:20 · What broke

Open `DECISIONS.md`. Two incidents, thirty seconds each.

**INC-016** — the first uplift run returned negative Qini. Both learners worse
than random.

> "Read literally, that refutes the entire thesis. It was variance — the
> unbiased holdout was 400 rows and the standard error exceeded the effect.
> The fix was more data and reporting across five worlds instead of one.
>
> The part that worries me is the asymmetry. Had that first seed produced a
> *good* number, I would have shipped it without checking."

**INC-025** — a baseline reimplemented at a second call site.

> "`rank_by_risk` returns one minus the recovery probability. In two places I
> passed the probability directly, which inverts it into a sure-thing picker.
> That made the baseline look weak and my policy look three times better than
> it is. No test caught it — I found it because a number on screen disagreed
> with a number from an earlier phase."

## 4:20 — 5:00 · What is not finished

> "The sweep says the policy beats risk ranking in all sixteen assumption
> configurations I tested. It also says that in four of those sixteen, doing
> nothing would have beaten my policy. I know the mechanism — the cancellation
> horizon makes it conservative for the wrong reason — and I have two
> candidate fixes I did not have time to build.
>
> I could have picked the configuration where it wins everywhere and reported
> that. The sweep exists so I could not."

Close on the repo:

> "One command runs it. Two CI contracts prove the evaluation cannot see the
> ground truth, and both have caught real violations. Every number here has a
> script that regenerates it."

---

## Things to cut if over time

Cut in this order:
1. The uplift method explanation (the segment table makes the point alone)
2. The Razorpay dashboard (the trail already proves execution)
3. INC-025 (keep INC-016 — the asymmetry point is stronger)

Never cut: the two-row policy table, the refusal, or the unfinished section.

## Things not to do

- Do not walk through the architecture diagram. Nobody watches a box diagram.
- Do not show passing tests. 240 green dots prove nothing on video.
- Do not apologise for the synthetic data. State the boundary once and move
  on — the off-policy validation is a *reason* to build a simulated world,
  not an excuse for one.
