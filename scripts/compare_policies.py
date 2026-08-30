"""Reproduces the policy comparison table in ADR-0018.

Runs uplift_ev against do_nothing, blind_retry, risk_topN and the oracle
ceiling on identical cases with the same seed, so the comparison is paired
rather than four separate measurements.

    python scripts/compare_policies.py [n_cases] [seed]

Defaults to 40,000 cases at seed 42, which takes about 40 seconds.
"""


import sys
from pathlib import Path

from recovery.calibration.models import WorldParameters
from recovery.compliance.engine import ComplianceEngine, load_policy
from recovery.diagnose.engine import DiagnosisEngine
from recovery.domain.enums import ActionType
from recovery.evaluate.policy_eval import (
    blind_retry_actions,
    evaluate_policies,
    oracle_actions,
    top_n_actions,
)
from recovery.policy.decision import CANDIDATE_ACTIONS
from recovery.policy.economics import Economics
from recovery.policy.runner import fit_policy, mandate_values, run_policy
from recovery.uplift.features import FeatureEncoder
from recovery.uplift.targeting import rank_by_risk
from recovery.world.generate import generate
from recovery.world.oracle.segments import Segment

pol = load_policy(Path("configs/compliance/policy.yaml"))
econ = Economics.from_policy(pol.costs_paise, pol.sleeping_dog_penalty)
params = WorldParameters.model_validate_json(
    Path("configs/generator/world_params.json").read_text()
)
n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

obs, orc = generate(params, n_cases=n_cases, seed=seed)
# obs, orc = generate(params, n_cases=40000, seed=42)
enc = FeatureEncoder.fit(obs.features)
fitted = fit_policy(obs.features, obs.logged, obs.realized, enc)
if fitted.cancellation is None:
    raise SystemExit(
        "Cancellation model did not fit. The sleeping-dog correction is "
        "inert and these numbers are not comparable to ADR-0018 (INC-019)."
    )

x_all = enc.transform(obs.features)
decisions = run_policy(
    fitted,
    obs.features,
    x_all,
    compliance=ComplianceEngine(pol),
    economics=econ,
    diagnosis=DiagnosisEngine.fit(obs.features),
)

amounts = [f.amount_paise for f in obs.features]
mvals = mandate_values(obs.features, econ)
ev = [d.decision.action.action_type for d in decisions]
risk = rank_by_risk(fitted.recovery, x_all)
budget = max(
    1, sum(1 for a in ev if a in (ActionType.SEND_PAYMENT_LINK, ActionType.PRE_DEBIT_NUDGE))
)

policies = {
    "do_nothing": [ActionType.NO_ACTION] * len(obs.features),
    "blind_retry": blind_retry_actions(
        [f.case_type.value for f in obs.features],
        [f.consecutive_mandate_failures for f in obs.features],
    ),
    "risk_topN": top_n_actions(list(risk), budget),
    "uplift_ev": ev,
    "oracle": oracle_actions(orc.outcomes, amounts, mvals, econ.costs_paise, CANDIDATE_ACTIONS),
}
res = evaluate_policies(policies, orc.outcomes, amounts, mvals, econ.costs_paise)
o = res["oracle"]
print(
    f"\n{'policy':12s} {'recovered':>12} {'cancel loss':>12} {'net Rs':>12} "
    f"{'%oracle':>8} {'contacts':>8} {'cancels':>8} {'dogs':>5}"
)
for n, r in res.items():
    print(
        f"{n:12s} {r.rupees(r.recovered_paise):12,.0f} "
        f"{r.rupees(r.cancellation_loss_paise):12,.0f} "
        f"{r.rupees(r.net_paise):12,.0f} {r.net_paise / o.net_paise:8.0%} "
        f"{r.contacts:8d} {r.mandates_cancelled:8d} "
        f"{r.segment_contacted.get(Segment.SLEEPING_DOG, 0):5d}"
    )
