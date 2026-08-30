import json
import sys
import time
from pathlib import Path

from recovery.calibration.models import WorldParameters
from recovery.compliance.engine import ComplianceEngine, load_policy
from recovery.diagnose.engine import DiagnosisEngine
from recovery.domain.enums import ActionType
from recovery.evaluate.policy_eval import blind_retry_actions, evaluate_policies, top_n_actions
from recovery.evaluate.sweep import default_grid, perturb, sample_grid
from recovery.policy.economics import Economics
from recovery.policy.runner import fit_policy, mandate_values, run_policy
from recovery.uplift.features import FeatureEncoder
from recovery.world.generate import generate

lo, hi, n_cases = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
pol = load_policy(Path("configs/compliance/policy.yaml"))
base = WorldParameters.model_validate_json(Path("configs/generator/world_params.json").read_text())
grid = sample_grid(default_grid(), 16, seed=0)
out = []
for i in range(lo, min(hi, len(grid))):
    t0 = time.time()
    point = grid[i]
    econ = Economics(
        costs_paise={ActionType(k): v for k, v in pol.costs_paise.items()},
        horizon_cycles=point.cancellation_horizon,
    )
    obs, orc = generate(perturb(base, point), n_cases=n_cases, seed=100 + i)
    enc = FeatureEncoder.fit(obs.features)
    fitted = fit_policy(obs.features, obs.logged, obs.realized, enc)
    x = enc.transform(obs.features)
    dec = run_policy(
        fitted,
        obs.features,
        x,
        compliance=ComplianceEngine(pol),
        economics=econ,
        diagnosis=DiagnosisEngine.fit(obs.features),
    )
    ev = [d.decision.action.action_type for d in dec]
    amounts = [f.amount_paise for f in obs.features]
    mv = mandate_values(obs.features, econ)
    risk = fitted.recovery.predict_baseline(x)
    budget = max(
        1, sum(1 for a in ev if a in (ActionType.SEND_PAYMENT_LINK, ActionType.PRE_DEBIT_NUDGE))
    )
    pols = {
        "do_nothing": [ActionType.NO_ACTION] * len(ev),
        "blind_retry": blind_retry_actions(
            [f.case_type.value for f in obs.features],
            [f.consecutive_mandate_failures for f in obs.features],
        ),
        "risk_topN": top_n_actions(list(risk), budget),
        "uplift_ev": ev,
    }
    r = evaluate_policies(pols, orc.outcomes, amounts, mv, econ.costs_paise)
    out.append(
        {
            "i": i,
            "label": point.label(),
            "horizon": point.cancellation_horizon,
            "cancelled_share": point.cancelled_share,
            "net": {k: v.net_paise for k, v in r.items()},
        }
    )
    print(
        f"[{i}] {time.time() - t0:5.1f}s  {point.label()}  "
        f"vs_risk=Rs {(r['uplift_ev'].net_paise - r['risk_topN'].net_paise) / 100:>11,.0f}",
        flush=True,
    )
Path(f"data/generated/sweep_{lo}.json").write_text(json.dumps(out))
