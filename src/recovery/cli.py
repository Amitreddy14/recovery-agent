"""Command-line entrypoint.

Calibration is a command, not a notebook: the same inputs must produce the
same `configs/generator/world_params.json` on any machine, and the file must
record what produced it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from recovery.calibration.calibrate import calibrate, degraded_time_fraction
from recovery.calibration.npci import load_snapshot

app = typer.Typer(add_completion=False, help="Recovery agent operations.")
console = Console()

DEFAULT_OUTPUT = Path("configs/generator/world_params.json")


@app.callback()
def main() -> None:
    """Recovery agent CLI.

    Present so typer keeps subcommand names stable: without a callback a
    single-command app collapses to a bare invocation, and every documented
    `recovery calibrate ...` line would break the moment a second command
    landed.
    """


@app.command("calibrate")
def calibrate_command(
    snapshot: Annotated[Path, typer.Option("--snapshot", help="Path to the NPCI CSV.")],
    provenance: Annotated[
        Path | None,
        typer.Option("--provenance", help="Defaults to <snapshot>.provenance.yaml"),
    ] = None,
    output: Annotated[Path, typer.Option("--output")] = DEFAULT_OUTPUT,
    allow_fixture: Annotated[
        bool,
        typer.Option(
            "--allow-fixture",
            help="Permit a synthetic fixture as input. Never for reported results.",
        ),
    ] = False,
) -> None:
    """Derive frozen world parameters from a published NPCI snapshot."""
    if "fixture" in snapshot.name and not allow_fixture:
        console.print(
            f"[red]Refusing to calibrate from {snapshot.name}[/red]: this is a "
            "synthetic fixture, not published evidence.\n"
            "Pass --allow-fixture only for smoke tests. See "
            "data/external/npci/README.md to obtain real data."
        )
        raise typer.Exit(code=2)

    provenance_path = provenance or snapshot.with_suffix(".provenance.yaml")
    snap = load_snapshot(snapshot, provenance_path)
    params = calibrate(snap)

    fraction = degraded_time_fraction(
        params.issuers[0].degradation_episode_rate_per_day,
        (0.5, 6.0),
    )

    table = Table(title=f"Calibrated from {snap.provenance.reporting_period}")
    table.add_column("Issuer")
    table.add_column("Share", justify="right")
    table.add_column("Published TD", justify="right")
    table.add_column("Baseline TD", justify="right")
    table.add_column("BD", justify="right")
    for profile in sorted(params.issuers, key=lambda p: -p.volume_share):
        table.add_row(
            profile.bank_name,
            f"{profile.volume_share:.1%}",
            f"{profile.published_td_rate:.3%}",
            f"{profile.baseline_td_rate:.3%}",
            f"{profile.published_bd_rate:.3%}",
        )
    console.print(table)
    console.print(
        f"Volume-weighted TD [bold]{snap.volume_weighted_td:.3%}[/bold] | "
        f"BD [bold]{snap.volume_weighted_bd:.3%}[/bold] | "
        f"success [bold]{snap.volume_weighted_success:.2%}[/bold]"
    )
    console.print(f"Issuers degraded [bold]{fraction:.2%}[/bold] of the time")
    console.print(
        f"[dim]{len(params.assumption_names)} Tier-2 assumptions in effect: "
        f"{', '.join(params.assumption_names)}[/dim]"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json.loads(params.model_dump_json()), indent=2) + "\n",
        encoding="utf-8",
    )
    console.print(f"\n[green]Wrote[/green] {output}")


@app.command("generate")
def generate_command(
    params_path: Annotated[
        Path, typer.Option("--params", help="Calibrated world parameters.")
    ] = DEFAULT_OUTPUT,
    n_cases: Annotated[int, typer.Option("--cases")] = 5000,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    out_dir: Annotated[Path, typer.Option("--out")] = Path("data/generated/batch"),
) -> None:
    """Generate a batch of recovery cases with quarantined potential outcomes."""
    from collections import Counter

    from recovery.calibration.models import WorldParameters
    from recovery.world.generate import generate, write_batch
    from recovery.world.oracle.segments import Segment, classify

    params = WorldParameters.model_validate_json(params_path.read_text(encoding="utf-8"))
    observable, oracle = generate(params, n_cases=n_cases, seed=seed)
    cases_p, logged_p, oracle_p = write_batch(observable, oracle, out_dir)

    counts = Counter(classify(o) for o in oracle.outcomes)
    table = Table(title=f"Batch seed={seed}  n={n_cases}")
    table.add_column("Segment")
    table.add_column("Cases", justify="right")
    table.add_column("Share", justify="right")
    for segment in Segment:
        if counts[segment]:
            table.add_row(segment.value, str(counts[segment]), f"{counts[segment] / n_cases:.1%}")
    console.print(table)

    holdout = sum(d.is_holdout for d in observable.logged)
    console.print(
        f"Randomised holdout [bold]{holdout / n_cases:.1%}[/bold] | "
        f"calibration [dim]{observable.params_provenance}[/dim]"
    )
    console.print(f"\n[green]Wrote[/green] {cases_p}\n[green]Wrote[/green] {logged_p}")
    console.print(
        f"[yellow]Wrote[/yellow] {oracle_p} "
        "[dim](quarantined - policy code must not read this)[/dim]"
    )


@app.command("diagnose")
def diagnose_command(
    params_path: Annotated[Path, typer.Option("--params")] = DEFAULT_OUTPUT,
    n_cases: Annotated[int, typer.Option("--cases")] = 8000,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Score the degradation detector against ground truth and the naive rule."""
    from recovery.calibration.models import WorldParameters
    from recovery.diagnose.issuer_health import IssuerHealthModel, ThresholdHealthModel
    from recovery.evaluate.diagnosis_eval import (
        evaluate_health_models,
        volume_stratified_report,
    )
    from recovery.world.generate import generate

    params = WorldParameters.model_validate_json(params_path.read_text(encoding="utf-8"))
    observable, oracle = generate(params, n_cases=n_cases, seed=seed)

    prevalence = sum(o.issuer_degraded_at_failure for o in oracle.outcomes) / n_cases
    console.print(f"True degradation prevalence: [bold]{prevalence:.2%}[/bold]\n")

    scores, naive = evaluate_health_models(observable.features, oracle.outcomes)
    table = Table(title="Degradation detection")
    for col in ("Detector", "Precision", "Recall", "F1", "Fires", "Cost/case"):
        table.add_column(col, justify="right" if col != "Detector" else "left")
    for s in [*scores, naive]:
        table.add_row(
            s.name,
            f"{s.precision:.3f}",
            f"{s.recall:.3f}",
            f"{s.f1:.3f}",
            f"{s.positive_rate:.2%}",
            f"{s.weighted_cost:.4f}",
        )
    console.print(table)

    best = min(scores, key=lambda s: s.weighted_cost)
    delta = (naive.weighted_cost - best.weighted_cost) / naive.weighted_cost
    console.print(
        f"Lowest expected cost: [bold]{best.name}[/bold] — "
        f"{delta:.0%} below the fixed-threshold rule\n"
    )

    bayes = IssuerHealthModel(decision_threshold=best.threshold or 0.8).fit(observable.features)
    report = volume_stratified_report(
        observable.features, oracle.outcomes, bayes, ThresholdHealthModel()
    )
    strat = Table(title="By observation volume (where the rule breaks)")
    for col in ("Volume", "n", "Bayes prec", "Bayes fires", "Rule prec", "Rule fires"):
        strat.add_column(col, justify="right" if col != "Volume" else "left")
    for bucket, v in report.items():
        strat.add_row(
            bucket,
            str(int(v["n"])),
            f"{v['bayes_precision']:.3f}",
            f"{v['bayes_fire_rate']:.1%}",
            f"{v['rule_precision']:.3f}",
            f"{v['rule_fire_rate']:.1%}",
        )
    console.print(strat)


if __name__ == "__main__":
    app()
