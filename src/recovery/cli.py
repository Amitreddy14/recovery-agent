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


if __name__ == "__main__":
    app()
