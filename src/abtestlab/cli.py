"""Typer CLI for replaying an experiment observation log.

``abtestlab monitor <log.csv>`` reads a chronological log of per-arm counts and
prints the look-by-look sequential replay plus the recommendation, including
what a naive dashboard peeker would have concluded instead.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from abtestlab.monitor import (
    DEFAULT_SRM_THRESHOLD,
    LookSequenceError,
    build_looks,
    check_srm,
    run_monitor,
)
from abtestlab.monitor.run import KEEP_RUNNING, REFUSED_SRM, SHIP

app = typer.Typer(help="Replay an A/B test observation log with an always-valid stopping rule.")
console = Console()

STATUS_STYLE = {SHIP: "bold green", REFUSED_SRM: "bold red", KEEP_RUNNING: "bold yellow"}

LogArgument = Annotated[
    Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Observation log CSV.")
]
IncrementalOption = Annotated[
    bool,
    typer.Option("--incremental", help="Rows hold per-period counts rather than running totals."),
]
ControlShareOption = Annotated[
    float, typer.Option("--control-share", min=1e-6, max=1 - 1e-6, help="Intended control share.")
]
SrmThresholdOption = Annotated[
    float, typer.Option("--srm-threshold", min=1e-12, max=0.5, help="SRM refusal p-value.")
]


def read_log(path: Path) -> list[dict[str, Any]]:
    """Read an observation log from CSV.

    Args:
        path: CSV file with a header row containing the four count columns and
            an optional ``label`` column.

    Returns:
        Rows in file order.

    Raises:
        LookSequenceError: If the file has no data rows.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise LookSequenceError(f"{path} contains no data rows")
    return rows


@app.command("monitor")
def monitor(
    log: LogArgument,
    incremental: IncrementalOption = False,
    alpha: Annotated[float, typer.Option(min=1e-6, max=0.499)] = 0.05,
    expected_share_control: ControlShareOption = 0.5,
    srm_threshold: SrmThresholdOption = DEFAULT_SRM_THRESHOLD,
) -> None:
    """Replay LOG and print the sequential monitoring report."""
    try:
        report = run_monitor(
            read_log(log),
            cumulative=not incremental,
            alpha=alpha,
            expected_share_control=expected_share_control,
            srm_threshold=srm_threshold,
        )
    except LookSequenceError as exc:
        console.print(f"[bold red]unreadable log:[/] {exc}")
        raise typer.Exit(code=2) from exc

    console.print(
        f"SRM gate: control took {report.srm.observed_share_control:.4%} of units "
        f"(chi-square p={report.srm.p_value:.4g}, threshold {report.srm.threshold:g}) -> "
        + ("[green]healthy[/]" if report.srm.healthy else "[red]BROKEN[/]")
    )

    if report.looks:
        table = Table(title=f"{log.name}: {len(report.looks)} looks")
        table.add_column("look")
        table.add_column("n/arm", justify="right")
        table.add_column("lift", justify="right")
        table.add_column("always-valid p", justify="right")
        table.add_column("fixed p", justify="right")
        for item in report.looks:
            safe = f"{item.always_valid_p:.4f}"
            naive = f"{item.fixed_horizon_p:.4f}"
            table.add_row(
                item.look.label,
                f"{item.look.n_treatment:,}",
                f"{item.relative_lift:+.2%}",
                f"[green]{safe}[/]" if item.crosses_always_valid else safe,
                f"[yellow]{naive}[/]" if item.crosses_fixed else naive,
            )
        console.print(table)

    style = STATUS_STYLE.get(report.status, "bold magenta")
    console.print(f"[{style}]{report.status.upper()}[/] {report.reason}")

    if report.naive_fixed is not None and report.always_valid is not None:
        safe_at = report.always_valid.stop_label or "never"
        naive_at = report.naive_fixed.stop_label or "never"
        console.print(
            f"always-valid rule fired at {safe_at}; naive fixed-horizon rule fired at {naive_at}."
        )
    if report.prob_treatment_beats_control is not None:
        console.print(
            f"At {report.decision_look}: P(treatment beats control) = "
            f"{report.prob_treatment_beats_control:.1%}, "
            f"expected loss if shipped = {report.expected_loss_if_ship:.5f}"
        )


def main() -> None:
    """Console script entry point."""
    app()


@app.command("srm")
def srm(
    log: LogArgument,
    incremental: IncrementalOption = False,
    expected_share_control: ControlShareOption = 0.5,
    srm_threshold: SrmThresholdOption = DEFAULT_SRM_THRESHOLD,
) -> None:
    """Check assignment health only, and exit non-zero if the split is broken.

    Cheap enough to run on every experiment export in CI: it touches exposure
    counts only and never looks at a conversion number.
    """
    try:
        looks = build_looks(read_log(log), cumulative=not incremental)
    except LookSequenceError as exc:
        console.print(f"[bold red]unreadable log:[/] {exc}")
        raise typer.Exit(code=2) from exc

    verdict = check_srm(
        looks[-1],
        expected_share_control=expected_share_control,
        threshold=srm_threshold,
    )
    console.print(
        f"{log.name}: control {verdict.observed_share_control:.4%} of "
        f"{looks[-1].total_units:,} units, expected {verdict.expected_share_control:.2%}, "
        f"chi-square={verdict.chi_square:.2f}, p={verdict.p_value:.4g}"
    )
    if verdict.healthy:
        console.print("[green]assignment healthy[/]")
        return
    console.print(f"[bold red]sample ratio mismatch[/] (p < {verdict.threshold:g})")
    raise typer.Exit(code=1)
