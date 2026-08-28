"""Ingest a published NPCI statistics snapshot.

Input is a CSV alongside a provenance sidecar. Both are required: a snapshot
without stated provenance is rejected at load time rather than silently
becoming an "assumed" number three phases later.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import yaml

from recovery.calibration import priors
from recovery.calibration.models import IssuerStatistic, NpciSnapshot, Provenance

REQUIRED_COLUMNS = frozenset({"bank_name", "td_rate", "bd_rate", "total_volume_mn"})


class CalibrationError(RuntimeError):
    """Raised when a snapshot is missing, malformed, or implausible."""


def load_snapshot(csv_path: Path, provenance_path: Path) -> NpciSnapshot:
    """Load and validate a snapshot from disk."""
    if not csv_path.exists():
        raise CalibrationError(
            f"No NPCI statistics at {csv_path}. See data/external/npci/README.md "
            "for how to obtain and record a snapshot."
        )
    if not provenance_path.exists():
        raise CalibrationError(
            f"No provenance sidecar at {provenance_path}. A snapshot without "
            "recorded provenance is not usable as calibration evidence."
        )

    provenance = _load_provenance(provenance_path)
    issuers = _load_issuers(csv_path)
    snapshot = NpciSnapshot(provenance=provenance, issuers=issuers)
    _check_against_published_bounds(snapshot)
    return snapshot


def _load_provenance(path: Path) -> Provenance:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CalibrationError(f"{path}: expected a YAML mapping")
    retrieved = raw.get("retrieved_on")
    if isinstance(retrieved, str):
        raw["retrieved_on"] = date.fromisoformat(retrieved)
    return Provenance.model_validate(raw)


def _load_issuers(path: Path) -> tuple[IssuerStatistic, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise CalibrationError(f"{path}: missing columns {sorted(missing)}")

        issuers: list[IssuerStatistic] = []
        for line_no, row in enumerate(reader, start=2):
            try:
                issuers.append(
                    IssuerStatistic(
                        bank_name=row["bank_name"].strip(),
                        td_rate=_percent_to_rate(row["td_rate"]),
                        bd_rate=_percent_to_rate(row["bd_rate"]),
                        volume_millions=float(row["total_volume_mn"].replace(",", "")),
                        approved_rate=(
                            _percent_to_rate(row["approved_rate"])
                            if row.get("approved_rate")
                            else None
                        ),
                    )
                )
            except (ValueError, KeyError) as exc:
                raise CalibrationError(f"{path}:{line_no}: {exc}") from exc

    if not issuers:
        raise CalibrationError(f"{path}: no data rows")
    return tuple(issuers)


def _percent_to_rate(value: str) -> float:
    """NPCI publishes percentages. We store rates.

    Accepts either form and normalises, because a snapshot pasted as `0.82`
    (percent) and one pasted as `0.0082` (rate) are both plausible and the
    difference is a silent 100x error in every downstream figure. Values above
    1.0 are unambiguously percentages; below that we require the CSV to state
    rates, which the README specifies.
    """
    parsed = float(value.strip().rstrip("%"))
    if parsed < 0:
        raise ValueError(f"negative rate: {parsed}")
    return parsed / 100.0 if parsed > 1.0 else parsed


def _check_against_published_bounds(snapshot: NpciSnapshot) -> None:
    """Reject snapshots inconsistent with published system-level figures."""
    td = snapshot.volume_weighted_td
    lo, hi = priors.SYSTEM_TD_RATE_RANGE
    if not lo <= td <= hi:
        raise CalibrationError(
            f"volume-weighted TD is {td:.4%}, outside the plausible published "
            f"band {lo:.2%}-{hi:.2%}. Check whether the source column is a "
            "percentage or a rate, and whether volumes are correct."
        )

    success = snapshot.volume_weighted_success
    slo, shi = priors.REMITTER_APPROVAL_RATE_RANGE
    if not slo <= success <= shi:
        raise CalibrationError(
            f"volume-weighted approval is {success:.4%}, outside the plausible "
            f"remitter band {slo:.1%}-{shi:.1%}. Note this is remitter-side "
            "approval across all UPI traffic, not merchant checkout success - "
            "the two have different denominators (INC-006)."
        )
