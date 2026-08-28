"""Published payment-system statistics used as calibration anchors.

Every constant here traces to a public source and is cited inline. These are
*not* tunable parameters — they are external facts the simulator must be
consistent with. If a calibrated world violates one of these bounds, the
calibration is wrong, not the bound.

Anything the sources do not publish is deliberately absent from this module
and lives in `assumptions.py` instead.
"""

from __future__ import annotations

from typing import Final

# --- NPCI decline classification -------------------------------------------
# NPCI Circular OC-149 (June 2022) sets ecosystem targets for the two decline
# classes it publishes per bank.

NPCI_TD_TARGET: Final[float] = 0.01
"""Technical Decline target: <1% of transactions. Bank/NPCI infrastructure."""

NPCI_BD_TARGET: Final[float] = 0.05
"""Business Decline target: <5% of transactions. User-side causes."""

# --- System-wide observed rates --------------------------------------------
# Bands are wide on purpose. They exist to catch a mis-parsed column or a
# percent/rate confusion, not to encode a forecast. The tight check is the
# per-row sum-to-100 invariant in `npci.py`, which is a property of the
# source data rather than an assumption of ours.
#
# Observed on the NPCI Top 50 Remitter table for 2026-07 (top 20 by volume,
# 23,152.76 Mn transactions): TD 0.376%, BD 10.884%, approved 88.737%.

SYSTEM_TD_RATE_RANGE: Final[tuple[float, float]] = (0.001, 0.020)
"""Plausible band for volume-weighted remitter TD.

Long-run trend is downward: roughly 8-10% in 2016, ~0.7-0.8% by 2025, 0.376%
observed in 2026-07. The floor is deliberately far below the current value so
continued improvement does not start rejecting valid snapshots.
"""

REMITTER_APPROVAL_RATE_RANGE: Final[tuple[float, float]] = (0.80, 0.96)
"""Plausible band for volume-weighted remitter approval rate.

NOT the same quantity as merchant checkout success. This covers all UPI
traffic on the remitter side, including P2P, where user-side business
declines (wrong PIN, insufficient balance, abandonment) are counted. The
merchant-side figure below is measured on a different denominator and the
two must not be compared.
"""

MERCHANT_P2M_SUCCESS_RANGE: Final[tuple[float, float]] = (0.90, 0.97)
"""Merchant-side P2M checkout success, post-retry.

Recorded for reference and used in Phase 3 as a target for the generated
merchant world. Deliberately NOT used to validate remitter snapshots -
conflating the two was INC-006.
"""

APPROVED_BD_TD_SUM_TOLERANCE: Final[float] = 0.0005
"""Per-row tolerance on approved + BD + TD = 1.0.

NPCI publishes these to two decimal places as percentages, so rounding can
put a row at 99.99 or 100.01. Anything further out means a parsing or source
problem.
"""

# --- RBI E-mandate Framework, 2026 -----------------------------------------
# Issued 2026-04-21. Mirrored in configs/compliance/policy.yaml; duplicated
# here because the generator needs them and must not import the compliance
# engine.

EMANDATE_AFA_CEILING_DEFAULT_PAISE: Final[int] = 15_000_00
EMANDATE_AFA_CEILING_ELEVATED_PAISE: Final[int] = 100_000_00
EMANDATE_PRE_DEBIT_LEAD_HOURS: Final[int] = 24


SOURCES: Final[dict[str, str]] = {
    "npci_oc149": (
        "NPCI Circular OC-149 (June 2022) - Technical and Business Decline "
        "classification and ecosystem targets"
    ),
    "npci_bd_td_uptime": (
        "NPCI, UPI Ecosystem Statistics - BD/TD & Uptime, published monthly "
        "per bank. https://www.npci.org.in/what-we-do/upi/upi-ecosystem-statistics"
    ),
    "npci_autopay": (
        "NPCI, Autopay Ecosystem Statistics. "
        "https://www.npci.org.in/what-we-do/autopay/ecosystem-statistics"
    ),
    "rbi_emandate_2026": ("RBI, Digital Payments - E-mandate Framework, 2026 (issued 2026-04-21)"),
}
