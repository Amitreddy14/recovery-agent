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
# System-wide TD fell from roughly 8-10% in 2016 to approximately 0.7-0.8%
# by 2025. Used as a sanity bound on the volume-weighted mean of any
# calibrated issuer set.

SYSTEM_TD_RATE_RANGE: Final[tuple[float, float]] = (0.005, 0.015)
"""Plausible band for volume-weighted system TD. Calibration fails outside."""

MERCHANT_BLENDED_SUCCESS_RANGE: Final[tuple[float, float]] = (0.90, 0.97)
"""Merchant-side blended success once BD is included. Below 0.90 indicates a
misconfigured world rather than a realistic one."""

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
