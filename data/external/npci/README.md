# NPCI statistics snapshots

Tier-1 calibration evidence. Everything here is **downloaded published data**,
never authored by us.

## How to obtain a snapshot

1. Go to NPCI's UPI Ecosystem Statistics page:
   <https://www.npci.org.in/what-we-do/upi/upi-ecosystem-statistics>
2. Open the **BD/TD & Uptime** section for the reporting month.
3. For remitter banks, record for each bank: name, Technical Decline %,
   Business Decline %, and transaction volume in millions.
4. Save as `snapshot_YYYY_MM.csv` with the schema below.
5. Write the matching `snapshot_YYYY_MM.provenance.yaml`.

Both files are required. `load_snapshot` refuses a CSV with no provenance
sidecar, because an unsourced number is indistinguishable from an invented
one once it is three phases downstream.

## CSV schema

```csv
bank_name,td_rate,bd_rate,volume_millions
HDFC Bank,0.0071,0.0412,9349.49
```

`td_rate` and `bd_rate` are **rates**, not percentages. The loader will
normalise a value above 1.0 as a percentage, but do not rely on that - state
rates explicitly. A percent/rate confusion is a silent 100x error in every
figure downstream, which is why the loader also validates the
volume-weighted aggregate against published system-level bounds and fails
loudly if it lands outside them.

## Provenance schema

```yaml
source_name: "NPCI UPI Ecosystem Statistics - BD/TD & Uptime"
source_url: "https://www.npci.org.in/what-we-do/upi/upi-ecosystem-statistics"
reporting_period: "2026-07"
retrieved_on: "2026-08-26"
retrieved_by: "your name"
notes: "Remitter bank table. Banks below 50M monthly volume excluded."
```

## Fixtures

`fixture_*.csv` files are **synthetic test data**, not NPCI figures. They
exist so the test suite runs without a network fetch. They are named
`fixture_` precisely so they can never be mistaken for evidence, and the
calibration CLI refuses to run against them without `--allow-fixture`.
