# NPCI statistics snapshots

Tier-1 calibration evidence. Everything here is **downloaded published data**,
never authored by us.

## Which file to use

The correct source is **UPI - Top 50 member performance - Remitter** for the
reporting month. One table, no join: it carries volume, Approved %, BD % and
TD % together.

Two files that look relevant but are not:

- **AEPS BD & TD** - Aadhaar Enabled Payment System. Banking-correspondent
  and micro-ATM transactions, a different rail with materially different
  decline behaviour. Using it models the wrong world.
- **Beneficiary** performance - the payee side. We model issuer-side declines,
  so the payer's bank is what matters. Use Remitter.

## How to obtain a snapshot

1. Go to NPCI's UPI Ecosystem Statistics page:
   <https://www.npci.org.in/what-we-do/upi/upi-ecosystem-statistics>
2. Download **UPI - Top 50 member performance - <Month> - Remitter**.
3. Take the top 15-20 rows by volume. Record the cutoff in provenance notes.
4. Exclude non-bank entities. The July 2026 table contains at least one
   technology provider (Tri O Tech Solutions) reporting 100% approved and 0%
   declines; it is not an issuing bank and would distort volume shares.
5. Save as `snapshot_YYYY_MM.csv` with the schema below.
6. Write the matching `snapshot_YYYY_MM.provenance.yaml`.

Both files are required. `load_snapshot` refuses a CSV with no provenance
sidecar, because an unsourced number is indistinguishable from an invented
one once it is three phases downstream.

## CSV schema

```csv
bank_name,total_volume_mn,approved_rate,bd_rate,td_rate
State Bank of India,6622.02,0.9029,0.0902,0.0068
HDFC Bank Ltd.,1737.22,0.9300,0.0699,0.0001
```

Column names mirror NPCI's own where possible. `approved_rate` is optional
but strongly preferred: when present, the loader validates
`approved + BD + TD = 1.0` for every row. That is an internal property of
NPCI's table rather than an assumption of ours, and it catches a
percent/rate confusion, a mis-mapped column and a bad row in one check.

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
