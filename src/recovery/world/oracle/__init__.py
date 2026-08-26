"""QUARANTINED: ground-truth potential outcomes.

This package holds Y(a) for every action `a` in the world simulator — the
counterfactuals that would be unobservable in production.

Nothing in `diagnose`, `uplift`, `policy`, `compliance`, `execute` or `api`
may import from here. The restriction is enforced by an import-linter
contract in CI, so a violation fails the build rather than silently
invalidating every metric this project reports.

Only `recovery.world` (to generate) and `recovery.evaluate` (to compute the
oracle ceiling and to validate off-policy estimators) may read this package.
"""
