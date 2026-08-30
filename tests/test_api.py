"""Console API tests.

The console is where a reviewer forms an impression, so the failure that
matters most is not a crash — it is a screen that renders confidently from
stale or wrong data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from recovery.api import app as app_module
from recovery.api.app import app


@pytest.fixture
def snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    data: dict[str, Any] = {
        "generated_at": "2026-08-29T00:00:00+00:00",
        "run": {
            "seed": 42,
            "n_cases": 10,
            "calibration": "NPCI 2026-07",
            "policy_version": "2026.08",
        },
        "reconciliation": {
            "at_risk_paise": 100000,
            "recovered_paise": 40000,
            "forfeited_paise": 5000,
            "intervention_cost_paise": 100,
            "net_paise": 34900,
            "unrecovered_paise": 60000,
            "balances": True,
            "contacts": 3,
            "attempts": 5,
            "mandates_cancelled": 1,
        },
        "policies": [],
        "gates": [],
        "segments": [],
        "actions": [],
        "cases": [{"case_id": "case_1", "action": "no_action", "blocked": {}}],
    }
    path = tmp_path / "console.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(app_module, "CONSOLE_SNAPSHOT", path)
    monkeypatch.setattr(app_module, "load_snapshot", lambda p=path: json.loads(p.read_text()))
    return data


class TestEndpoints:
    def test_snapshot_is_served(self, snapshot: dict[str, Any]) -> None:
        response = TestClient(app).get("/api/snapshot")
        assert response.status_code == 200
        assert response.json()["run"]["seed"] == 42

    def test_case_lookup(self, snapshot: dict[str, Any]) -> None:
        assert TestClient(app).get("/api/cases/case_1").status_code == 200

    def test_unknown_case_is_404_not_empty(self, snapshot: dict[str, Any]) -> None:
        """An empty 200 would render as a blank panel and look like a case
        with no detail rather than a case that does not exist."""
        response = TestClient(app).get("/api/cases/nope")
        assert response.status_code == 404
        assert "nope" in response.json()["detail"]

    def test_index_is_served(self) -> None:
        response = TestClient(app).get("/")
        assert response.status_code == 200
        assert b"Reconciliation" in response.content

    def test_missing_snapshot_says_how_to_fix_it(self, tmp_path: Path) -> None:
        """An empty screen is an invitation to act, so the error names the
        command rather than reporting that something went wrong."""
        from fastapi import HTTPException

        from recovery.api.app import load_snapshot

        with pytest.raises(HTTPException) as exc:
            load_snapshot(tmp_path / "absent.json")
        assert "--build" in exc.value.detail


class TestSnapshotShape:
    def test_reconciliation_balances_by_construction(self, snapshot: dict[str, Any]) -> None:
        r = snapshot["reconciliation"]
        assert r["recovered_paise"] + r["unrecovered_paise"] == r["at_risk_paise"]

    def test_no_read_write_endpoints(self) -> None:
        """The console is a window onto a completed run. Nothing here mutates
        state, and the route table should make that obvious."""
        methods = {m for route in app.routes for m in getattr(route, "methods", set())}
        assert methods <= {"GET", "HEAD"}
