"""The console API.

Serves a frozen snapshot plus the static console. Deliberately small: two
endpoints and a file mount. The console is a window onto a completed run, not
a live control plane, so there is nothing here that mutates state.

This module imports nothing that can reach `recovery.world`. That is enforced
by CI, and it is what guarantees no endpoint can serve a counterfactual: the
snapshot builder in `evaluate` decides what is exposed, and the API can only
hand over what it finds in the file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from recovery.paths import CONSOLE_SNAPSHOT

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Recovery agent console",
    description="Read-only view of a completed recovery run.",
    version="0.11",
)


def load_snapshot(path: Path = CONSOLE_SNAPSHOT) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"No snapshot at {path}. Build one with: python -m recovery.cli console --build"
            ),
        )
    return dict(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/snapshot")
def snapshot() -> dict[str, Any]:
    """The whole run in one payload.

    One request rather than six endpoints: the payload is a few hundred KB
    and the console needs all of it to render anything useful. Splitting it
    would add round trips and loading states for no benefit.
    """
    return load_snapshot()


@app.get("/api/cases/{case_id}")
def case(case_id: str) -> dict[str, Any]:
    data = load_snapshot()
    for row in data["cases"]:
        if row["case_id"] == case_id:
            return dict(row)
    raise HTTPException(status_code=404, detail=f"No case {case_id} in this snapshot")


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness only.

    Deliberately does not read the snapshot. A health check that deserialises
    567KB every ten seconds is a load generator, and it buries the request log
    under identical lines.
    """
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
