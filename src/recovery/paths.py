"""Shared filesystem locations.

Deliberately imports nothing. Both the snapshot builder (which needs the
world) and the API (which must not be able to reach it) need this path, and a
neutral module is how they share it without creating a dependency edge
between them.
"""

from __future__ import annotations

from pathlib import Path

CONSOLE_SNAPSHOT = Path("data/generated/console.json")
"""Frozen output of a completed run, written by evaluate, served by api."""
