"""Lightweight structs returned across the sim API."""

from __future__ import annotations

from typing import NamedTuple


class StepInfo(NamedTuple):
    """Outcome of one env step (Gymnasium-style mapping)."""

    terminated: bool
    truncated: bool
    # Set when terminated is True — which net scored this tick ("left" / "right").
    scoring_net: str | None = None
