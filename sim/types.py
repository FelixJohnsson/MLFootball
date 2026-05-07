"""Lightweight structs returned across the sim API."""

from __future__ import annotations

from typing import NamedTuple


class StepInfo(NamedTuple):
    """Outcome of one env step (Gymnasium-style mapping)."""

    terminated: bool
    truncated: bool
    # When a ball crossed a goal mouth this physics step ("left" / "right" −x/+x mouths); unchanged if no goal.
    scoring_net: str | None = None
