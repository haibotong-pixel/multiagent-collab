from __future__ import annotations

from pocketflow import AsyncFlow

from macollab.flows.patterns.draft_refine import build_draft_refine_flow
from macollab.flows.patterns.single import build_single_flow

PATTERNS = {
    "single": build_single_flow,
    "draft_refine": build_draft_refine_flow,
}


def build_flow(pattern: str) -> AsyncFlow:
    return PATTERNS[pattern]()
