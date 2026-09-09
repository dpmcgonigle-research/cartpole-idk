from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RepresentationName = Literal[
    "state", "transition", "state_action", "state_action_next_state", "window"
]


@dataclass(frozen=True, slots=True)
class IDKExperimentConfig:
    representation: RepresentationName = "state"
    observation_source: Literal["true", "agent"] = "true"
    action_source: Literal["commanded", "executed"] = "executed"
    window_length: int = 25
    psi: int = 32
    t: int = 200
    random_state: int = 42
