from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cartpole_idk.generation.onset import NormalOnset
from cartpole_idk.generation.perturbations.base import Perturbation


@dataclass(slots=True)
class ActionFlip(Perturbation):
    onset: NormalOnset
    probability: float = 0.15
    name: str = field(default="action_flip", init=False)
    _onset_step: int = field(default=0, init=False)
    _rng: np.random.Generator | None = field(default=None, init=False)

    def reset(self, rng: np.random.Generator, max_steps: int) -> dict:
        if not 0 <= self.probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        self._rng = rng
        self._onset_step = NormalOnset(
            self.onset.mean, self.onset.std, self.onset.minimum, max_steps - 1
        ).sample(rng)
        return {"onset": self._onset_step, "flip_probability": self.probability}

    def executed_action(self, commanded_action: int, step: int) -> int:
        if step >= self._onset_step and self._rng is not None:
            if self._rng.random() < self.probability:
                return 1 - int(commanded_action)
        return int(commanded_action)
