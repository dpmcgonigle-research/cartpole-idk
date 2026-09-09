from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..onset import NormalOnset
from .base import Perturbation


@dataclass(slots=True)
class ActionDelay(Perturbation):
    """Delay commands after onset, holding the onset command while the queue fills."""

    onset: NormalOnset
    delay_steps: int = 3
    name: str = field(default="action_delay", init=False)
    _onset_step: int = field(default=0, init=False)
    _queue: deque[int] = field(default_factory=deque, init=False)

    def reset(self, rng: np.random.Generator, max_steps: int) -> dict:
        if self.delay_steps < 1:
            raise ValueError("delay_steps must be >= 1")
        self._onset_step = NormalOnset(
            self.onset.mean, self.onset.std, self.onset.minimum, max_steps - 1
        ).sample(rng)
        self._queue.clear()
        return {"onset": self._onset_step, "delay_steps": self.delay_steps}

    def executed_action(self, commanded_action: int, step: int) -> int:
        if step < self._onset_step:
            return int(commanded_action)
        self._queue.append(int(commanded_action))
        if len(self._queue) <= self.delay_steps:
            return int(self._queue[0])
        return int(self._queue.popleft())
