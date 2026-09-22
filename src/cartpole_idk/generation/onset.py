from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class NormalOnset:
    """Normally distributed onset timestep, rounded and clamped to configured bounds."""

    mean: float
    std: float = 0.0
    minimum: int = 0
    maximum: int | None = None

    def sample(self, rng: np.random.Generator) -> int:
        """Sample a zero-based onset timestep.

        Args:
            rng: Random generator used when the onset standard deviation is nonzero.
        """
        value = self.mean if self.std == 0 else rng.normal(self.mean, self.std)
        step = int(round(value))
        upper = self.maximum if self.maximum is not None else step
        return max(self.minimum, min(step, upper))
