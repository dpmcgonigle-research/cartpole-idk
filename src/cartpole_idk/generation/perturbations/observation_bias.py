from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cartpole_idk.generation.onset import NormalOnset
from cartpole_idk.generation.perturbations.base import Perturbation

FEATURE_INDEX = {
    "cart_position": 0,
    "cart_velocity": 1,
    "pole_angle": 2,
    "pole_angular_velocity": 3,
}


@dataclass(slots=True)
class ObservationBias(Perturbation):
    """Add bias and optional Gaussian noise to one policy-observation feature after onset."""

    onset: NormalOnset
    feature: str = "pole_angle"
    bias: float = 0.04
    noise_std: float = 0.0
    ramp_steps: int = 0
    name: str = field(default="observation_bias", init=False)
    _onset_step: int = field(default=0, init=False)
    _rng: np.random.Generator | None = field(default=None, init=False)

    def reset(self, rng: np.random.Generator, max_steps: int) -> dict:
        """Validate sensor settings and sample onset for a new episode.

        Args:
            rng: Episode random generator for onset and stochastic effects.
            max_steps: Episode horizon used to cap onset at max_steps - 1.
        """
        if self.feature not in FEATURE_INDEX:
            raise ValueError(f"Unknown feature: {self.feature}")
        if self.noise_std < 0 or self.ramp_steps < 0:
            raise ValueError("noise_std and ramp_steps must be non-negative")
        self._rng = rng
        self._onset_step = NormalOnset(
            self.onset.mean, self.onset.std, self.onset.minimum, max_steps - 1
        ).sample(rng)
        return {
            "onset": self._onset_step,
            "feature": self.feature,
            "bias": self.bias,
            "noise_std": self.noise_std,
            "ramp_steps": self.ramp_steps,
        }

    def agent_observation(self, true_obs: np.ndarray, step: int) -> np.ndarray:
        """Apply the configured bias ramp and noise to a copy of the true observation.

        Args:
            true_obs: Unmodified environment state.
            step: Zero-based episode timestep controlling onset and ramp progress.
        """
        obs = np.asarray(true_obs, dtype=np.float32).copy()
        if step < self._onset_step:
            return obs
        idx = FEATURE_INDEX[self.feature]
        scale = (
            1.0
            if self.ramp_steps == 0
            else min(1.0, (step - self._onset_step + 1) / self.ramp_steps)
        )
        obs[idx] += self.bias * scale
        if self._rng is not None and self.noise_std > 0:
            obs[idx] += float(self._rng.normal(0.0, self.noise_std))
        return obs
