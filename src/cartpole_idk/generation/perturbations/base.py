from __future__ import annotations

import numpy as np


class Perturbation:
    """Per-episode action/observation transformation; the base implementation is nominal."""

    name = "none"

    def reset(self, rng: np.random.Generator, max_steps: int) -> dict:
        """Return nominal episode metadata; subclasses initialize perturbation state here.

        Args:
            rng: Episode random generator available to perturbations.
            max_steps: Episode horizon available for clamping onset times.
        """
        return {"onset": None}

    def agent_observation(self, true_obs: np.ndarray, step: int) -> np.ndarray:
        """Return an unmodified copy of the true state for the policy.

        Args:
            true_obs: Environment observation before any sensor perturbation.
            step: Zero-based episode timestep; unused for nominal behavior.
        """
        return np.asarray(true_obs, dtype=np.float32).copy()

    def executed_action(self, commanded_action: int, step: int) -> int:
        """Pass the policy command through unchanged.

        Args:
            commanded_action: Discrete action selected by the policy.
            step: Zero-based episode timestep; unused for nominal behavior.
        """
        return int(commanded_action)
