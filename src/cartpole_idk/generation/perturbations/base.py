from __future__ import annotations

import numpy as np


class Perturbation:
    name = "none"

    def reset(self, rng: np.random.Generator, max_steps: int) -> dict:
        return {"onset": None}

    def agent_observation(self, true_obs: np.ndarray, step: int) -> np.ndarray:
        return np.asarray(true_obs, dtype=np.float32).copy()

    def executed_action(self, commanded_action: int, step: int) -> int:
        return int(commanded_action)
