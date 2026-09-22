from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    """Feed-forward network mapping observations to one Q-value per discrete action."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]):
        """Construct ReLU hidden layers and a linear action-value output.

        Args:
            obs_dim: Number of observation components.
            action_dim: Number of discrete actions.
            hidden_sizes: Width of each hidden layer, in order.
        """
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for size in hidden_sizes:
            layers += [nn.Linear(in_dim, size), nn.ReLU()]
            in_dim = size
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute action values for observation tensors.

        Args:
            x: Observation tensor with obs_dim components on its last axis.
        """
        return self.net(x)


@dataclass(slots=True)
class Transition:
    """One replay-buffer interaction: state, action, reward, next state, and ending flag."""

    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    """Bounded history of transitions with reproducible sampling without replacement."""

    def __init__(self, capacity: int, seed: int):
        """Create an empty buffer that evicts the oldest transitions at capacity.

        Args:
            capacity: Maximum number of stored transitions.
            seed: Seed for replay-sampling randomness.
        """
        self._data: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        """Number of currently stored transitions."""
        return len(self._data)

    def add(self, transition: Transition) -> None:
        """Append one environment interaction to the replay history.

        Args:
            transition: Interaction record to retain.
        """
        self._data.append(transition)

    def sample(self, n: int) -> list[Transition]:
        """Draw a replay minibatch without replacement.

        Args:
            n: Number of transitions to draw; must not exceed the current buffer size.
        """
        return self._rng.sample(list(self._data), n)


class DQNAgent:
    """Epsilon-greedy DQN policy with online and target networks and a replay update step."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...],
        learning_rate: float,
        gamma: float,
        seed: int,
        device: str = "cpu",
    ):
        """Initialize matching online/target networks and the optimizer.

        Args:
            obs_dim: Number of observation components.
            action_dim: Number of discrete actions.
            hidden_sizes: Network hidden-layer widths.
            learning_rate: Adam optimizer step size.
            gamma: Discount factor for future rewards.
            seed: Seed for network initialization and exploratory actions.
            device: PyTorch device for network parameters and input tensors.
        """
        torch.manual_seed(seed)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.gamma = gamma
        self.device = torch.device(device)
        self.online = QNetwork(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.target = QNetwork(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        """Choose an epsilon-greedy action; epsilon zero gives the greedy policy.

        Args:
            obs: Single environment observation.
            epsilon: Probability of selecting a uniformly random action.
        """
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_dim))
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.online(x).argmax(dim=1).item())

    def update(self, batch: list[Transition]) -> float:
        """Apply one target-network Bellman update and return the scalar Huber loss.

        Args:
            batch: Replay transitions providing observations, actions, rewards, and end flags.
        """
        obs = torch.as_tensor(
            np.stack([t.obs for t in batch]), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            [t.action for t in batch], dtype=torch.long, device=self.device
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        )
        next_obs = torch.as_tensor(
            np.stack([t.next_obs for t in batch]), dtype=torch.float32, device=self.device
        )
        done = torch.as_tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        q = self.online(obs).gather(1, actions).squeeze(1)
        with torch.no_grad():
            target = rewards + self.gamma * (1.0 - done) * self.target(next_obs).max(1).values

        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        """Copy online-network weights into the target network."""
        self.target.load_state_dict(self.online.state_dict())
