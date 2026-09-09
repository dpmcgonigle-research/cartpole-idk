import numpy as np

from cartpole_idk.training.dqn import DQNAgent, ReplayBuffer, Transition


def test_agent_action_is_valid():
    agent = DQNAgent(4, 2, (16,), 1e-3, 0.99, 1)
    assert agent.act(np.zeros(4, dtype=np.float32)) in {0, 1}


def test_replay_buffer():
    buffer = ReplayBuffer(10, seed=1)
    t = Transition(np.zeros(4), 0, 1.0, np.ones(4), False)
    for _ in range(5):
        buffer.add(t)
    assert len(buffer) == 5
    assert len(buffer.sample(3)) == 3
