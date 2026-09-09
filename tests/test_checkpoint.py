from cartpole_idk.training.checkpoint import load_checkpoint, save_checkpoint
from cartpole_idk.training.dqn import DQNAgent


def test_checkpoint_roundtrip(tmp_path):
    agent = DQNAgent(4, 2, (8,), 1e-3, 0.99, 1)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(agent, path, step=123, metadata={"x": 1})
    loaded, payload = load_checkpoint(path)
    assert loaded.hidden_sizes == (8,)
    assert payload["step"] == 123
    assert payload["metadata"]["x"] == 1
