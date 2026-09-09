from __future__ import annotations

from cartpole_idk.storage import Trajectory


def plot_phase_space(trajectory: Trajectory):
    import matplotlib.pyplot as plt

    obs = trajectory.true_observations
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(obs[:, 0], obs[:, 1])
    axes[0].set(xlabel="Cart position", ylabel="Cart velocity", title="Cart phase space")
    axes[1].plot(obs[:, 2], obs[:, 3])
    axes[1].set(xlabel="Pole angle", ylabel="Pole angular velocity", title="Pole phase space")
    return fig, axes
