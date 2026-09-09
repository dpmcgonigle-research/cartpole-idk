from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_training(run_dir: str | Path):
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    train = pd.read_csv(run_dir / "training_metrics.csv")
    eval_df = pd.read_csv(run_dir / "evaluation_metrics.csv")
    fig, ax = plt.subplots()
    ax.plot(train["step"], train["return"], alpha=0.25, label="training return")
    if len(train) >= 20:
        ax.plot(
            train["step"], train["return"].rolling(20).mean(), label="20-episode moving average"
        )
    ax.plot(eval_df["step"], eval_df["mean_return"], marker="o", label="greedy eval mean")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return")
    ax.set_title("CartPole DQN training")
    ax.legend()
    return fig, ax
