"""Fit a reusable, versioned scaler and IDK basis."""

from pathlib import Path
from typing import Any

import click

from cartpole_idk.cli.selection import selection
from cartpole_idk.idk.pipeline import fit_dataset
from cartpole_idk.logging import configure_logging
from cartpole_idk.model import FitConfig, IDKConfig, WindowConfig


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("dataset", type=click.Path(path_type=Path))
@click.option(
    "--ids-file",
    type=click.Path(path_type=Path),
    help="One trajectory ID per line; default all",
)
@click.option("--mode", type=click.Choice(["whole", "window"]), default="whole")
@click.option("--window-length", type=int, default=25, help="Raw transitions per unit")
@click.option("--stride", type=int, default=1)
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option(
    "--representation",
    type=click.Choice(["state", "transition", "state_action", "state_action_next_state", "window"]),
    default="state",
)
@click.option("--observation-source", type=click.Choice(["true", "agent"]), default="true")
@click.option("--action-source", type=click.Choice(["commanded", "executed"]), default="executed")
@click.option("--representation-window-length", type=int, default=25)
@click.option("--psi", type=int, default=32)
@click.option("--t", type=int, default=200)
@click.option("--random-state", type=int, default=42)
def main(
    dataset: Path,
    output: Path,
    ids_file: Path | None,
    mode: str,
    window_length: int,
    stride: int,
    representation_window_length: int,
    **idk_options: Any,
) -> None:
    """Fit selected DATASET trajectories and save the model in OUTPUT.

    \b
    Args:
        dataset: Generated or prepared trajectory dataset directory.
        output: New or empty directory for the fitted scaler and IDK basis.
        ids_file: One trajectory ID per line; omitted means all trajectories.
        mode: Use whole trajectories or fixed-length windows as analysis units.
        window_length: Transitions per unit in window mode.
        stride: Transitions between consecutive unit starts in window mode.
        representation_window_length: Observation rows per temporal feature
            vector when representation is "window"; distinct from unit length.
        **idk_options: Feature and isolation-kernel settings from Click:
            representation: State, transition, state/action, state/action/next
                state, or flattened temporal-window features.
            observation_source: True environment states or agent observations.
            action_source: Commanded or actually executed actions.
            psi: Sampled feature rows (region centers) per isolation partition.
                Must be at least 2 and no greater than the fitting row count.
            t: Number of independently sampled isolation partitions.
                The embedding has t * psi columns.
            random_state: Seed for reproducible partition sampling.

    Notes:

        Only selected fitting data determines the scaler and basis. Use
        cartpole-embed to transform data with the saved model. Larger psi or t
        increases embedding size and computation.
    """
    configure_logging()
    try:
        config = FitConfig(
            dataset=dataset.resolve(),
            trajectory_ids=selection(dataset, ids_file),
            unit=WindowConfig.model_validate(
                dict(mode=mode, window_length=window_length, stride=stride)
            ),
            idk=IDKConfig.model_validate(
                {**idk_options, "window_length": representation_window_length}
            ),
        )
        fit_dataset(config).save(output)
    except (ValueError, KeyError, OSError) as exc:
        raise click.UsageError(str(exc)) from exc


if __name__ == "__main__":
    main()
