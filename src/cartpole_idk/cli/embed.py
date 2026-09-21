"""Transform trajectories using a previously saved fit artifact."""

from pathlib import Path

import click

from cartpole_idk.artifacts import FitArtifact
from cartpole_idk.cli.selection import selection
from cartpole_idk.idk.pipeline import embed_dataset
from cartpole_idk.logging import configure_logging
from cartpole_idk.model import EmbedConfig, WindowConfig


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--fit", "fit_path", required=True, type=click.Path(path_type=Path))
@click.option("--dataset", required=True, type=click.Path(path_type=Path))
@click.option(
    "--ids-file",
    type=click.Path(path_type=Path),
    help="One trajectory ID per line; default all",
)
@click.option("--mode", type=click.Choice(["whole", "window"]), default="whole")
@click.option("--window-length", type=int, default=25, help="Raw transitions per unit")
@click.option("--stride", type=int, default=1)
@click.option("--output", required=True, type=click.Path(path_type=Path))
def main(
    fit_path: Path,
    dataset: Path,
    output: Path,
    ids_file: Path | None,
    mode: str,
    window_length: int,
    stride: int,
) -> None:
    """Save sparse embeddings using a previously fitted model.

    \b
    Args:
        fit_path: Saved fit artifact containing the scaler and IDK basis.
        dataset: Generated or prepared trajectory dataset directory.
        output: New or empty directory for embeddings and unit metadata.
        ids_file: One trajectory ID per line; omitted means all trajectories.
        mode: Embed whole trajectories or fixed-length windows.
        window_length: Transitions per unit in window mode.
        stride: Transitions between consecutive window starts.

    Notes:

        Representation settings come from the fit artifact; nothing is refit
        or resampled. Only complete windows are used, with L transitions
        retaining L + 1 observations.
    """
    configure_logging()
    try:
        fit = FitArtifact.load(fit_path)
        config = EmbedConfig(
            dataset=dataset.resolve(),
            trajectory_ids=selection(dataset, ids_file),
            fit_artifact=fit_path.resolve(),
            fit_id=fit.fit_id,
            fit=fit.config,
            unit=WindowConfig.model_validate(
                dict(mode=mode, window_length=window_length, stride=stride)
            ),
        )
        embed_dataset(fit, config).save(output)
    except (ValueError, KeyError, OSError) as exc:
        raise click.UsageError(str(exc)) from exc


if __name__ == "__main__":
    main()
