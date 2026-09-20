import logging

import click

from cartpole_idk.logging import configure_logging
from cartpole_idk.preparation import prepare_dataset


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("dataset", nargs=-1, required=True)
@click.option("--output", required=True, help="New or empty output dataset directory", type=str)
@click.option("--episode-length", default=100, type=int)
@click.option("--min-episode-length", default=25, type=int)
@click.option("--success-threshold", default=300, type=int)
@click.option("--success-buffer", default=50, type=int)
@click.option("--episode-start", default=None, type=int)
@click.option("--seed", default=1000, type=int)
def main(
    dataset,
    output,
    episode_length,
    min_episode_length,
    success_threshold,
    success_buffer,
    episode_start,
    seed,
) -> None:
    configure_logging()
    try:
        store = prepare_dataset(
            dataset=dataset,
            output=output,
            episode_length=episode_length,
            min_episode_length=min_episode_length,
            success_threshold=success_threshold,
            success_buffer=success_buffer,
            episode_start=episode_start,
            seed=seed,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    logging.getLogger(__name__).info(
        "Preparation complete: %d segments saved in %s; details in preparation_report.json",
        len(store.manifest()),
        store.root,
    )


if __name__ == "__main__":
    main()
