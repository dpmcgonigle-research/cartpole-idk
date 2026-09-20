"""Click subcommands for reproducible experience analytics."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import click

from cartpole_idk.analytics.clustering import ClusterConfig
from cartpole_idk.analytics.metrics import DEFAULT_MAX_PAIRS, METRICS
from cartpole_idk.analytics.workflow import AnalysisConfig, execute_analysis
from cartpole_idk.cli.options import VariadicOptionsCommand
from cartpole_idk.idk.config import IDKExperimentConfig
from cartpole_idk.logging import configure_logging


def common_options(function: Callable[..., Any]) -> Callable[..., Any]:
    """Shared dataset, unit construction and IDK fitting options."""
    decorators = [
        click.argument("dataset", type=click.Path(path_type=Path)),
        click.option("--output", type=click.Path(path_type=Path), required=True),
        click.option(
            "--fit-ids-file",
            type=click.Path(path_type=Path),
            help="Defaults to all dataset trajectories",
        ),
        click.option(
            "--analysis-ids-file",
            type=click.Path(path_type=Path),
            help="Defaults to all dataset trajectories",
        ),
        click.option(
            "--window-length", type=int, default=25, help="Raw transitions per analysis unit"
        ),
        click.option("--stride", type=int, default=1),
        click.option(
            "--representation",
            default="state",
            type=click.Choice(
                ["state", "transition", "state_action", "state_action_next_state", "window"]
            ),
        ),
        click.option(
            "--representation-window-length",
            type=int,
            default=25,
            help="Observation rows per temporal representation feature (distinct from unit length)",
        ),
        click.option("--observation-source", type=click.Choice(["true", "agent"]), default="true"),
        click.option(
            "--action-source", type=click.Choice(["commanded", "executed"]), default="executed"
        ),
        click.option("--psi", type=int, default=32),
        click.option("--t", type=int, default=200),
        click.option("--random-state", type=int, default=42),
        click.option(
            "--max-pairs",
            type=int,
            default=DEFAULT_MAX_PAIRS,
            help="Maximum entries in any dense pairwise matrix",
        ),
    ]
    for decorator in reversed(decorators):
        function = decorator(function)
    return function


def metric_options(function: Callable[..., Any]) -> Callable[..., Any]:
    function = click.option("--epsilon", type=float, help="Required positive smoothing for KL")(
        function
    )
    return click.option("--metric", type=click.Choice(list(METRICS)), default="idk")(function)


def neighbor_options(function: Callable[..., Any]) -> Callable[..., Any]:
    decorators = [
        click.option(
            "--reference-ids-file", type=click.Path(path_type=Path), help="Defaults to fit IDs"
        ),
        click.option("--nominal-reference-ids-file", type=click.Path(path_type=Path)),
        click.option("--failure-reference-ids-file", type=click.Path(path_type=Path)),
        click.option("--k", type=int, default=5),
        click.option("--include-self", is_flag=True),
        click.option("--exclude-same-trajectory", is_flag=True),
        click.option(
            "--max-overlap",
            type=float,
            help="Max intersection/shorter length for same-source reference windows",
        ),
    ]
    for decorator in reversed(decorators):
        function = decorator(function)
    return function


def run_analysis(command: str, options: dict[str, Any]) -> None:
    """Translate Click parameters into the existing analytics configuration."""
    if options.get("metric") == "kl" and (
        options.get("epsilon") is None or options["epsilon"] <= 0
    ):
        raise click.UsageError("--metric kl requires --epsilon > 0")
    options = {"command": command, **options}
    config = AnalysisConfig(
        dataset=options["dataset"],
        output=options["output"],
        mode=options["mode"],
        window_length=options["window_length"],
        stride=options["stride"],
        max_pairs=options["max_pairs"],
        idk=IDKExperimentConfig(
            representation=options["representation"],
            observation_source=options["observation_source"],
            action_source=options["action_source"],
            window_length=options["representation_window_length"],
            psi=options["psi"],
            t=options["t"],
            random_state=options["random_state"],
        ),
    )
    names = {
        "fit": "fit_ids_file",
        "analysis": "analysis_ids_file",
        "reference": "reference_ids_file",
        "nominal": "nominal_reference_ids_file",
        "failure": "failure_reference_ids_file",
        "group_a": "group_a_ids_file",
        "group_b": "group_b_ids_file",
    }
    selections = {role: options.get(flag) for role, flag in names.items()}
    if command == "cluster":
        options["cluster_config"] = ClusterConfig(
            **{name: options[name] for name in ClusterConfig.__dataclass_fields__}
        )
    try:
        execute_analysis(command, config, selections=selections, options=options)
    except (ValueError, KeyError, OSError) as exc:
        raise click.UsageError(str(exc)) from exc


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Analyze trajectories in a shared IDK feature space."""
    configure_logging()


@main.command()
@common_options
@click.option("--mode", type=click.Choice(["whole", "window"]), default="whole")
@metric_options
def pairwise(**options: Any) -> None:
    """Compute pairwise similarities or distances between analysis units."""
    run_analysis("pairwise", options)


@main.command()
@common_options
@click.option("--mode", type=click.Choice(["whole", "window"]), default="whole")
@metric_options
@neighbor_options
def neighbors(**options: Any) -> None:
    """Retrieve reference neighbors and aggregate likeness scores."""
    run_analysis("neighbors", options)


@main.command()
@common_options
@click.option("--query-id", required=True)
@metric_options
@neighbor_options
def rolling(**options: Any) -> None:
    """Compute reference likeness through a query trajectory's windows."""
    run_analysis("rolling", {**options, "mode": "window"})


@main.command(cls=VariadicOptionsCommand)
@common_options
@click.option("--mode", type=click.Choice(["whole", "window"]), default="whole")
@click.option(
    "--algorithm", type=click.Choice(["hdbscan", "dbscan", "spectral", "dpgmm"]), default="hdbscan"
)
@click.option("--distance", type=click.Choice(["idk-distance", "js"]), default="idk-distance")
@click.option(
    "--categorical", multiple=True, default=("perturbation_type", "checkpoint_id"), type=str
)
@click.option("--eps", type=float, default=0.1, help="DBSCAN radius")
@click.option("--min-samples", type=int)
@click.option("--min-cluster-size", type=int, default=5)
@click.option("--cluster-selection-method", type=click.Choice(["eom", "leaf"]), default="eom")
@click.option("--n-clusters", type=int, default=2)
@click.option("--svd-components", type=int, default=10)
@click.option("--n-components", type=int, default=10)
@click.option("--max-iter", type=int, default=500)
@click.option("--weight-concentration-prior", type=float)
def cluster(**options: Any) -> None:
    """Cluster units and evaluate their correspondence with metadata."""
    run_analysis("cluster", options)


@main.command()
@common_options
@click.option("--mode", type=click.Choice(["whole", "window"]), default="whole")
@click.option("--group-a-ids-file", type=click.Path(path_type=Path), required=True)
@click.option("--group-b-ids-file", type=click.Path(path_type=Path), required=True)
@click.option("--metric", type=click.Choice(["rbf-mmd"]), default="rbf-mmd")
@click.option("--bandwidth", type=float, help="RBF sigma; default pooled median distance")
@click.option("--estimator", type=click.Choice(["biased", "unbiased"]), default="biased")
@click.option("--permutations", type=int, default=0)
def population(**options: Any) -> None:
    """Compare populations using RBF-MMD and optional permutations."""
    run_analysis("population", options)


if __name__ == "__main__":
    main()
