"""Click subcommands for reproducible experience analytics."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import click

from cartpole_idk.analytics.metrics import DEFAULT_MAX_PAIRS, METRICS
from cartpole_idk.analytics.workflow import execute_analysis
from cartpole_idk.cli.options import VariadicOptionsCommand
from cartpole_idk.logging import configure_logging
from cartpole_idk.model import (
    AnalysisConfig,
    ClusterConfig,
    MetricConfig,
    NeighborConfig,
    PopulationConfig,
)


def common_options(function: Callable[..., Any]) -> Callable[..., Any]:
    """Common artifact inputs and result destination; no fitting controls."""
    for decorator in reversed(
        [
            click.argument("embeddings", type=click.Path(path_type=Path)),
            click.option("--output", type=click.Path(path_type=Path), required=True),
            click.option(
                "--max-pairs",
                type=int,
                default=DEFAULT_MAX_PAIRS,
                help="Maximum entries in any dense pairwise matrix",
            ),
        ]
    ):
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
            "--reference",
            type=click.Path(path_type=Path),
            help="Reference embedding artifact; default input artifact",
        ),
        click.option("--nominal-reference", type=click.Path(path_type=Path)),
        click.option("--failure-reference", type=click.Path(path_type=Path)),
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
    """Convert Click parameters immediately to validated pipeline models."""
    try:
        nested: dict[str, Any] = {}
        if command in {"pairwise", "neighbors", "rolling"}:
            nested["metric"] = MetricConfig.model_validate(
                {key: options.pop(key) for key in MetricConfig.model_fields if key in options}
            )
        if command in {"neighbors", "rolling"}:
            nested["neighbors"] = NeighborConfig.model_validate(
                {key: options.pop(key) for key in NeighborConfig.model_fields if key in options}
            )
        if command == "cluster":
            nested["cluster"] = ClusterConfig.model_validate(
                {key: options.pop(key) for key in ClusterConfig.model_fields if key in options}
            )
        if command == "population":
            options.pop("metric")  # CLI supports only rbf-mmd for this command.
            nested["population"] = PopulationConfig.model_validate(
                {key: options.pop(key) for key in PopulationConfig.model_fields if key in options}
            )
        for key, value in options.items():
            if isinstance(value, Path):
                options[key] = value.resolve()
        config = AnalysisConfig.model_validate({"command": command, **options, **nested})
        execute_analysis(config)
    except (ValueError, KeyError, OSError) as exc:
        raise click.UsageError(str(exc)) from exc


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Analyze saved embeddings without fitting or transforming trajectories.

    Notes:

        Choose a subcommand for pairwise comparisons, neighbor retrieval,
        rolling scores, clustering, or population comparisons. Artifacts used
        together must share the same fitted model. See each subcommand's help
        for its arguments.
    """
    configure_logging()


@main.command()
@common_options
@metric_options
@click.option("--reference", type=click.Path(path_type=Path))
def pairwise(**options: Any) -> None:
    """Compute pairwise similarities or distances between analysis units.

    \b
    Args:
        **options: Click arguments passed to the analysis configuration:
            embeddings: Query embedding artifact directory.
            output: Result directory.
            reference: Reference artifact; defaults to the query artifact.
            metric: idk similarity, idk-distance, cosine similarity, JS
                divergence, or directional KL divergence (query to reference).
            epsilon: Positive smoothing required only for KL.
            max_pairs: Maximum entries allowed in a dense pairwise matrix.

    Notes:

        Native IDK similarity is the embedding dot product divided by t.
        Output rows correspond to query units and columns to reference units.
    """
    run_analysis("pairwise", options)


@main.command()
@common_options
@metric_options
@neighbor_options
def neighbors(**options: Any) -> None:
    """Retrieve reference neighbors and aggregate likeness scores.

    \b
    Args:
        **options: Click arguments passed to the analysis configuration:
            embeddings: Query embedding artifact directory.
            output: Result directory.
            reference: Reference artifact; defaults to the query artifact.
            nominal_reference: Nominal reference artifact, paired with failure_reference.
            failure_reference: Failure reference artifact; replaces reference
                together with nominal_reference.
            metric: IDK/cosine similarity or IDK/JS/KL distance or divergence.
            epsilon: Positive smoothing required only for KL.
            k: Maximum eligible neighbors per query and reference group.
            include_self: Allow a unit to match itself; excluded by default.
            exclude_same_trajectory: Exclude references from the query trajectory.
            max_overlap: Allowed same-source window intersection divided by
                the shorter window length, between zero and one.
            max_pairs: Maximum entries allowed in a dense pairwise matrix.

    Notes:

        Scores average available neighbors. With nominal and failure groups,
        a positive likeness margin favors the nominal group.
    """
    run_analysis("neighbors", options)


@main.command()
@common_options
@click.option("--query-id", required=True)
@metric_options
@neighbor_options
def rolling(**options: Any) -> None:
    """Compute reference likeness through a query trajectory's windows.

    \b
    Args:
        **options: Click arguments passed to the analysis configuration:
            embeddings: Artifact containing existing window embeddings.
            query_id: Trajectory ID whose windows are scored.
            output: Result directory.
            reference: Reference artifact; defaults to the input artifact.
            nominal_reference: Nominal artifact, paired with failure_reference.
            failure_reference: Failure artifact; use both groups instead of reference.
            metric: IDK/cosine similarity or IDK/JS/KL distance or divergence.
            epsilon: Positive smoothing required only for KL.
            k: Maximum eligible neighbors per window and reference group.
            include_self: Allow a window to match itself; excluded by default.
            exclude_same_trajectory: Exclude references from the query trajectory.
            max_overlap: Maximum intersection/shorter-length ratio for
                same-source windows, between zero and one.
            max_pairs: Maximum entries allowed in a dense pairwise matrix.

    Notes:

        Windows must already exist in the artifact; this command does not
        extract them. Overlapping windows can produce dependent scores.
    """
    run_analysis("rolling", options)


@main.command(cls=VariadicOptionsCommand)
@common_options
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
@click.option("--random-state", type=int, default=42)
def cluster(**options: Any) -> None:
    """Cluster units and evaluate their correspondence with metadata.

    \b
    Args:
        **options: Click arguments passed to the analysis configuration:
            embeddings: Embedding artifact directory.
            output: Result directory.
            algorithm: HDBSCAN, DBSCAN, spectral, or Dirichlet-process Gaussian mixture.
            distance: IDK distance or square-root JS for density clustering.
            categorical: Metadata fields used to evaluate the resulting clusters.
            eps: DBSCAN neighborhood radius.
            min_samples: Density neighborhood size for DBSCAN/HDBSCAN.
            min_cluster_size: Minimum HDBSCAN cluster size.
            cluster_selection_method: HDBSCAN excess-of-mass (eom) or leaf selection.
            n_clusters: Requested number of spectral clusters.
            svd_components: Reduced feature dimensions for the Gaussian mixture.
            n_components: Maximum mixture components.
            max_iter: Maximum mixture fitting iterations.
            weight_concentration_prior: Mixture concentration; omitted uses sklearn's default.
            random_state: Seed for spectral clustering, SVD, and mixture fitting.
            max_pairs: Maximum entries allowed in a dense pairwise matrix.

    Notes:

        Algorithm-specific settings apply only to that method. Metadata labels
        are used for evaluation, not fitting. Density methods label noise as -1.
    """
    run_analysis("cluster", options)


@main.command()
@common_options
@click.option("--group-b", type=click.Path(path_type=Path), required=True)
@click.option("--metric", type=click.Choice(["rbf-mmd"]), default="rbf-mmd")
@click.option("--bandwidth", type=float, help="RBF sigma; default pooled median distance")
@click.option("--estimator", type=click.Choice(["biased", "unbiased"]), default="biased")
@click.option("--permutations", type=int, default=0)
@click.option("--random-state", type=int, default=42)
def population(**options: Any) -> None:
    """Compare populations using RBF-MMD and optional permutations.

    \b
    Args:
        **options: Click arguments passed to the analysis configuration:
            embeddings: Group A embedding artifact directory.
            group_b: Group B embedding artifact directory.
            output: Result directory.
            metric: rbf-mmd, the supported population discrepancy measure.
            bandwidth: RBF sigma; omitted uses the pooled median positive distance.
            estimator: Biased or unbiased estimator of squared MMD.
            permutations: Number of label shuffles; zero skips the significance test.
            random_state: Seed for permutation sampling.
            max_pairs: Maximum entries allowed in a dense pairwise matrix.

    Notes:

        Smaller MMD means more similar populations. The unbiased estimate needs
        at least two units per group and may be negative. Permutation tests
        assume exchangeable units; overlapping windows may violate this.
    """
    run_analysis("population", options)


if __name__ == "__main__":
    main()
