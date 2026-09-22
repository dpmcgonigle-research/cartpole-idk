#!/usr/bin/env python3
"""Compare nominal and failing CartPole datasets and IDK embeddings.

This script is intended as a methodological sanity-check tool for the
CartPole-IDK experiments discussed in this project. It deliberately does NOT
fit a scaler or an IDK basis. Instead, it compares already-generated datasets
and, optionally, already-created fit/embedding artifacts.

Checks performed:
  1. Raw 4-D state distributions: x, x_dot, theta, theta_dot.
  2. Standardized state distributions, when a compatible 4-D scaler is saved.
  3. A simple non-IDK trajectory-summary logistic-regression baseline.
  4. Outside-isolation-region occupancy mass from saved IDK embeddings.
  5. Pairwise partition-wise Jensen-Shannon divergence for:
       nominal-nominal, failure-failure, nominal-failure.

Typical usage:

python dataset-comparisons.py \
  --nominal-dir datasets/prepared/.../nominal/trajectories \
  --failure-dir datasets/prepared/.../failure/trajectories \
  --fit-artifact artifacts/fit/.../psi-32_t-100_transition \
  --embedding-artifact artifacts/embeddings/.../whole \
  --output analysis/dataset-comparison

Notes:
- true_observations is assumed to be shaped (N, 4) and ordered as
  [x, x_dot, theta, theta_dot].
- embeddings.npz is assumed to be a SciPy sparse matrix saved with save_npz().
- units.parquet must contain a source trajectory identifier matching the
  trajectory_id stored in the trajectory .npz files.
- psi/t are inferred from fit config/metadata when possible; otherwise pass
  --psi and --t explicitly.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

FEATURE_NAMES = ("x", "x_dot", "theta", "theta_dot")


def _scalar_to_str(value: Any) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        value = arr.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_trajectories(directory: Path, group: str) -> list[dict[str, Any]]:
    """Load all trajectory NPZs under a directory recursively."""
    files = sorted(directory.rglob("*.npz"))
    records: list[dict[str, Any]] = []

    for path in files:
        with np.load(path, allow_pickle=False) as z:
            if "true_observations" not in z.files:
                continue

            obs = np.asarray(z["true_observations"], dtype=np.float64)
            if obs.ndim != 2 or obs.shape[1] != 4:
                continue

            trajectory_id = (
                _scalar_to_str(z["trajectory_id"])
                if "trajectory_id" in z.files
                else path.stem
            )

        records.append(
            {
                "trajectory_id": trajectory_id,
                "group": group,
                "path": path,
                "observations": obs,
            }
        )

    if not records:
        raise RuntimeError(
            f"No trajectory NPZ files containing true_observations found under {directory}"
        )

    return records


def concatenate_states(records: list[dict[str, Any]]) -> np.ndarray:
    return np.concatenate([r["observations"] for r in records], axis=0)


def describe(values: np.ndarray) -> dict[str, float | int]:
    q = np.quantile(values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q25": float(q[2]),
        "median": float(q[3]),
        "q75": float(q[4]),
        "q95": float(q[5]),
        "q99": float(q[6]),
        "max": float(np.max(values)),
        "max_abs": float(np.max(np.abs(values))),
    }


def compare_state_populations(
    nominal: np.ndarray,
    failure: np.ndarray,
    *,
    space_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-group summaries and between-group feature comparisons."""
    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for j, feature in enumerate(FEATURE_NAMES):
        n = nominal[:, j]
        f = failure[:, j]

        for group, values in (("nominal", n), ("failure", f)):
            row: dict[str, Any] = {
                "space": space_name,
                "group": group,
                "feature": feature,
            }
            row.update(describe(values))
            summary_rows.append(row)

        pooled_var = (
            ((len(n) - 1) * np.var(n, ddof=1) + (len(f) - 1) * np.var(f, ddof=1))
            / (len(n) + len(f) - 2)
        )
        pooled_sd = np.sqrt(pooled_var)
        cohens_d = (
            float((np.mean(f) - np.mean(n)) / pooled_sd)
            if pooled_sd > 0
            else np.nan
        )

        ks = ks_2samp(n, f)
        comparison_rows.append(
            {
                "space": space_name,
                "feature": feature,
                "mean_nominal": float(np.mean(n)),
                "mean_failure": float(np.mean(f)),
                "cohens_d_failure_minus_nominal": cohens_d,
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "wasserstein_1d": float(wasserstein_distance(n, f)),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(comparison_rows)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _recursive_find_int(obj: Any, names: set[str]) -> int | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in names and isinstance(value, int):
                return int(value)
        for value in obj.values():
            found = _recursive_find_int(value, names)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _recursive_find_int(value, names)
            if found is not None:
                return found
    return None


def infer_psi_t(
    fit_artifact: Path | None,
    psi: int | None,
    t: int | None,
) -> tuple[int | None, int | None]:
    if fit_artifact is None:
        return psi, t

    docs = [
        _load_json(fit_artifact / "config.json"),
        _load_json(fit_artifact / "metadata.json"),
    ]

    if psi is None:
        for doc in docs:
            psi = _recursive_find_int(doc, {"psi", "samples_per_partition"})
            if psi is not None:
                break

    if t is None:
        for doc in docs:
            t = _recursive_find_int(doc, {"t", "n_partitions", "partitions"})
            if t is not None:
                break

    return psi, t


def load_scaler(fit_artifact: Path) -> tuple[np.ndarray, np.ndarray]:
    path = fit_artifact / "scaler.npz"
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        mean_key = next(
            (k for k in ("mean_", "mean", "center_", "center") if k in keys),
            None,
        )
        scale_key = next(
            (k for k in ("scale_", "scale", "std_", "std") if k in keys),
            None,
        )
        if mean_key is None or scale_key is None:
            raise KeyError(
                f"Could not identify scaler mean/scale arrays in {path}; keys={sorted(keys)}"
            )
        mean = np.asarray(z[mean_key], dtype=np.float64).reshape(-1)
        scale = np.asarray(z[scale_key], dtype=np.float64).reshape(-1)

    return mean, scale


def trajectory_length_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trajectory_id": r["trajectory_id"],
                "group": r["group"],
                "n_observations": len(r["observations"]),
                "n_transitions": len(r["observations"]) - 1,
                "source_file": str(r["path"]),
            }
            for r in records
        ]
    )


def run_simple_baseline(
    records: list[dict[str, Any]],
    random_state: int,
) -> dict[str, float | int]:
    """Test whether crude raw-state summaries already separate the groups."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X: list[np.ndarray] = []
    y: list[int] = []

    for r in records:
        obs = r["observations"]
        features = np.concatenate(
            [
                np.mean(obs, axis=0),
                np.std(obs, axis=0),
                np.min(obs, axis=0),
                np.max(obs, axis=0),
                np.max(np.abs(obs), axis=0),
            ]
        )
        X.append(features)
        y.append(0 if r["group"] == "nominal" else 1)

    X_arr = np.stack(X)
    y_arr = np.asarray(y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=random_state),
    )

    accuracy = cross_val_score(model, X_arr, y_arr, cv=cv, scoring="accuracy")
    auc = cross_val_score(model, X_arr, y_arr, cv=cv, scoring="roc_auc")

    return {
        "n_trajectories": int(len(y_arr)),
        "n_summary_features": int(X_arr.shape[1]),
        "accuracy_mean": float(accuracy.mean()),
        "accuracy_std": float(accuracy.std()),
        "roc_auc_mean": float(auc.mean()),
        "roc_auc_std": float(auc.std()),
    }


def find_unit_trajectory_id_column(units: pd.DataFrame) -> str:
    for column in (
        "trajectory_id",
        "source_trajectory_id",
        "parent_trajectory_id",
        "episode_id",
    ):
        if column in units.columns:
            return column
    raise KeyError(
        "Could not infer trajectory ID column from units.parquet; "
        f"columns={list(units.columns)}"
    )


def normalize_trajectory_id(trajectory_id: str) -> str:
    """Normalize IDs created by mixed-dataset preparation.

    Example:
        source_0_traj_20260921T125254317443Z_f2878f35
    becomes:
        traj_20260921T125254317443Z_f2878f35
    """
    value = str(trajectory_id)
    while True:
        normalized = re.sub(r"^source_\d+_", "", value)
        if normalized == value:
            return value
        value = normalized


def label_embedding_units(
    units: pd.DataFrame,
    nominal_ids: set[str],
    failure_ids: set[str],
) -> np.ndarray:
    column = find_unit_trajectory_id_column(units)

    nominal_normalized = {normalize_trajectory_id(x) for x in nominal_ids}
    failure_normalized = {normalize_trajectory_id(x) for x in failure_ids}

    overlap = nominal_normalized & failure_normalized
    if overlap:
        raise ValueError(
            "Normalized trajectory IDs overlap between nominal and failure sets. "
            f"First overlaps: {sorted(overlap)[:10]}"
        )

    labels: list[str] = []
    unknown: list[tuple[str, str]] = []

    for trajectory_id in units[column].astype(str):
        normalized = normalize_trajectory_id(trajectory_id)

        if normalized in nominal_normalized:
            labels.append("nominal")
        elif normalized in failure_normalized:
            labels.append("failure")
        else:
            labels.append("unknown")
            unknown.append((trajectory_id, normalized))

    if unknown:
        raise ValueError(
            f"{len(unknown)} embedding rows could not be matched after trajectory-ID "
            f"normalization. First (original, normalized) IDs: {unknown[:10]}"
        )

    return np.asarray(labels)


def outside_mass_per_unit(
    embeddings,
    labels: np.ndarray,
    *,
    psi: int,
    t: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate missing occupancy mass per isolation partition."""
    if embeddings.shape[1] != psi * t:
        raise ValueError(
            f"embedding width={embeddings.shape[1]}, but psi*t={psi*t}"
        )

    rows: list[dict[str, Any]] = []

    for i in range(embeddings.shape[0]):
        dense = embeddings.getrow(i).toarray().ravel().reshape(t, psi)
        assigned_mass = dense.sum(axis=1)
        outside = np.clip(1.0 - assigned_mass, 0.0, 1.0)
        rows.append(
            {
                "embedding_row": i,
                "group": labels[i],
                "outside_mass_mean": float(outside.mean()),
                "outside_mass_median": float(np.median(outside)),
                "outside_mass_max": float(outside.max()),
                "partitions_with_any_outside": int(np.sum(outside > 1e-12)),
            }
        )

    per_unit = pd.DataFrame(rows)
    summary = (
        per_unit.groupby("group")
        .agg(
            outside_mass_mean_mean=("outside_mass_mean", "mean"),
            outside_mass_mean_std=("outside_mass_mean", "std"),
            outside_mass_mean_median=("outside_mass_mean", "median"),
            outside_mass_mean_min=("outside_mass_mean", "min"),
            outside_mass_mean_max=("outside_mass_mean", "max"),
            outside_mass_max_mean=("outside_mass_max", "mean"),
            partitions_with_any_outside_mean=("partitions_with_any_outside", "mean"),
        )
        .reset_index()
    )
    return per_unit, summary


def augmented_partition_probabilities(
    embedding_row: np.ndarray,
    *,
    psi: int,
    t: int,
) -> np.ndarray:
    blocks = embedding_row.reshape(t, psi)
    outside = np.clip(1.0 - blocks.sum(axis=1), 0.0, 1.0)[:, None]
    probs = np.concatenate([blocks, outside], axis=1)
    sums = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, sums, out=np.zeros_like(probs), where=sums > 0)
    return probs


def js_idk_divergence(
    a: np.ndarray,
    b: np.ndarray,
    *,
    psi: int,
    t: int,
) -> float:
    """Mean partition-wise Jensen-Shannon divergence, base 2."""
    pa = augmented_partition_probabilities(a, psi=psi, t=t)
    pb = augmented_partition_probabilities(b, psi=psi, t=t)

    divergences = np.empty(t, dtype=np.float64)
    for partition in range(t):
        # scipy returns sqrt(JS divergence); square to recover divergence.
        js_distance = jensenshannon(pa[partition], pb[partition], base=2.0)
        divergences[partition] = js_distance * js_distance

    return float(divergences.mean())


def pairwise_js_summary(
    embeddings,
    labels: np.ndarray,
    *,
    psi: int,
    t: int,
    max_per_group: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    nominal_idx = np.flatnonzero(labels == "nominal")
    failure_idx = np.flatnonzero(labels == "failure")

    if len(nominal_idx) > max_per_group:
        nominal_idx = rng.choice(nominal_idx, max_per_group, replace=False)
    if len(failure_idx) > max_per_group:
        failure_idx = rng.choice(failure_idx, max_per_group, replace=False)

    selected = np.concatenate([nominal_idx, failure_idx])
    dense = {
        int(i): embeddings.getrow(int(i)).toarray().ravel().astype(np.float64)
        for i in selected
    }

    rows: list[dict[str, Any]] = []

    def add_within(indices: np.ndarray, pair_type: str) -> None:
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i, j = int(indices[a]), int(indices[b])
                rows.append(
                    {
                        "pair_type": pair_type,
                        "row_i": i,
                        "row_j": j,
                        "js_divergence": js_idk_divergence(
                            dense[i], dense[j], psi=psi, t=t
                        ),
                    }
                )

    def add_between(a_idx: np.ndarray, b_idx: np.ndarray, pair_type: str) -> None:
        for i0 in a_idx:
            for j0 in b_idx:
                i, j = int(i0), int(j0)
                rows.append(
                    {
                        "pair_type": pair_type,
                        "row_i": i,
                        "row_j": j,
                        "js_divergence": js_idk_divergence(
                            dense[i], dense[j], psi=psi, t=t
                        ),
                    }
                )

    add_within(nominal_idx, "nominal-nominal")
    add_within(failure_idx, "failure-failure")
    add_between(nominal_idx, failure_idx, "nominal-failure")

    pairs = pd.DataFrame(rows)
    summary = (
        pairs.groupby("pair_type")["js_divergence"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    return pairs, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare nominal and failure CartPole datasets and IDK embeddings."
    )
    parser.add_argument("--nominal-dir", type=Path, required=True)
    parser.add_argument("--failure-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-artifact", type=Path)
    parser.add_argument("--embedding-artifact", type=Path)
    parser.add_argument("--psi", type=int)
    parser.add_argument("--t", type=int)
    parser.add_argument("--js-max-per-group", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    nominal_records = load_trajectories(args.nominal_dir, "nominal")
    failure_records = load_trajectories(args.failure_dir, "failure")
    all_records = nominal_records + failure_records

    nominal_ids = {r["trajectory_id"] for r in nominal_records}
    failure_ids = {r["trajectory_id"] for r in failure_records}
    overlap = nominal_ids & failure_ids
    if overlap:
        raise ValueError(
            f"Trajectory IDs appear in both groups: {sorted(overlap)[:10]}"
        )

    nominal_states = concatenate_states(nominal_records)
    failure_states = concatenate_states(failure_records)

    print(
        f"Loaded {len(nominal_records)} nominal and "
        f"{len(failure_records)} failure trajectories"
    )
    print(f"Raw states: nominal={nominal_states.shape}, failure={failure_states.shape}")

    lengths = trajectory_length_table(all_records)
    lengths.to_csv(args.output / "trajectory_lengths.csv", index=False)
    (
        lengths.groupby("group")[["n_observations", "n_transitions"]]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .to_csv(args.output / "trajectory_length_summary.csv")
    )

    raw_summary, raw_comparison = compare_state_populations(
        nominal_states, failure_states, space_name="raw_state"
    )
    raw_summary.to_csv(args.output / "raw_state_summary.csv", index=False)
    raw_comparison.to_csv(args.output / "raw_state_comparisons.csv", index=False)

    print("\n=== Raw-state comparison ===")
    print(
        raw_comparison[
            [
                "feature",
                "mean_nominal",
                "mean_failure",
                "cohens_d_failure_minus_nominal",
                "ks_statistic",
                "wasserstein_1d",
            ]
        ].to_string(index=False)
    )

    if args.fit_artifact is not None:
        mean, scale = load_scaler(args.fit_artifact)
        pd.DataFrame(
            {"dimension": np.arange(len(mean)), "mean": mean, "scale": scale}
        ).to_csv(args.output / "scaler_parameters.csv", index=False)

        if len(mean) == 4 and len(scale) == 4:
            nominal_z = (nominal_states - mean) / scale
            failure_z = (failure_states - mean) / scale
            z_summary, z_comparison = compare_state_populations(
                nominal_z, failure_z, space_name="standardized_state"
            )
            z_summary.to_csv(
                args.output / "standardized_state_summary.csv", index=False
            )
            z_comparison.to_csv(
                args.output / "standardized_state_comparisons.csv", index=False
            )
            print("\n=== Standardized-state comparison ===")
            print(
                z_comparison[
                    [
                        "feature",
                        "mean_nominal",
                        "mean_failure",
                        "cohens_d_failure_minus_nominal",
                        "ks_statistic",
                        "wasserstein_1d",
                    ]
                ].to_string(index=False)
            )
        else:
            print(
                f"\nScaler dimension is {len(mean)}, not 4. "
                "Skipping direct raw-state standardization comparison. "
                "This is expected for transition/state-action representations."
            )

    if not args.skip_baseline:
        baseline = run_simple_baseline(all_records, args.random_state)
        (args.output / "simple_baseline.json").write_text(
            json.dumps(baseline, indent=2), encoding="utf-8"
        )
        print("\n=== Simple non-IDK baseline ===")
        print(json.dumps(baseline, indent=2))

    if args.embedding_artifact is not None:
        embeddings_path = args.embedding_artifact / "embeddings.npz"
        units_path = args.embedding_artifact / "units.parquet"

        embeddings = load_npz(embeddings_path).tocsr()
        units = pd.read_parquet(units_path)

        if len(units) != embeddings.shape[0]:
            raise ValueError(
                f"units.parquet rows={len(units)} but embeddings rows={embeddings.shape[0]}"
            )

        labels = label_embedding_units(units, nominal_ids, failure_ids)
        psi, t = infer_psi_t(args.fit_artifact, args.psi, args.t)
        if psi is None or t is None:
            raise ValueError(
                "Could not infer psi/t. Pass --psi and --t explicitly or provide "
                "a fit artifact with readable config/metadata."
            )

        print(
            f"\nEmbedding matrix={embeddings.shape}; psi={psi}; t={t}; "
            f"expected width={psi*t}"
        )
        print(
            f"Embedding groups: nominal={np.sum(labels == 'nominal')}, "
            f"failure={np.sum(labels == 'failure')}"
        )

        outside_per_unit, outside_summary = outside_mass_per_unit(
            embeddings, labels, psi=psi, t=t
        )
        outside_per_unit.to_parquet(
            args.output / "outside_mass_per_unit.parquet", index=False
        )
        outside_summary.to_csv(
            args.output / "outside_mass_summary.csv", index=False
        )
        print("\n=== Outside-isolation-region mass ===")
        print(outside_summary.to_string(index=False))

        js_pairs, js_summary = pairwise_js_summary(
            embeddings,
            labels,
            psi=psi,
            t=t,
            max_per_group=args.js_max_per_group,
            random_state=args.random_state,
        )
        js_pairs.to_parquet(args.output / "js_pairwise.parquet", index=False)
        js_summary.to_csv(args.output / "js_summary.csv", index=False)
        print("\n=== Pairwise partition-wise JS divergence ===")
        print(js_summary.to_string(index=False))

    print(f"\nWrote outputs to: {args.output}")


if __name__ == "__main__":
    main()
