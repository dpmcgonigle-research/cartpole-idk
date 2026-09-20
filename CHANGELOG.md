# Changelog

## [0.1.2] - 2026-09-20

### Added

- Whole-trajectory and rolling-window IDK analytics: shared fitting, sparse pairwise metrics, reference neighbors and likeness, sklearn clustering and diagnostics, population RBF-MMD/permutations, reproducible artifacts, optional plots, and the `cartpole-analyze` CLI.

## [0.1.1] - 2026-09-16

### Added

- Typed Pydantic generation reports with return mean, standard deviation, min, max, and quartiles; migrated all CLI entrypoints to Click, centralized logging in `cartpole_idk.logging`, and replaced relative imports with absolute imports.

- Entrypoint for preparing datasets for fitting
  - "Random" n-length segments (so as to start in many different states)
  - n-length segments that start from timestep 0
  - Minimum length for (failure) episodes

## [0.1.0] - 2026-09-08

### Added
- DQN training harness.
- Periodic greedy evaluation and checkpoint metrics.
- Configurable trajectory generation.
- Action-delay, action-flip, and observation-bias perturbations.
- Normally distributed perturbation onset.
- NPZ trajectory storage and Parquet manifest.
- Queryable `TrajectoryStore`.
- Recorded-state replay.
- Thin `pyidk` integration.
- CLI entry points.
- Pytest tests and fixtures.
