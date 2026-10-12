# Changelog

## [0.2.0] - 2026-10-08

### Added

- `WindowAnalytics`, `WindowScoringConfig`, and `cartpole-analyze windows` for current k-transition window scoring using a saved fit and raw trajectories. Sparse point features are cached and reused across window-size, stride, and metric sweeps, without requiring saved window embedding artifacts.
- Windowed IDK similarity, IDK distance, outside mass, cosine similarity, JS divergence, and explicitly smoothed KL divergence. Reference windows can occur anywhere in each reference trajectory, with independent strides, seeded sampling per trajectory, mean or top-k reduction, and source-aware self/trajectory/overlap exclusions.
- Per-window score tables, endpoint mean/median/IQR summaries across query trajectories, eligible/selected reference counts, exact reference-window selections, skipped-trajectory reports, and saved configuration/provenance. Plotting remains in notebooks, with usage examples in the README.
- `transition_action`, a 9D `[state, next_state, action]` representation using a scalar action, alongside the existing 10D one-hot `state_action_next_state` representation.

### Changed

- Pinned pyidk to the 0.2.1 implementation and delegated metric arithmetic, outside mass, window occupancy aggregation, and reference reduction to its APIs.
- Existing neighbor and trajectory-familiarity scoring now use blocked pyidk reference reduction. For `neighbors` and `rolling`, `--max-pairs` limits a working pairwise block instead of the total query-by-reference population.

## [0.1.4] - 2026-10-04

### Changed

Flattened nested click option/argument decorators for entrypoints

## [0.1.3] - 2026-09-27

### Added

- support.py for cartpole-analyze, for out of basis type tests
- Configurable uniform or normal initial states in `cartpole-generate`, with independent per-component parameters, seeded sampling, and provenance preserved through prepared and derived artifacts.
- Added the same initial-state options to `cartpole-train`, applied to training and evaluation episodes and saved in run configurations and checkpoint metadata.

## [0.1.2] - 2026-09-20

### Changed

- Separated IDK into `cartpole-fit`, `cartpole-embed`, and artifact-only `cartpole-analyze`, with validated Pydantic configuration, versioned/checksummed model and sparse embedding artifacts, and load/save APIs. Removed `cartpole-idk-evaluate`; use `cartpole-analyze neighbors` for top-k scoring.

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
