# cartpole-idk

A research harness for studying Isolation Distributional Kernel (IDK) on CartPole
before moving to embodied robotics tasks.

This repository does **not** implement IDK. The generic implementation lives in the
separate `pyidk` package and is consumed here as a dependency.

## Overview

- [Overview](#overview)
- [Research Flow](#research-flow)
- [Installation](#installation)
- [Train](#train)
- [Generate Trajectories](#generate-trajectories)
- [Prepare Datasets](#prepare-datasets)
- [Stored Trajectory Semantics](#stored-trajectory-semantics)
- [Query](#query)
- [Replay](#replay)
- [IDK Representations](#idk-representations)
- [Fit](#fit)
- [Embed](#embed)
- [Artifact Formats and Python APIs](#artifact-formats-and-python-apis)
- [Analytics](#analytics)
- [Tests](#tests)

[Back to Top](#cartpole-idk)

## Research Flow

```text
train policy
    -> save checkpoints + evaluation metrics
    -> generate nominal / perturbed trajectories
    -> store + index + replay trajectories
    -> cartpole-prepare: extract segments for fitting
    -> cartpole-fit: save scaler + IDK basis artifact
    -> cartpole-embed: save whole/window embeddings
    -> cartpole-analyze: metrics, retrieval, clustering, populations
    -> evaluate failure detection / lead time
```

[Back to Top](#cartpole-idk)

## Installation

Python 3.11+ and Git are required. `pyidk` is installed automatically from
GitHub over HTTPS, pinned to a commit for reproducibility.

Create the development environment:

```bash
make env
source .venv/bin/activate
```

To include notebook dependencies:

```bash
python -m pip install -e ".[dev,nb]"
```

For development on a sibling `pyidk` checkout, run
`python -m pip install -e ../pyidk` after installing this project. Reinstalling
this project may restore the pinned GitHub version.

[Back to Top](#cartpole-idk)

## Train

CLI commands write timestamped progress messages to stderr. Training reports
periodic step counts, evaluation results, and saved checkpoints; generation
reports periodic episode counts and the final report location. Query tables and
IDK scores remain on stdout for piping or redirection.

```bash
cartpole-train   --run-dir runs/dqn_seed42   --total-steps 200000   --eval-every 5000   --eval-episodes 20   --checkpoint-every 10000   --hidden-sizes 128 128   --max-episode-steps 500   --seed 42
```

Training is measured in environment steps. Periodic evaluation runs are greedy
(`epsilon=0`) and do not update the model.

[Back to Top](#cartpole-idk)

## Generate Trajectories

Nominal:

```bash
cartpole-generate   --checkpoint runs/dqn_seed42/checkpoints/step_000200000.pt   --output datasets/generated/nominal   --episodes 100
```

Action delay:

```bash
cartpole-generate   --checkpoint runs/dqn_seed42/checkpoints/step_000200000.pt   --output datasets/generated/action_delay   --episodes 100   --perturbation action-delay   --onset-mean 50   --onset-std 5   --delay-steps 3
```

Action flip:

```bash
cartpole-generate   --checkpoint runs/dqn_seed42/checkpoints/step_000200000.pt   --output datasets/generated/action_flip   --episodes 100   --perturbation action-flip   --onset-mean 75   --onset-std 8   --flip-probability 0.5
```

Observation bias:

```bash
cartpole-generate   --checkpoint runs/dqn_seed42/checkpoints/step_000200000.pt   --output datasets/generated/angle_bias   --episodes 100   --perturbation observation-bias   --feature pole_angle   --onset-mean 60   --onset-std 5   --bias 0.24   --noise-std 0.01   --ramp-steps 15
```

Generation writes `generation_report.json` with per-trajectory records and
`return_statistics`: `mean`, `std_dev` (population standard deviation), `min`,
`max`, `q1`, `median`, and `q3` (linearly interpolated quartiles). Statistics are
`null` when no episodes are generated. Load reports as typed objects with
`GenerationReport.from_file(path)` from `cartpole_idk.model`; older reports
without statistics are supported.

### Perturbations

Perturbations introduce controlled changes to actions or observations during an
episode to study how familiarity and failure detection respond. Nominal generation
uses `--perturbation none` (the default).

Each perturbed episode samples a zero-based onset step from a normal distribution
with `--onset-mean` and `--onset-std`, rounds it to an integer, and clamps it to
`0` through `--max-episode-steps - 1`. Set `--onset-std 0` for a fixed onset.
Behavior is nominal before onset; episodes that end earlier never experience the
perturbation.

- **Action delay** (`action-delay`): executes commands `--delay-steps` steps late
  after onset, holding the onset command while the delay queue fills.
- **Action flip** (`action-flip`): swaps the commanded action (`0` ↔ `1`) with
  `--flip-probability` at each step from onset onward.
- **Observation bias** (`observation-bias`): adds `--bias` to the selected
  `--feature` in the policy's observation, leaving the true state unchanged.
  Features are `cart_position`, `cart_velocity`, `pole_angle`, and
  `pole_angular_velocity`. `--ramp-steps` linearly increases the bias to full
  strength over that many steps, starting at onset; `0` applies it immediately.
  `--noise-std` adds zero-mean Gaussian noise to that feature from onset onward;
  the noise is not ramped.

The sampled onset and perturbation parameters are saved in trajectory metadata.

[Back to Top](#cartpole-idk)

## Prepare Datasets

Prepare one contiguous segment per eligible generated trajectory for IDK fitting:

```bash
# Randomly sample segments from different episode parts
cartpole-prepare datasets/generated/nominal --output datasets/prepared/nominal --seed 42

# Combine multiple generated datasets:
cartpole-prepare datasets/generated/nominal datasets/generated/action_delay --output datasets/prepared/combined
# Start every segment at timestep zero:
cartpole-prepare datasets/generated/nominal --output datasets/prepared/from_zero --episode-start 0
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--episode-length` | 100 | Maximum segment length in transitions |
| `--min-episode-length` | 25 | Skip segments shorter than this |
| `--success-threshold` | 300 | Source lengths at or above this are successes |
| `--success-buffer` | 50 | Exclude this many steps before the success threshold |
| `--episode-start` | None | Fixed start for all segments; otherwise sample successes and take failure tails |
| `--seed` | 1000 | Seed for reproducible uniform sampling |

Successful segments always have `episode-length` transitions. Their start is sampled
uniformly from the integers `0` through
`success-threshold - success-buffer - episode-length`, inclusive. With defaults,
starts range from 0 to 150, and each segment fits within the boundary `[0, 250]`.
Failures use their final `episode-length` transitions, or the entire trajectory if
shorter, provided they meet the minimum length.

An explicit `--episode-start` overrides start selection for both successes and
failures. Failures ending before a full segment is available retain the remaining
steps if they meet the minimum; otherwise they are skipped. Fixed starts must
still allow a full successful segment inside the buffered boundary. Invalid
lengths, buffers, and starts are rejected.

Pass one or more input directories before `--output`. All inputs use the same
sampling options and contribute to a single output dataset. Sampling is reproducible
for the same seed and input order. Repeated source directories are rejected.
For multiple inputs, output trajectory IDs are prefixed with `source_<index>_`
(zero-based input order) to prevent collisions; the original ID and dataset path
remain in each segment's metadata. Single-input IDs remain unchanged.

The output must be new or empty. It uses the same NPZ/Parquet format as generated
datasets and can be passed directly to the IDK fitting/evaluation code. Observations
retain the final next state (`T+1` observations for `T` transitions). The manifest
records source IDs, source lengths/returns, success labels, and segment offsets
(`segment_stop` is exclusive for transitions). Perturbation onsets are adjusted to
segment-local coordinates; negative onsets mean the perturbation was already active.
`preparation_report.json` records options and saved/skipped counts.

[Back to Top](#cartpole-idk)

## Stored Trajectory Semantics

Each `.npz` stores:

- `true_observations`: environment state
- `agent_observations`: state shown to policy
- `commanded_actions`: policy output
- `executed_actions`: action sent to environment
- `rewards`
- `terminated`
- `truncated`

Observation arrays have length `T+1`; action/reward arrays have length `T`.

A Parquet manifest indexes trajectory metadata.

[Back to Top](#cartpole-idk)

## Query

```bash
cartpole-query datasets/generated/action_delay   --perturbation action_delay   --max-return 250
```

[Back to Top](#cartpole-idk)

## Replay

```bash
cartpole-replay datasets/generated/action_delay --trajectory <trajectory-id>
```

Replay uses recorded true states rather than re-executing actions.

[Back to Top](#cartpole-idk)

## IDK Representations

The application adapter supports:

- state
- transition
- state-action
- state-action-next-state
- temporal window

Normalization uses `pyidk.Standardizer`, fitted only on reference/training experience.

[Back to Top](#cartpole-idk)

## Fit

IDK work is split into three explicit, reusable stages:

```text
trajectory dataset -> cartpole-fit -> fit artifact
trajectory dataset + fit artifact -> cartpole-embed -> embedding artifact
embedding artifacts -> cartpole-analyze -> analysis results
```

**Fit population** selects the observations used to construct the standardizer and
isolation basis. **Embedded populations** are transformed with that saved model.
**Analysis/reference populations** are selected embedding artifacts used for
comparison. Choose nominal-only, joint nominal/failure, or another explicit fit
population. Neither embedding nor analytics fits a scaler or IDK basis. Analytics
also never loads raw trajectories or generates embeddings.

```bash
cartpole-fit datasets/experiment \
  --ids-file splits/fit.txt --representation transition \
  --psi 32 --t 200 --random-state 42 --output artifacts/fit
```

ID files contain one trajectory ID per line; blank lines and `#` comments are
ignored. Omit `--ids-file` to select all trajectories in manifest order. Provide
explicit disjoint fit/test selections for held-out experiments. Labels are not
features and do not enter fitting or clustering.

Fit options include representation, observation source (`true` or `agent`), action
source (`commanded` or `executed`), `--representation-window-length`, `--psi`, `--t`,
and `--random-state`. Fit defaults to whole trajectories; `--mode window`,
`--window-length`, and `--stride` allow fitting on a selected window population.
Overlapping fitting windows weight repeated observations multiple times.

To recover basis centroid values in original feature units, use
`original = standardized * scale + mean`, where `standardized` is `centers`
from `basis.npz`, and `scale` and `mean` come from `scaler.npz`. Apply this
per feature across all partitions and centers.

[Back to Top](#cartpole-idk)

## Embed

```bash
cartpole-embed --fit artifacts/fit --dataset datasets/experiment \
  --ids-file splits/test.txt --mode window --window-length 25 --stride 5 \
  --output artifacts/embeddings/test

cartpole-embed --fit artifacts/fit --dataset datasets/experiment \
  --ids-file splits/reference.txt --mode window --window-length 25 --stride 5 \
  --output artifacts/embeddings/reference
```

Representation, standardization, and IDK parameters come exclusively from the fit
artifact. They cannot be overridden during embedding or analysis. Whole mode
produces one embedding per trajectory. Window mode retains complete contiguous
windows only, never crossing an episode boundary or padding a short episode.
A 25-step unit contains 25 transitions and 26 boundary observations, preserving
the repository's `T+1` convention. A 100-step trajectory yields 76 windows when
length is 25 and stride is 1.

Representation-window length and analysis-unit length are independent. A unit
with 25 transitions produces 26 state rows, 25 transition/state-action rows, or
`27-L` temporal representation rows for representation length `L`, if positive.
Too-short units and empty representations are recorded in artifact metadata;
all-empty populations raise an error.

Unit provenance includes IDs, whole/window mode, half-open step intervals,
episode return/length, checkpoint, perturbation parameters and onset, and time to
termination/failure where recorded endpoint flags make it known. Prepared prefixes
without endpoint flags remain censored. Original source offsets are retained.
Unit IDs include a source-dataset namespace to avoid collisions across artifacts.

[Back to Top](#cartpole-idk)

## Artifact Formats and Python APIs

```text
fit_artifact/                     embedding_artifact/
    config.json                      config.json
    scaler.npz                       units.parquet
    basis.npz                        embeddings.npz
    fit_manifest.parquet             metadata.json
    metadata.json
```

Fit arrays store scaler means/scales and isolation centers, radii, sample indices,
and distance metric. Embeddings are CSR matrices saved by SciPy. No pickle is used.
The unit tables have explicit ordered row indices, IDs, mode, start/end steps,
raw length, and `metadata_json` preserving nested episode/perturbation metadata.
Loaded units do not require or contain raw trajectory arrays.

Configurations are frozen Pydantic v2 models (`FitConfig`, `EmbedConfig`,
`IDKConfig`, `WindowConfig`, `AnalysisConfig`, and analysis-specific models) in
`cartpole_idk.model`. Use `model_dump(mode="json")`, `model_validate_json(...)`,
and `model_json_schema()` for serialization, loading, and schemas. Unknown fields,
invalid numeric settings, missing KL smoothing, and incompatible role options
are rejected.

Metadata records format version 1, package versions, dimensionality, skipped units,
file SHA-256 checksums, and a content identity. Embedding configuration includes
its parent fit ID and complete fit configuration/provenance. Copying an artifact
preserves its identity; altered files fail integrity validation. Comparisons
require the same parent fit hash and compatible representation metadata.
Directories must be new or empty; artifacts are staged and published atomically.

```python
from cartpole_idk.artifacts import FitArtifact, EmbeddingArtifact
from cartpole_idk.analytics import pairwise, top_k_neighbors

fit = FitArtifact.load("artifacts/fit")  # reconstructs pyidk without fitting
query = EmbeddingArtifact.load("artifacts/embeddings/test")
reference = EmbeddingArtifact.load("artifacts/embeddings/reference")
query.compatible_with(reference)
distances = pairwise(query.embeddings, reference.embeddings, metric="idk-distance")
retrieval = top_k_neighbors(query.embeddings, reference.embeddings, k=5)
# retrieval.neighbors contains individual matches; retrieval.scores contains means.
```

`fit_dataset(FitConfig)` and `embed_dataset(FitArtifact, EmbedConfig)` live in
`cartpole_idk.idk.pipeline` and return artifacts with `.save(path)` methods.
Save a newly fitted model before embedding so its content identity is established.
The saved embedding artifact is sufficient for analytics: its source dataset and
parent fit directory can be offline. Raw datasets must remain unchanged when
repeating the fitting/embedding stages themselves.

[Back to Top](#cartpole-idk)

## Analytics

### Metrics and neighbors

| Metric | Meaning | Nearest |
| --- | --- | --- |
| `idk` | Native `pyidk` dot product divided by `t` | Highest |
| `idk-distance` | `sqrt(max(0, Kxx + Kyy - 2Kxy))` | Lowest |
| `cosine` | Dot product divided by embedding norms | Highest |
| `js` | Mean partition JS divergence, including outside occupancy, in nats | Lowest |
| `kl` | Smoothed directional `KL(query || reference)`, in nats | Lowest |

Native similarity is not cosine, and self-similarity need not equal one. Cosine
returns zero for comparisons involving a zero vector. JS retains an outside-region
cell and exposes divergence, not its square root. KL requires `--epsilon > 0`;
smoothing applies to all cells, including outside, before renormalization.
These values are not probabilities or calibrated uncertainty.

Neighbors default to `k=5`, exclude self matches, and retain individual references
alongside aggregate scores. `--exclude-same-trajectory` and `--max-overlap` restrict
trivial rolling-window matches. Overlap is intersection divided by shorter length.
No eligible neighbors yields count zero and a missing score. Without `--reference`,
the input artifact itself is the reference library.

Supply both `--nominal-reference` and `--failure-reference` embedding artifacts to
retain separate likeness scores. Margin is nominal minus failure for similarities,
and failure minus nominal for distances, so positive means more nominal-like.
Rolling analysis selects existing windows by `--query-id`; it rejects whole-mode
artifacts and never creates new windows.

```bash
cartpole-analyze pairwise artifacts/embeddings/test \
  --metric idk-distance --output analysis/pairwise

cartpole-analyze neighbors artifacts/embeddings/test \
  --reference artifacts/embeddings/reference --metric cosine --k 5 \
  --exclude-same-trajectory --output analysis/neighbors

cartpole-analyze rolling artifacts/embeddings/test --query-id TRAJECTORY_ID \
  --nominal-reference artifacts/embeddings/nominal \
  --failure-reference artifacts/embeddings/failure --output analysis/rolling
```

Pairwise also accepts `--reference` for a rectangular query-versus-reference matrix.

### Clustering and population comparison

Clustering supports sklearn HDBSCAN (default), DBSCAN, spectral clustering on native
IDK affinity, and sparse SVD followed by a DP-style Bayesian Gaussian mixture.
Density methods use IDK distance or `sqrt(mean partition JS)` via `--distance js`.
Choose `--n-clusters` for spectral, or `--svd-components`/`--n-components` for mixtures.
Outputs include cluster labels/noise, estimator-specific strengths or posteriors,
return statistics, categorical composition, AMI/purity, and silhouette for density
methods with a valid distance matrix. Noise is excluded from evaluation by default;
undefined silhouette is null. Clustering remains unsupervised.

Population RBF-MMD compares populations of IDK vectors using
`exp(-||mu_i-mu_j||²/(2*sigma²))`. The default biased MMD squared includes diagonal
terms. `--estimator unbiased` excludes them, needs two units per group, and may be
negative. Default sigma is the median positive pooled Euclidean distance (1 if
all coincide); `--bandwidth` overrides it. Actual bandwidth is reported and fixed
through seeded permutations, whose p-value is `(extreme+1)/(B+1)`.
Permutation inference requires exchangeable rows; overlapping windows are generally
dependent. Use independent trajectories or an appropriate independent sampling
scheme for inference. No automatic blocked permutation scheme is imposed.

```bash
cartpole-analyze cluster artifacts/embeddings/test --algorithm hdbscan \
  --min-cluster-size 20 --distance idk-distance --output analysis/clusters

cartpole-analyze population artifacts/embeddings/nominal \
  --group-b artifacts/embeddings/failure --permutations 1000 \
  --output analysis/population
```

All commands use Click. Analytics outputs include typed `config.json`, input hashes
and role ordering in `metadata.json`, unit tables, and command-specific results.
Pairwise matrices have `pairwise_axes.json`; embeddings and bases are referenced,
not copied or regenerated. The default `--max-pairs 4000000` limits each dense
pairwise result (32 MB per array, with additional intermediates possible). MMD
applies this to the pooled matrix. Change selections/stride during embedding or
explicitly raise the analysis limit. Embedding storage remains sparse.

Optional plotting helpers remain in `cartpole_idk.visualization.analytics` and
require matplotlib. Plots do not replace quantitative cluster diagnostics.

### Migration

`cartpole-idk-evaluate` has been removed. Its top-k scoring behavior is provided by
`cartpole-analyze neighbors` after explicit fitting and embedding. The old
`cartpole-analyze DATASET --fit-ids-file ...` interface is intentionally removed.
Move fit settings to `cartpole-fit`, unit settings to `cartpole-embed`, and supply
embedding artifact directories to analysis. Prior analysis-output directories
lack the new artifact contracts; regenerate them with `cartpole-embed`.

Existing low-level `fit_reference()` remains an explicit in-memory fitting helper
in `idk`; no analytics entrypoint calls it. Raw-unit construction moved from
`analytics.units` to `idk.units`; fit/embed orchestration lives in `idk.pipeline`.

[Back to Top](#cartpole-idk)

## Tests

```bash
pytest
```

[Back to Top](#cartpole-idk)
