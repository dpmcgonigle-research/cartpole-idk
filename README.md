# cartpole-idk

A research harness for studying Isolation Distributional Kernel (IDK) on CartPole
before moving to embodied robotics tasks.

This repository does **not** implement IDK. The generic implementation lives in the
separate `pyidk` package and is consumed here as a dependency.

## Overview

- [Overview](#overview)
- [Research Flow](#research-flow)
- [Installation](#installation)
- [Usage](#usage)
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
1. train policy, save checkpoints + evaluation metrics
2. generate / perturb / store trajectories
3. extract segments for fitting
4. scale and fit segments, save scaler + IDK basis artifact
5. embed trajectories
6. generate and analyze analytics, metrics, clustering
7. evaluate failure detection capabilities/potential
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

## Usage

### Train

CLI commands write timestamped progress messages to stderr. Training reports
periodic step counts, evaluation results, and saved checkpoints; generation
reports periodic episode counts and the final report location. Query tables and
IDK scores remain on stdout for piping or redirection.

```bash
cartpole-train   --run-dir runs/dqn_seed42   --total-steps 200000   --eval-every 5000   --eval-episodes 20   --checkpoint-every 10000   --hidden-sizes 128 128   --max-episode-steps 500   --seed 42
```

Training is measured in environment steps. Periodic evaluation runs are greedy
(`epsilon=0`) and do not update the model.

Each periodic evaluation writes mean/std return, median return, mean episode
length, `full_length_rate` (the fraction reaching `--max-episode-steps`), and
`p10_return`, `p25_return`, `p50_return`, `p75_return`, and `p90_return` to
`evaluation_metrics.csv`. Percentiles
use linear interpolation across evaluation episode returns; `p50_return` equals
`median_return`. The progress log includes the completion fraction and all five
percentiles. Checkpoint evaluations record the same metrics in checkpoint
metadata and logging.

Add `--debug-init-conds` to write `RUN_DIR/failure_init_conds.log`:

```bash
cartpole-train --run-dir runs/dqn_debug --debug-init-conds \
  --eval-every 5000 --eval-episodes 100 --max-episode-steps 500
```

The file appends a section for each periodic and checkpoint evaluation, labeled
with the training step, evaluation kind, timestamp, seed, episode count, and
maximum steps. Each episode with return **less than the configured maximum**
records its return, episode number, seed, and exact initial environment state
(`x`, `x_dot`, `theta`, `theta_dot`) before the first action. States retain the
sampled precision; angles are radians. Evaluations with no failures get an
explicit "No failures" entry. The flag is saved in training configuration and
checkpoint metadata; when omitted, no failure log is created.

Training accepts the same [initial-state options](#initial-state-sampling) as
generation. They apply to every training and evaluation episode, and are saved
under `initial_state` in `config.json` and the checkpoint's training configuration:

```bash
cartpole-train --run-dir runs/dqn_uniform --total-steps 200000 \
  --initial-distribution uniform \
  --initial-x-min -2.2 --initial-x-max 2.2 \
  --initial-theta-min -0.13962634 --initial-theta-max 0.13962634
```

Angles are radians (the example uses ±8 degrees); angular velocities are radians
per second. Training seeds the environment once and continues its RNG across
resets. Evaluation seeds each episode separately. Defaults remain uniform
`[-0.05, 0.05]`; exact runs for historical seeds change because sampling follows
Gymnasium's reset draw.

[Back to Top](#cartpole-idk)

### Generate Trajectories

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

#### Initial-state sampling

`cartpole-generate` samples each starting-state component independently in the
order `[x, x_dot, theta, theta_dot]` (position, velocity, pole angle, angular
velocity). Angles use radians. Uniform sampling is the default, with bounds
`[-0.05, 0.05]` for every component:

```bash
cartpole-generate --checkpoint runs/dqn_seed42/checkpoints/step_000200000.pt \
  --output datasets/generated/uniform-starts --episodes 100 \
  --initial-distribution uniform \
  --initial-x-min -1.0 --initial-x-max 1.0 \
  --initial-x-dot-min -0.5 --initial-x-dot-max 0.5 \
  --initial-theta-min -0.15 --initial-theta-max 0.15 \
  --initial-theta-dot-min -0.75 --initial-theta-dot-max 0.75
```

Normal sampling uses independent, untruncated Gaussians. Each component defaults
to mean `0.0` and standard deviation `0.05`; a standard deviation of zero fixes
that component at its mean. Samples are not clipped to CartPole's termination bounds.

```bash
cartpole-generate --checkpoint runs/dqn_seed42/checkpoints/step_000200000.pt \
  --output datasets/generated/normal-starts --episodes 100 \
  --initial-distribution normal \
  --initial-x-mean 0.0 --initial-x-std 0.5 \
  --initial-x-dot-mean 0.0 --initial-x-dot-std 0.25 \
  --initial-theta-mean 0.0 --initial-theta-std 0.05 \
  --initial-theta-dot-mean 0.0 --initial-theta-dot-std 0.25
```

Only the selected distribution's parameters are used and validated: uniform bounds
must satisfy `min < max`; normal standard deviations must be nonnegative. Parameters
must be finite. Sampling uses `env.unwrapped.np_random` after reset with
`seed + episode_index`, before any policy action or observation perturbation.
The TimeLimit wrapper remains active. Defaults preserve the old initialization
distribution, but exact rollouts for old seeds change because reset consumes its
own random draw before the custom sample is drawn.

`generation_report.json` records `initial_state`, containing the distribution and
its active parameters (for example, `parameters.x_min` for uniform sampling).
Each trajectory's existing metadata contains `initial_state_config` with the same
settings and `initial_state` with the actual sampled four-component state. True
observation arrays store its float32 representation. Prepared segments and later
fit/embedding/analysis unit metadata retain these fields as source provenance;
the recorded initial state still refers to the original rollout, even when a
prepared segment starts later. Reports and datasets without these fields remain
readable. No additional metadata files are introduced.

#### Perturbations

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

### Prepare Datasets

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

### Query

```bash
cartpole-query datasets/generated/action_delay   --perturbation action_delay   --max-return 250
```

[Back to Top](#cartpole-idk)

### Replay

```bash
cartpole-replay datasets/generated/action_delay --trajectory <trajectory-id>
```

Replay uses recorded true states rather than re-executing actions.

[Back to Top](#cartpole-idk)

### IDK Representations

The application adapter supports:

- `state`: 4D observations
- `transition`: 8D `[state, next_state]`
- `transition_action`: 9D `[state, next_state, action]`, with scalar action 0/1
- `state_action`: 5D `[state, action]`, with scalar action 0/1
- `state_action_next_state` (alias `state_action_state`): 9D `[state, action, next_state]`, with scalar action 0/1
- `window`: flattened temporal observation features

Normalization uses `pyidk.Standardizer`, fitted only on reference/training experience.
The action column is standardized along with the state columns. `--action-source`
selects commanded or executed actions; both use the same scalar encoding.

Raw and prepared trajectory stores already contain scalar actions and can be reused.
Existing 6D/10D one-hot action fits and embeddings must be regenerated with
`cartpole-fit` and `cartpole-embed`, followed by recomputing window analytics and
notebook results. Saved configurations now record `action_encoding: "scalar"`;
legacy one-hot artifacts are rejected rather than interpreted as scalar artifacts.
The IDK embedding width remains `t * psi`.

[Back to Top](#cartpole-idk)

### Fit

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
population. Neither embedding nor analytics fits a scaler or IDK basis. The
`windows` analysis command reads raw trajectories and caches point features from
the saved fit; other analysis commands use existing embedding artifacts.

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

### Embed

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

### Artifact Formats and Python APIs

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

### Analytics

#### Moving-window metrics

This integration requires **pyidk 0.2.1**, pinned in `pyproject.toml`. Metric
arithmetic, outside mass, window occupancy aggregation, and blocked reference
reduction are delegated to pyidk. CartPole supplies representations, trajectory
provenance, reference sampling/exclusions, and report tables.

Use `WindowAnalytics` with a saved fit and raw datasets. Each session applies the
saved representation, scaler, and point kernel once per distinct dataset/ID
selection, then reuses those sparse point features across window-size, stride,
and metric sweeps. Identical query/reference selections share a cache. You do
not need `cartpole-embed --mode window` for these scores. Whole-trajectory mean
embeddings alone cannot reconstruct window scores. Changing psi, t, the scaler,
or representation requires a new fit and session.

```python
from pathlib import Path
from cartpole_idk.artifacts import FitArtifact
from cartpole_idk.analytics import WindowAnalytics, WindowScoringConfig
from cartpole_idk.cli.selection import selection
from cartpole_idk.model import DatasetSelection

query_path = Path("datasets/generated/action_flip")
reference_path = Path("datasets/generated/nominal")
session = WindowAnalytics(
    FitArtifact.load("artifacts/fit"),
    DatasetSelection(dataset=query_path, trajectory_ids=selection(query_path, None)),
    {"reference": DatasetSelection(
        dataset=reference_path, trajectory_ids=selection(reference_path, None)
    )},
)
results = {}
for length in (10, 25, 50):
    results[length] = session.score(WindowScoringConfig(
        window_length=length,
        reference_stride=5,
        reference_windows_per_trajectory=20,
        random_state=42,
        metrics=("idk", "idk-distance", "outside-mass", "cosine", "js"),
        aggregation="mean",
    ))

scores = results[25].scores       # one row per query window / metric / reference group
summary = results[25].summary     # mean, median, q25, q75, n_windows, n_scored
results[25].save("analysis/window_25")
# In a notebook, group summary by metric/reference_group/source_dataset,
# plot mean or median against end_step, and shade between q25 and q75.
```

For every query interval `[start_step, end_step)`, `window_length` counts raw
**transitions**. The score becomes available at observation `end_step`. A
k-transition state window contains k+1 states; transition and action-based
representations contain k rows. Temporal representation length L contributes
k+2−L rows and requires k+2−L > 0. Only complete windows are emitted, with no
padding or crossing trajectory boundaries. `source_start_step` and
`source_end_step` retain original coordinates for prepared segments. Short
trajectories are recorded in `skipped`.

Reference windows have the **same transition length** as the query window, but
can start anywhere in each reference trajectory. No query/reference timestep
alignment is imposed. `reference_stride` defines candidate starts; an optional
`reference_windows_per_trajectory` cap samples without replacement using the
seed, then restores start order. The selected windows are available in
`reference_windows`, so a saved run records the exact reference bank.

- `aggregation="mean"` equally weights **all eligible reference windows**.
- `aggregation="top_k", k=5` averages the five best eligible windows in the
  entire reference bank (or fewer if available). IDK/cosine select the largest
  values; distance/JS/KL select the smallest. Ties use reference-bank order.
- Windows are equally weighted, so long trajectories contribute more when no
  cap is set. A cap limits this effect; short trajectories can still contribute
  fewer windows. The cap applies to each stored trajectory, including prepared
  segments; it does not enforce equal weighting of original sources.
- Exact same-source intervals are excluded by default. Use
  `exclude_same_trajectory=True` to exclude every reference from the original
  query trajectory, or `max_overlap=0` to exclude overlapping transition
  intervals from that source. Adjacent state windows can still share a boundary
  observation. These policies use source dataset/trajectory provenance and
  apply before reference reduction.
- No eligible references yields NaN and counts of zero. `outside-mass` needs
  no reference: it averages the query's missing occupancy across partitions,
  appears once with `reference_group="__basis__"`, and remains available even
  when no reference match is eligible. An all-zero embedding has outside mass
  one and can have distance zero to another all-zero embedding.
- Add `"kl"` to `metrics` with an explicit positive `epsilon` to calculate
  directional KL(query || reference), including the outside category.

The default summary groups by `source_dataset`, metric, reference group,
analysis identity, and local `end_step`. Its IQR is across the **per-query
scores**, after reference reduction; it is neither a spread over reference
matches nor a confidence interval. `n_scored` records the number of nonmissing
scores, which may decline as trajectories end. Overlapping windows are dependent.
For different metadata groups or original-source coordinates, call
`summarize_window_scores(scores, group_by=("perturbation_type",),
endpoint="source_end_step")`. Each stored query window receives equal weight.
Plotting remains in notebooks.

The CLI uses the same session and accepts repeated flags for a sweep:

```bash
cartpole-analyze windows --fit artifacts/fit \
  --dataset datasets/generated/action_flip --ids-file splits/test.txt \
  --reference nominal datasets/generated/nominal \
  --reference-ids nominal splits/reference.txt \
  --window-length 10 --window-length 25 --window-length 50 \
  --reference-stride 5 --reference-windows-per-trajectory 20 \
  --aggregation mean --output analysis/windows
```

Each `window_LENGTH` directory contains `scores.parquet`, `summary.parquet`,
`reference_windows.parquet`, `skipped.parquet`, and `metadata.json` with the fit
identity, selections, settings, and software versions. Input datasets must remain
unchanged for reproducibility. Query embeddings and pairwise work are batched;
the sparse point caches, selected reference bank, and returned tables remain in
memory. Use strides/reference caps to control their size. The older `rolling`
command continues to support previously saved window embedding artifacts.

#### Metrics

The pairwise metrics compare **individual trajectories**, each represented by one
IDK vector summarizing its state occupancy. For the state representation, write a
recorded trajectory as $\tau = (x_0,\ldots,x_{N-1})$, where each $x_i$ is an observed
state vector and $N$ is the number of state observations. In CartPole,
$x_i = (x,\dot{x},\theta,\dot{\theta})$.

In distribution notation, $\mu_P = \mathbb{E}_{X \sim P}[\phi(X)]$, where $X$ is a
random state drawn from $P$ and $\phi$ contains the isolation-region features.
For a particular trajectory, $P_\tau$ is its empirical state distribution: drawing
from $P_\tau$ means choosing one of its observed states uniformly. Its embedding is
$\mu_\tau = \mu_{P_\tau} = \frac{1}{N}\sum_{i=0}^{N-1}\phi(x_i)$.
Each coordinate is the fraction of observations assigned to a particular isolation
region. All trajectories use the same fitted feature map, with $t$ partitions.

Thus, comparing trajectories $\tau$ and $\eta$ compares **where their states occur
and how frequently they occupy those regions**. For example, in one partition,
occupancies $(0.8,0.2)$ and $(0.75,0.25)$ have a smaller IDK distance than
$(0.8,0.2)$ and $(0.2,0.8)$. Averaging discards ordering: trajectories with the same
state observations in different orders have identical state embeddings. A prepared
segment or window is summarized in the same way; transition representations instead
average features of transitions.

For JS and KL, $p_r$ and $q_r$ are the region-occupancy probabilities of trajectories
$\tau$ and $\eta$ in partition $r$, including an outside-region category, and
$m_r = (p_r + q_r)/2$.
The index $j$ identifies a category within partition $r$: $j=1,\ldots,\psi$
enumerates its $\psi$ isolation regions, and $j=\psi+1$ denotes the outside-region
category. Thus, $p_{rj}$ is the fraction of trajectory $\tau$'s observations in
category $j$ of partition $r$, and $q_{rj}$ is the corresponding fraction for
trajectory $\eta$. KL sums over all $\psi+1$ categories, then averages over the
$t$ partitions.

For population comparisons, $U,U' \sim \mathcal{P}$ and $V,V' \sim \mathcal{Q}$
are independent draws of IDK embedding vectors; the Gaussian kernel is
$k_\sigma(u,v) = \exp(-\lVert u-v\rVert^2/(2\sigma^2))$, with $\sigma > 0$.

| Measure | Formula | Interpretation | Drawback |
| --- | --- | --- | --- |
| IDK similarity (`idk`) | $K(\tau,\eta) = \langle\mu_\tau,\mu_\eta\rangle/t$ | Overlap in two trajectories' state occupancy; a positive-semidefinite kernel on their empirical distributions. Inner product scaled by $1/(\# partitions$) | Depends on occupancy concentration as well as overlap; self-similarity need not be one. |
| IDK-induced distance (`idk-distance`) | $d(\tau,\eta) = \lVert\mu_\tau-\mu_\eta\rVert/\sqrt{t}$ | Difference between two trajectories' occupancy fractions; the Euclidean distance scaled by $1/\sqrt(\# partitions)$. | Distinct trajectories or state distributions can share an embedding, so zero distance does not establish that trajectories are identical. |
| Cosine similarity (`cosine`) | $\langle\mu_\tau,\mu_\eta\rangle/(\lVert\mu_\tau\rVert\lVert\mu_\eta\rVert)$ | Angular agreement between trajectory occupancy vectors, invariant to positive rescaling. | Discards magnitude and cannot distinguish proportional embeddings; undefined for zero vectors without a convention. |
| Jensen–Shannon divergence (`js`) | $\frac{1}{2t}\sum_{r=1}^{t}[D_{\mathrm{KL}}(p_r\parallel m_r)+D_{\mathrm{KL}}(q_r\parallel m_r)]$ | Average information that region membership provides about which of the two equally likely trajectories supplied a state. | Captures only distinctions retained by the partitions; within-region differences remain invisible. |
| Kullback–Leibler divergence (`kl`) | $\frac{1}{t}\sum_{r=1}^{t}\sum_{j=1}^{\psi+1} p_{rj}\log(p_{rj}/q_{rj})$ | Average excess logarithmic loss from describing trajectory $\tau$'s region occupancy with trajectory $\eta$'s probabilities. | Asymmetric and potentially infinite at zero reference probabilities; smoothing changes the compared distributions. |

#### Basis support

`outside_mass` measures occupancy outside all learned isolation regions in each
partition. It is a basis-support diagnostic, not a calibrated probability or a
similarity metric. For each partition, it is `1 - sum(cell occupancies)`; the
cells are never renormalized. Low outside mass means most of the trajectory/window
is represented by learned regions; high outside mass means much of it falls
outside the fitted basis support. There is no universal OOD cutoff: thresholds
should later be calibrated using held-out nominal data.

```bash
cartpole-analyze support artifacts/embeddings/test --output analysis/support
cartpole-analyze support artifacts/embeddings/test \
  --output analysis/support-by-source --group-by source_dataset
```

The artifact supplies `t` and `psi`; neither the scaler nor the basis is refitted.
Any existing scalar metadata column can be used for grouping, with missing values
retained as a separate group. Output directories must be new or empty.

- `support.parquet`: one row per unit in artifact order, retaining unit/source IDs,
  local and available source window offsets, and existing metadata. Scores are
  `outside_mass_mean`, `outside_mass_median`, `outside_mass_min`, `outside_mass_max`,
  `outside_mass_std`, `partitions_with_any_outside`, and `support_rate_mean`.
- `support_partitions.parquet`: `embedding_row`, `unit_id`, zero-based `partition`,
  and `outside_mass`, ordered by unit then partition.
- `support_summary.json`: `n_units`, `t`, `psi`, and `statistics` for each score
  (mean, population std-dev, min, max, q1, median, q3 across units).
- With `--group-by`, `support_group_summary.parquet`: one row per group and score,
  containing `group_by`, `group_value`, `n_units`, `metric`, and the same statistics.
- Standard `config.json`, `metadata.json`, and `units.parquet` preserve analysis provenance.

Per-unit standard deviations also use the population definition. “Any outside”
counts partitions with strictly positive missing mass. Only out-of-range floating
point excursions up to `1e-10` are clipped; larger invalid occupancies are rejected.
Summaries weight each unit equally, so overlapping windows are not independent
trajectory observations. Per-partition values remain available for later calibration
or temporal analysis; this command does not perform either.

#### Clustering and population comparison

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

All commands use Click. Analyses of saved embeddings include typed `config.json`, input hashes
and role ordering in `metadata.json`, unit tables, and command-specific results.
Pairwise matrices have `pairwise_axes.json`; embeddings and bases are referenced,
not copied or regenerated. The default `--max-pairs 4000000` limits each dense
pairwise result (32 MB per array, with additional intermediates possible). MMD
applies this to the pooled matrix. Change selections/stride during embedding or
explicitly raise the analysis limit. For `neighbors` and `rolling`, `--max-pairs`
now limits a working block rather than rejecting a large total population:
pyidk reduces references without allocating the full pairwise matrix. The new
`windows` command controls blocks with `--query-batch-size` and
`--reference-block-size`. Embedding storage remains sparse.

Optional plotting helpers remain in `cartpole_idk.visualization.analytics` and
require matplotlib. Plots do not replace quantitative cluster diagnostics.

[Back to Top](#cartpole-idk)

## Tests

```bash
make test
```

or

```bash
pytest
```

[Back to Top](#cartpole-idk)
