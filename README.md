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
- [Analytics](#analytics)
- [Tests](#tests)

[Back to Top](#cartpole-idk)

## Research Flow

```text
train policy
    -> save checkpoints + evaluation metrics
    -> generate nominal / perturbed trajectories
    -> store + index + replay trajectories
    -> build pyidk representations
    -> fit nominal IDK reference data
    -> measure familiarity
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

## Analytics

`cartpole-analyze` analyzes whole trajectories or fixed-length subtrajectories in
one shared IDK feature space. Three population roles remain separate:

- **Fit population:** constructs the standardizer and isolation basis. Select
  nominal-only, joint nominal/failure, or another explicit population.
- **Analysis population:** transformed with the frozen scaler and basis. No query
  values or labels are used to refit either object.
- **Reference library:** embedded experiences searched for likeness. This may
  differ from both fitting and analysis data.

Use ID files containing one trajectory ID per line (blank lines and `#` comments
are ignored). `--fit-ids-file` and `--analysis-ids-file` default to all dataset
trajectories; `--reference-ids-file` defaults to the fit IDs. These defaults are
exploratory, **not held-out evaluation**. Provide disjoint files for held-out
experiments. Select fitting populations explicitly; labels never enter the kernel
or unsupervised clustering.

### Units and representations

`--mode whole` makes one distribution embedding per episode. `--mode window`
uses `--window-length` raw transitions and `--stride`, retaining complete windows
only. A 25-step unit covers transitions `[start, start+25)` and includes 26
observations, exactly preserving this repository's `T+1` semantics. Thus a
100-step trajectory has 76 windows at length 25 and stride 1. No unit crosses an
episode boundary.

Representation is independent of unit construction. `--representation` accepts
`state`, `transition`, `state_action`, `state_action_next_state`, or `window`.
`--representation-window-length` controls the last representation's observation
window, separately from the analysis unit length. A 25-transition unit yields
26 state rows, 25 transition/state-action rows, or `27-L` temporal feature rows
for representation length `L` (if positive). Too-short units are recorded in
`skipped_units.parquet`; a population with no usable units raises an error.
`--observation-source true|agent` and `--action-source commanded|executed` use
the existing feature adapter.

Provenance includes source IDs, step intervals, original episode length/return,
checkpoint and perturbation metadata, onset, time since onset at the window end,
and time remaining to termination/failure when recorded flags establish an endpoint.
Prepared segments without endpoint flags are censored: time-to-termination is
unknown, not inferred from a short stored segment. Nested metadata is JSON-encoded
in Parquet columns. Success labels are retained when available, never inferred
from length by analytics.

### Pairwise metrics

| CLI metric | Python function | Meaning | Nearest |
| --- | --- | --- | --- |
| `idk` | `pairwise_idk_similarity` | Native `pyidk` dot product divided by `t` | Highest |
| `idk-distance` | `pairwise_idk_distance` | `sqrt(max(0, Kxx + Kyy - 2Kxy))` | Lowest |
| `cosine` | `pairwise_cosine` | Dot product divided by embedding norms | Highest |
| `js` | `pairwise_js_divergence` | Mean per-partition JS divergence, in nats | Lowest |
| `kl` | `pairwise_kl_divergence` | Mean smoothed `KL(query || reference)`, in nats | Lowest |

Native IDK similarity is not cosine, and its diagonal need not be one. Cosine
removes norm information; comparisons involving any zero vector return zero.
JS and KL add an explicit outside-region cell to each partition, preserving
unassigned mass. JS is **divergence**, not its square-root distance. KL requires
`--epsilon` greater than zero, adds that value to every cell including outside,
then renormalizes; it is asymmetric and optional. None of these values is a
probability or calibrated uncertainty.

Low-level functions accept sparse matrices and `t`; callers must supply the same
basis. The higher-level `pairwise(EmbeddingSet, ...)` enforces basis identity and
uses the small `METRICS` registry for direction and dispatch.

```python
from cartpole_idk.analytics import build_units, fit_embeddings, pairwise, top_k_neighbors
from cartpole_idk.idk import IDKExperimentConfig
from cartpole_idk.storage import TrajectoryStore

store = TrajectoryStore("datasets/experiment")
fit_units = build_units([store.get(tid) for tid in fit_ids], mode="window", window_length=25)
basis = fit_embeddings(fit_units, IDKExperimentConfig(representation="transition", psi=32, t=200))
query_units = build_units([store.get(tid) for tid in query_ids], mode="window", window_length=25)
queries = basis.transform(query_units)
distances = pairwise(queries, basis.fitting, metric="idk-distance")
retrieval = top_k_neighbors(queries, basis.fitting, metric="idk-distance", k=5)
# retrieval.neighbors retains individual matches; retrieval.scores has per-query means.
```

### Neighbors and rolling likeness

`top_k_neighbors` defaults to `k=5`, excludes exact self matches, and caps k at
eligible reference count. Ties preserve reference order. No eligible references
produces count zero and a missing score. `--exclude-same-trajectory` and
`--max-overlap` control trivial within-episode retrieval; overlap is intersection
length divided by the shorter interval, using original source coordinates.

Supply both `--nominal-reference-ids-file` and `--failure-reference-ids-file` to
retain separate component scores and neighbor tables. The margin is nominal minus
failure for similarities and failure minus nominal for distances: positive always
means more nominal-like. High likeness to both populations indicates overlapping
support; low likeness to both can indicate unfamiliar experience. Interpret
"high likeness" as low values when using distances.

The `rolling` command embeds a query trajectory's windows and writes a tidy time
series. Python `rolling_likeness` supports multiple metrics and named reference
libraries; `reference_likeness` returns both components and the margin. Plotting
is separate from these APIs.

### Clustering and external validity

HDBSCAN is the default, using `sklearn.cluster.HDBSCAN` with precomputed IDK
distances. DBSCAN uses the same distance option. `--distance js` uses
`sqrt(mean partition JS)` as a distance. Spectral clustering uses native IDK
similarity as precomputed affinity and requires a chosen `--n-clusters`.
`dpgmm` applies sparse `TruncatedSVD` then `BayesianGaussianMixture` with a
Dirichlet-process-style weight prior. Choose `--svd-components` and
`--n-components` appropriate to the population size; invalid sizes are rejected.

Outputs retain cluster IDs, noise flags (`-1`), HDBSCAN membership strengths where
available, or mixture component posteriors and reduced coordinates. These are
estimator-specific quantities, not calibrated failure probabilities. Mixture
weights, means, covariances and SVD components are saved in `components.npz`.

Diagnostics include silhouette on the density methods' distance matrix (excluding
noise), adjusted mutual information and purity for `--categorical` fields, and
per-cluster count/mean/median/population-standard-deviation of episode return plus
category composition. Undefined silhouette is null. Spectral/mixture runs report
null silhouette because they do not cluster using a precomputed distance matrix.
Window summaries weight each window, so longer episodes can contribute more units.

### Population comparison

`population_mmd` compares **populations of IDK embedding vectors** using an RBF
kernel, `exp(-||mu_i-mu_j||²/(2*sigma²))`. This is distinct from comparing two
individual experiences with IDK distance. The default statistic is biased MMD
squared (diagonals included). `--estimator unbiased` excludes diagonals, requires
two units per group, and may return a negative finite-sample estimate.

The default bandwidth is the median positive pooled Euclidean distance, with
sigma=1 when all vectors coincide; `--bandwidth` supplies an explicit sigma.
The actual bandwidth is reported and held fixed during permutations.
`--permutations` enables a seeded permutation test with `(extreme+1)/(B+1)` p-value.
Permutations exchange embedding rows, requiring exchangeable samples for inference.
Overlapping windows are dependent: use whole independent trajectories or an
appropriate independently sampled design for inferential conclusions. No automatic
block/trajectory-level permutation scheme is imposed.

### Commands and artifacts

```bash
cartpole-analyze pairwise datasets/experiment \
  --fit-ids-file splits/fit.txt --analysis-ids-file splits/test.txt \
  --mode window --window-length 25 --stride 5 --representation transition \
  --metric idk-distance --output analysis/pairwise

cartpole-analyze neighbors datasets/experiment \
  --fit-ids-file splits/fit.txt --analysis-ids-file splits/test.txt \
  --reference-ids-file splits/reference.txt --metric cosine --k 5 \
  --exclude-same-trajectory --output analysis/neighbors

cartpole-analyze rolling datasets/experiment \
  --fit-ids-file splits/fit.txt --query-id TRAJECTORY_ID \
  --window-length 25 --stride 1 --representation transition \
  --nominal-reference-ids-file splits/nominal.txt \
  --failure-reference-ids-file splits/failure.txt --output analysis/rolling

cartpole-analyze cluster datasets/experiment \
  --fit-ids-file splits/fit.txt --analysis-ids-file splits/test.txt \
  --mode window --window-length 25 --stride 5 --algorithm hdbscan \
  --min-cluster-size 20 --distance idk-distance --output analysis/clusters

cartpole-analyze population datasets/experiment \
  --fit-ids-file splits/fit.txt --group-a-ids-file splits/nominal_test.txt \
  --group-b-ids-file splits/failure_test.txt --mode whole \
  --metric rbf-mmd --permutations 1000 --output analysis/population
```

Use `cartpole-analyze COMMAND --help` for the complete options. The analytics
CLI uses Click, matching the other project entrypoints.

Every run requires a new/empty output directory and records `config.json`,
`summary.json`, `units.parquet`, sparse `embeddings.npz`, and numeric `basis.npz`.
The unit manifest is deduplicated across fitting/analysis/reference roles;
`embedding_row` identifies its sparse matrix row and `config.json` records ordered
unit IDs for every role. `pairwise_axes.json` identifies pairwise row/column order.
Config includes resolved trajectory IDs, source files, scaler settings, basis
identity, metric/algorithm parameters, random state, and package versions.
Only command-relevant results are written (Parquet tables, numeric arrays, or
small JSON summaries). Embeddings are reused within a run without refitting.
Reproducibility assumes source dataset contents remain unchanged.

Pairwise methods and clustering need quadratic memory. The default
`--max-pairs 4000000` guard limits any dense result to four million float entries
(32 MB per array; intermediate arrays and estimators can require several times
more). Increase stride, restrict ID selections, or explicitly raise the limit.
MMD applies the limit to the pooled `(n_a+n_b)²` matrix. JS/KL process partition
blocks without densifying the complete embedding matrix.

Optional helpers in `cartpole_idk.visualization.analytics` plot pairwise heatmaps,
rolling scores, cluster composition, or a two-dimensional sparse SVD projection.
They require matplotlib from the development/notebook extras. Display projections
are exploratory illustrations, not quantitative evidence of cluster quality.

[Back to Top](#cartpole-idk)

## Tests

```bash
pytest
```

[Back to Top](#cartpole-idk)
