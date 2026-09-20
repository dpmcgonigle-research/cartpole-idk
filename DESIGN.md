# Design

`pyidk` owns reusable kernel machinery.

`cartpole_idk` owns:
- policy training,
- CartPole data generation,
- perturbations,
- trajectory storage/provenance,
- application-specific feature assembly,
- familiarity experiments,
- visualization.

For each interval `t -> t+1`:
- `true_observations[t]` is the environment state before action.
- `agent_observations[t]` is what the policy sees.
- `commanded_actions[t]` is the policy output.
- `executed_actions[t]` is what reaches the environment.
- `true_observations[t+1]` is the resulting state.

Initial perturbations attack different parts of the loop:
- action delay: timing/control mismatch,
- action flip: actuator/control corruption,
- observation bias: sensing corruption.

The initial IDK normalization baseline is training-only standardization.

Future work:
- threshold calibration and lead-time metrics,
- explicit split manifests,
- perturbation duration/ramping,
- environment-dynamics perturbations,
- checkpoint selection by achieved return.

## Analytics

`analytics/units.py` constructs episode-bounded whole or window analysis units.
Windows count transitions, preserving T+1 observations. Source intervals are
half-open; prepared segments retain original offsets and censoring information.
No success/failure label is invented. Metadata is used only after embedding or
explicitly to select populations outside metric functions.

`idk.fit_sequence_batch()` is shared by existing `fit_reference()` and the new
analytics fitting orchestration. All representations still use
`build_sequence_batch()`. Fitting, transformation, retrieval reference libraries,
and queries are separate roles; only fitting values construct the standardizer
and pyidk basis. In window mode, overlapping fitting windows weight repeated
observations multiple times; fit composition and stride are explicit research choices.

`analytics/embeddings.py` attaches fitted-space identity to sparse embeddings.
`metrics.py` delegates native similarity to pyidk and implements derived distances
and occupancy comparisons. Partition-wise JS/KL retain an outside cell. A small
metric registry states ranking direction. `neighbors.py` provides matches,
aggregate scores, rolling tables, and separate nominal/failure likeness.

Scikit-learn owns clustering and dimensionality reduction. Density methods use
precomputed IDK or square-root JS distance; spectral uses native IDK affinity;
DP-style mixtures operate on sparse SVD coordinates. External labels only enter
post-clustering diagnostics. Population RBF-MMD operates on the next hierarchy:
a population of trajectory/window embeddings. Permutation units are rows; inferential
use requires exchangeability, and overlapping windows do not generally satisfy it.

`workflow.py` caches embeddings per unit within one run; `reporting.py` persists
deduplicated sparse embeddings, explicit role/axis ordering, fitted numeric basis,
configuration and software versions. Dense quadratic outputs have a configurable
size guard. No serialized Python pickle is required for run artifacts.

The analytics CLI uses Click with shared option decorators and five focused
subcommands, matching the other project entrypoints.
The optional suggested `all` command is omitted: researchers explicitly choose
which analyses to run rather than incur every quadratic calculation.
