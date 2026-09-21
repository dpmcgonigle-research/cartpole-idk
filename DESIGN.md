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

## Explicit IDK stages

`pyidk` owns the generic kernels. `idk/units.py` constructs episode-bounded raw
segments and `idk/features.py` retains all representation assembly. `idk/pipeline.py`
provides explicit `fit_dataset()` and `embed_dataset()` operations. Fitting learns
only from its selected feature rows; embedding uses the saved representation,
standardizer and isolation basis without fitting or resampling.

Pydantic v2 configuration and metadata contracts live in `model.py`, separate from
large numeric arrays. The legacy `IDKExperimentConfig` name aliases `IDKConfig`.
Configurations are frozen, validated and directly JSON-serializable. Dynamic
trajectory metadata remains a metadata dictionary, not a configuration mechanism.

`artifacts/FitArtifact` persists config, scaler arrays, basis arrays, fitting-unit
provenance and versioned metadata. Loading assigns the public `pyidk.IsolationBasis`
state directly and restores `Standardizer` arrays; no fit call occurs.
`EmbeddingArtifact` stores sparse CSR embeddings, ordered unit records, configuration,
parent fit identity and metadata. Unit provenance has optional raw arrays during
construction; loading artifacts leaves those arrays absent.

Artifact version 1 includes file checksums and a SHA-256 content identity. Loaders
validate file presence/integrity, configuration, row order, representation width,
scaler scales, basis dimensions/radii/sample indices, and sparse occupancy constraints.
Embedding comparisons validate common fit identity and representation provenance.
Artifact writes use staging directories followed by atomic publication; no pickle.

`analytics` consumes existing embeddings only. Its workflow has no raw dataset,
fit or transform operations. Its typed `AnalysisConfig` selects artifacts for
query/reference/nominal/failure/group roles. Reports record paths and content hashes,
configuration and role ordering, without copying fitted models or embeddings.
Raw data and parent fit directories can be offline during analysis.

The Click pipeline is `cartpole-fit -> cartpole-embed -> cartpole-analyze`.
The obsolete `cartpole-idk-evaluate` command is removed; its retrieval functionality
is provided by `analyze neighbors`. Explicit low-level in-memory fitting helpers
remain in `idk` for Python users, without a competing analytics workflow.

Units count transitions and retain T+1 observations. Start/end intervals are
half-open, source offsets survive preparation, and censored prefixes do not acquire
invented failure times. Whole and window embeddings use the same metrics. Overlapping
fitting windows weight repeated rows; selection and stride remain research choices.

Native similarity delegates to pyidk. JS/KL preserve outside occupancy; metric
ranking direction is explicit. Scikit-learn owns clustering/reduction, and known
labels enter only post-clustering diagnostics. RBF-MMD operates on populations of
IDK embeddings; row-wise permutation inference requires exchangeability, which
rolling windows do not generally provide. Dense pairwise operations have size guards.
