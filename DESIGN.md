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
- rolling-window familiarity,
- threshold calibration and lead-time metrics,
- explicit split manifests,
- perturbation duration/ramping,
- environment-dynamics perturbations,
- checkpoint selection by achieved return.
