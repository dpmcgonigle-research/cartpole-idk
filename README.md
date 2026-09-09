# cartpole-idk

A research harness for studying Isolation Distributional Kernel (IDK) on CartPole
before moving to embodied robotics tasks.

This repository does **not** implement IDK. The generic implementation lives in the
separate `pyidk` package and is consumed here as a dependency.

## Research flow

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

## Generate trajectories

Nominal:

```bash
cartpole-generate   --checkpoint runs/dqn_seed42/checkpoints/step_000200000.pt   --output datasets/nominal   --episodes 100
```

Action delay:

```bash
cartpole-generate   --checkpoint runs/dqn_seed42/checkpoints/step_000200000.pt   --output datasets/action_delay   --episodes 100   --perturbation action-delay   --onset-mean 50   --onset-std 5   --delay-steps 3
```

Action flip:

```bash
cartpole-generate   --checkpoint runs/dqn_seed42/checkpoints/step_000200000.pt   --output datasets/action_flip   --episodes 100   --perturbation action-flip   --onset-mean 75   --onset-std 8   --flip-probability 0.15
```

Observation bias:

```bash
cartpole-generate   --checkpoint runs/dqn_seed42/checkpoints/step_000200000.pt   --output datasets/angle_bias   --episodes 100   --perturbation observation-bias   --feature pole_angle   --onset-mean 60   --onset-std 5   --bias 0.04   --noise-std 0.01   --ramp-steps 15
```

## Stored trajectory semantics

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

## Query

```bash
cartpole-query datasets/action_delay   --perturbation action_delay   --max-return 250
```

## Replay

```bash
cartpole-replay datasets/action_delay --trajectory <trajectory-id>
```

Replay uses recorded true states rather than re-executing actions.

## IDK representations

The application adapter supports:

- state
- transition
- state-action
- state-action-next-state
- temporal window

Normalization uses `pyidk.Standardizer`, fitted only on reference/training experience.

## Tests

```bash
pytest
```
