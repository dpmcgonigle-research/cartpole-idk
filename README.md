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
- [Stored Trajectory Semantics](#stored-trajectory-semantics)
- [Query](#query)
- [Replay](#replay)
- [IDK Representations](#idk-representations)
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
cartpole-generate   --checkpoint runs/dqn_seed42/checkpoints/step_000200000.pt   --output datasets/nominal   --episodes 100
```

Action delay:

```bash
cartpole-generate   --checkpoint runs/dqn_seed42/checkpoints/step_000200000.pt   --output datasets/action_delay   --episodes 100   --perturbation action-delay   --onset-mean 50   --onset-std 5   --delay-steps 3
```

Action flip:

```bash
cartpole-generate   --checkpoint runs/dqn_seed42/checkpoints/step_000200000.pt   --output datasets/action_flip   --episodes 100   --perturbation action-flip   --onset-mean 75   --onset-std 8   --flip-probability 0.5
```

Observation bias:

```bash
cartpole-generate   --checkpoint runs/dqn_seed42/checkpoints/step_000200000.pt   --output datasets/angle_bias   --episodes 100   --perturbation observation-bias   --feature pole_angle   --onset-mean 60   --onset-std 5   --bias 0.24   --noise-std 0.01   --ramp-steps 15
```

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
cartpole-query datasets/action_delay   --perturbation action_delay   --max-return 250
```

[Back to Top](#cartpole-idk)

## Replay

```bash
cartpole-replay datasets/action_delay --trajectory <trajectory-id>
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

## Tests

```bash
pytest
```

[Back to Top](#cartpole-idk)
