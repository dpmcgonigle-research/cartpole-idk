## 09/22/2026 - 10/06/2026 Plan
- Created [cartpole-idk-experiments.docx](docs/cartpole-idk-experiments.docx)
- Completed remaining section 1 experiments
    - Verified that 2 populations of nominal trajectories are inseparable
        - [nom-v-nom-4d-state.ipynb](notebooks/1-sanity-check/nom-v-nom-4d-state.ipynb)
        - [nom-v-nom-8d-transitions.ipynb](notebooks/1-sanity-check/nom-v-nom-8d-transitions.ipynb)
    - Scrambled the timesteps of one nominal population and verified that the 4-d state is inseparable, but the 8-d transition state is separable
        - [nom-v-nom-scrambled.ipynb](notebooks/1-sanity-check/nom-v-nom-scrambled.ipynb)
    - Added progressive noise to one of the dimensions to erify that the similarity monotonically decreases
        - [progressive_noise_similarity.ipynb](notebooks/1-sanity-check/progressive_noise_similarity.ipynb)

## 09/22/2026 Advising Notes
- Discussed moving forward with the plan notes in 09/22 - 09/29 section

## 09/22/2026 - 09/29/2026 Plan
- Create a more meaningful separability test than the initial sanity check.  Likely something like:
    - Perfect play vs. perfect play + perturbations
    - Perfect play with varied initial conditions vs less than perfect play (vs perturbations?)

## 09/15/2026 - 09/22/2026 Progress
- Completed modular harness for IDK + cartpole
    - `cartpole-train` saves DQN checkpoints
    - `cartpole-generate` saves off cartpole episodes generated from DQN checkpoints
    - `cartpole-prepare` filters episodes, down-selects mini-episodes, and does other pre-processing
    - `cartpole-fit` fits an IDK basis to the prepared datasets
    - `cartpole-embed` takes unseen episodes and generates "occupancy vectors"
    - `cartpole-analyze` runs various analytics to better understand the results
- Ran simple "sanity check" experiment; fitted basis on 500 "perfect play" trajectories.  Embedded a 200-trajectory dataset with 100 "perfect play" and 100 failures.
    - 4D "vanilla state": https://github.com/dpmcgonigle-research/cartpole-idk/blob/main/notebooks/cluster-example-4d-state.ipynb
    - 8D temporal state transitions: https://github.com/dpmcgonigle-research/cartpole-idk/blob/main/notebooks/cluster-example-8d-transitions.ipynb
    - The "perfect play" episodes all did pretty much the exact same thing (with *tiny* fluctuations), so it wasn't super informative, but I was at least able to cleanly separate the two types of behaviors.