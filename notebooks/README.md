#   Notebooks

##  Description
See [cartpole-idk-experiments.docx](../docs/cartpole-idk-experiments.docx) for information about how these notebooks are organized.

##  Sections
1. Initial Sanity Check experiments -- preliminary experiments to validate the IDK representations for cartpole.
2. IDK familiarity experiments -- dial in representations and parameters.
3. Competence measurement experiments -- try to discern varying levels of play.
4. Failure measurement experiments -- try to discern varying levels of failure.
5. Mixed cause failure experiments -- broaden the failure cases.
6. Generalization experiments -- extend evaluation to unseen conditions.

## IDK familiarity notebooks

- [Non-IDK nominal baselines](2-familiarity/idk-familiarity-nonidk-baselines.ipynb): compact randomized-start comparison of fixed 10-neighbor distance and nominal-selected GMM likelihood against saved IDK outside mass and familiarity at ψ=64, t=200. Uses state/transition representations, 25/50-transition windows, and the existing nominal-calibrated alarm protocol. Includes executed results and interpretation; requires the randomized-start datasets, focus fits, and numerical cache. Exports tables and provenance to `artifacts/analysis/nonidk-baselines`.
- [Nominal vs action delay](2-idk-familiarity/nominal-vs-action-delay.ipynb): compares nominal and delayed populations across 30 state/transition bases and four window sizes. Includes population metrics, score curves, detection rates, false alarms, latency, and parameter sensitivity. Uses nominal episodes to calibrate alarms. Requires original source datasets and fit artifacts. Saves tables and figures under `artifacts/analysis/idk-familiarity-action-delay`.

## Prospective failure prediction

- [Randomized-start failure prediction](4-single-failure/idk-failure-prediction-randinit.ipynb): H=50 representation experiment with separate delay-severity RBF SVMs, nested episode-grouped tuning, and four inputs: current state, matched raw history, compact IDK, and history + IDK. The 5D IDK vector includes a lagged familiarity difference, so its matched raw context is 30 transitions / 31 state endpoints (124D). Reports AUPRC, AUROC, paired fold differences, compression tradeoffs, and a frozen fresh-data check. Reuses the randomized-start trajectories, focus fits, and familiarity score cache; versioned exports live under `artifacts/analysis/idk-failure-prediction-randinit/history-rbf`.
