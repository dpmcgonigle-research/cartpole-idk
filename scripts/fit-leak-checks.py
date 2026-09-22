from pathlib import Path
import hashlib
import json
import numpy as np


def observation_hash(path: Path) -> str:
    with np.load(path, allow_pickle=False) as z:
        x = np.ascontiguousarray(z["true_observations"])

    h = hashlib.sha256()
    h.update(str(x.shape).encode())
    h.update(str(x.dtype).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def load_info(directory: str):
    result = {}

    for path in sorted(Path(directory).glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            traj_id = str(z["trajectory_id"])
            obs = z["true_observations"]

            metadata = json.loads(str(z["metadata_json"]))

        result[path] = {
            "hash": observation_hash(path),
            "trajectory_id": traj_id,
            "shape": obs.shape,
            "metadata": metadata,
        }

    return result


#   GENERATED
#fit_dir = "datasets/generated/200000-iter_nominal/500_episodes/trajectories"
#holdout_dir = "datasets/generated/200000-iter_nominal/100_episodes/trajectories"
#   PREPARED
fit_dir = "datasets/prepared/200000-iter_nominal_500_episodes/trajectories"
holdout_dir = "datasets/prepared/10K-natfail-100ep_200K-nom-100ep/trajectories"

fit = load_info(fit_dir)
holdout = load_info(holdout_dir)

fit_hashes = {v["hash"]: p for p, v in fit.items()}

matches = []

for p, info in holdout.items():
    if info["hash"] in fit_hashes:
        matches.append((fit_hashes[info["hash"]], p))

print("Exact duplicated trajectories:", len(matches))

for a, b in matches[:20]:
    print(a)
    print(b)

def extract_seed(info):
    md = info["metadata"]

    for key in (
        "environment_seed",
        "env_seed",
        "seed",
    ):
        if key in md:
            return md[key]

    return None


fit_seeds = {
    extract_seed(v)
    for v in fit.values()
    if extract_seed(v) is not None
}

holdout_seeds = {
    extract_seed(v)
    for v in holdout.values()
    if extract_seed(v) is not None
}

print("Overlapping environment seeds:",
      sorted(fit_seeds & holdout_seeds))

near_matches = []

for hp, hinfo in holdout.items():
    with np.load(hp, allow_pickle=False) as z:
        hx = z["true_observations"]

    for fp, finfo in fit.items():
        if finfo["shape"] != hx.shape:
            continue

        with np.load(fp, allow_pickle=False) as z:
            fx = z["true_observations"]

        if np.allclose(hx, fx, rtol=1e-7, atol=1e-8):
            near_matches.append((fp, hp))

print("Near-identical trajectories:", len(near_matches))

